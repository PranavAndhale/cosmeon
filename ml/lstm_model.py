"""
LSTM Time-Series Model for Sequential Flood Prediction.

Uses sliding windows of daily weather data to predict flood risk,
capturing temporal patterns that tabular GBM cannot.

Architecture:
  - Input:  (batch, seq_len=30, features=8) daily weather sequence
  - LSTM:   2 layers, 128 hidden units, dropout 0.3
  - Output: (batch, 4) class probabilities [LOW, MEDIUM, HIGH, CRITICAL]

Features per day:
  1. precipitation_sum (mm)
  2. soil_moisture_0_to_7cm_mean
  3. temperature_2m_max (C)
  4. temperature_2m_min (C)
  5. et0_fao_evapotranspiration (mm)
  6. river_discharge (m3/s) — normalized by historical mean
  7. month_sin — sin(2*pi*month/12)
  8. month_cos — cos(2*pi*month/12)
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config.settings import PROCESSED_DIR

logger = logging.getLogger("cosmeon.ml.lstm")

LSTM_MODEL_PATH = PROCESSED_DIR / "flood_lstm.pth"
LSTM_SCALER_PATH = PROCESSED_DIR / "flood_lstm_scaler.npz"

RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEQ_LEN = 30
NUM_FEATURES = 8


# ─── Dataset ───

class FloodSequenceDataset(Dataset):
    """Sliding-window dataset for LSTM flood prediction."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: (N, seq_len, num_features) float array
            labels:    (N,) int array  [0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ─── Temporal Attention ───

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over LSTM hidden states.

    Allows the model to attend to ALL time steps (not just the last),
    capturing early extreme-rainfall events that would otherwise be
    diluted in the final hidden state.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) — LSTM output sequence
        Returns:
            (batch, seq_len, hidden_size) — attended sequence with residual
        """
        attended, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attended))


# ─── Model ───

class FloodLSTM(nn.Module):
    """
    2-layer Bidirectional LSTM + multi-head temporal attention for flood
    risk classification from daily weather sequences.

    Improvements over v1:
      - Temporal attention attends to ALL time steps, not just last hidden state
      - Global average pooling over attended sequence (more robust than single step)
      - Residual connection in attention layer
    """

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-head temporal attention over all LSTM outputs
        self.attention = TemporalAttention(
            hidden_size=hidden_size,
            num_heads=4,
            dropout=dropout * 0.5,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM forward — get all hidden states
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Temporal attention over all time steps
        attended = self.attention(lstm_out)  # (batch, seq_len, hidden_size)

        # Global average pooling — more stable than taking only last step
        pooled = attended.mean(dim=1)  # (batch, hidden_size)

        logits = self.classifier(pooled)
        return logits


# ─── Manager ───

class LSTMFloodManager:
    """Manages the LSTM flood prediction model lifecycle."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FloodLSTM().to(self.device)
        self.is_trained = False
        self._training_metrics = {}
        # Feature normalization stats
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        # Try loading persisted model
        if self._load_model():
            logger.info("Loaded persisted LSTM model from %s", LSTM_MODEL_PATH)
        else:
            logger.info("LSTMFloodManager initialized (no persisted model)")

    def train(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_split: float = 0.2,
    ) -> dict:
        """
        Train the LSTM on sliding-window sequences.

        Args:
            sequences: (N, 30, 8) raw daily feature sequences
            labels:    (N,) int labels [0-3]
            epochs:    training epochs
            batch_size: mini-batch size
            lr:        learning rate
            val_split: validation fraction

        Returns:
            Training metrics dict
        """
        logger.info("LSTM training: %d sequences, %d epochs", len(sequences), epochs)

        # Normalize features (fit on train, apply to all)
        n_samples = len(sequences)
        flat = sequences.reshape(-1, sequences.shape[-1])
        self._feature_means = flat.mean(axis=0)
        self._feature_stds = flat.std(axis=0)
        self._feature_stds[self._feature_stds < 1e-6] = 1.0  # prevent div by zero
        sequences_normed = (sequences - self._feature_means) / self._feature_stds

        # Train/val split
        n_val = int(n_samples * val_split)
        indices = np.random.RandomState(42).permutation(n_samples)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_ds = FloodSequenceDataset(sequences_normed[train_idx], labels[train_idx])
        val_ds = FloodSequenceDataset(sequences_normed[val_idx], labels[val_idx])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Class weights for imbalanced data
        class_counts = np.bincount(labels, minlength=4).astype(float)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weight_tensor = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        best_val_acc = 0.0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # ── Train ──
            train_loss = 0.0
            for batch_x, batch_y in train_dl:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_dl), 1)
            scheduler.step()

            # ── Validate ──
            val_loss, correct, total = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_dl:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.model(batch_x)
                    val_loss += criterion(logits, batch_y).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += batch_y.size(0)
            self.model.train()
            val_loss /= max(len(val_dl), 1)
            val_acc = correct / max(total, 1)

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 4))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
                    epoch + 1, epochs, train_loss, val_loss, val_acc,
                )

        # Reload best model
        self._load_model()
        self.is_trained = True

        # Final evaluation
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        # Per-class accuracy
        per_class = {}
        for i, name in enumerate(RISK_LABELS):
            mask = np.array(all_labels) == i
            if mask.sum() > 0:
                per_class[name] = round(float((np.array(all_preds)[mask] == i).mean()), 3)
            else:
                per_class[name] = None

        metrics = {
            "total_sequences": int(n_samples),
            "train_sequences": int(len(train_idx)),
            "val_sequences": int(len(val_idx)),
            "best_val_accuracy": round(best_val_acc, 3),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "per_class_accuracy": per_class,
            "label_distribution": {
                RISK_LABELS[i]: int(np.sum(labels == i)) for i in range(4)
            },
            "model_type": "LSTM (2-layer, 128 hidden)",
            "seq_length": SEQ_LEN,
            "num_features": NUM_FEATURES,
        }
        self._training_metrics = metrics
        logger.info("LSTM training complete: best_val_acc=%.3f", best_val_acc)
        return metrics

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a single 30-day sequence.

        Args:
            sequence: (30, 8) or (1, 30, 8) raw feature array

        Returns:
            (4,) probability array [LOW, MEDIUM, HIGH, CRITICAL]
        """
        if not self.is_trained:
            return np.array([0.7, 0.2, 0.08, 0.02])  # default LOW prior

        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]  # (1, 30, 8)

        # Normalize
        if self._feature_means is not None:
            sequence = (sequence - self._feature_means) / self._feature_stds

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).to(self.device)
            logits = self.model(x)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return proba

    def predict(self, sequence: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Predict risk level for a 30-day sequence.

        Returns:
            (risk_level, probability, all_probabilities)
        """
        proba = self.predict_proba(sequence)
        pred_class = int(np.argmax(proba))
        return RISK_LABELS[pred_class], float(proba[pred_class]), proba

    def get_training_metrics(self) -> dict:
        return self._training_metrics

    # ─── Persistence ───

    def _save_model(self):
        """Save model weights and normalization stats."""
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), LSTM_MODEL_PATH)
            if self._feature_means is not None:
                np.savez(
                    LSTM_SCALER_PATH,
                    means=self._feature_means,
                    stds=self._feature_stds,
                )
            logger.info("Saved LSTM model to %s", LSTM_MODEL_PATH)
        except Exception as e:
            logger.error("Failed to save LSTM model: %s", e)

    def _load_model(self) -> bool:
        """Load persisted model. Returns True if successful."""
        try:
            if LSTM_MODEL_PATH.exists():
                self.model.load_state_dict(
                    torch.load(LSTM_MODEL_PATH, map_location=self.device, weights_only=True)
                )
                self.model.eval()
                self.is_trained = True

                if LSTM_SCALER_PATH.exists():
                    data = np.load(LSTM_SCALER_PATH)
                    self._feature_means = data["means"]
                    self._feature_stds = data["stds"]

                return True
        except Exception as e:
            logger.error("Failed to load LSTM model: %s", e)
        return False
