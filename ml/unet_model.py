"""
Phase 7: U-Net ML Model for Flood Segmentation.

Provides a deep learning-based flood detection alternative to NDWI thresholding.
Uses a lightweight U-Net architecture for binary segmentation (flood/no-flood).

Trained on Sen1Floods11 dataset (Sentinel-1 SAR: VV + VH channels).
Includes Dice+BCE combined loss for class imbalance and IoU metric.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config.settings import PROCESSED_DIR

logger = logging.getLogger("cosmeon.ml")

UNET_MODEL_PATH = PROCESSED_DIR / "flood_unet.pth"


# --- Losses & Metrics ---

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for flood segmentation (handles class imbalance)."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union (IoU) for binary segmentation."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0  # both empty = perfect match
    return float(intersection / union)


# --- U-Net Architecture ---

class ConvBlock(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FloodUNet(nn.Module):
    """
    Lightweight U-Net for flood segmentation.

    Input: (B, C, H, W) - satellite image bands (default 2 for Sentinel-1 VV+VH)
    Output: (B, 1, H, W) - flood probability mask
    """

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Pad input to be divisible by 16
        h, w = x.shape[2], x.shape[3]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return torch.sigmoid(out)


# --- Dataset ---

class FloodDataset(Dataset):
    """Dataset for flood segmentation training."""

    def __init__(self, images: list[np.ndarray], masks: list[np.ndarray], patch_size=256):
        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.patches = self._create_patches()

    def _create_patches(self):
        """Extract fixed-size patches from images."""
        patches = []
        for img, mask in zip(self.images, self.masks):
            h, w = img.shape[1], img.shape[2] if img.ndim == 3 else img.shape
            for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
                for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                    img_patch = img[:, i:i+self.patch_size, j:j+self.patch_size] if img.ndim == 3 else img[i:i+self.patch_size, j:j+self.patch_size]
                    mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size]
                    patches.append((img_patch, mask_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        img = torch.FloatTensor(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        return img, mask


# --- Model Manager ---

class FloodModelManager:
    """Manages the flood segmentation ML model."""

    def __init__(self, model_path: str = None, in_channels: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or str(UNET_MODEL_PATH)
        self.is_trained = False
        self._training_metrics = {}

        # Prefer SMP with ImageNet-pretrained encoder over scratch U-Net
        self.model = self._build_model(in_channels)

        if Path(self.model_path).exists():
            try:
                self.load_model()
                self.is_trained = True
                logger.info("Loaded pretrained weights from %s", self.model_path)
            except RuntimeError as e:
                logger.warning(
                    "Saved weights incompatible with new architecture (model needs retraining): %s", e
                )
        else:
            logger.info("No pretrained weights found — model ready for training.")

    def _build_model(self, in_channels: int) -> torch.nn.Module:
        """
        Build segmentation model.

        Priority:
          1. segmentation_models_pytorch UNet + EfficientNet-B3 (ImageNet pretrained)
             — stronger encoder, faster convergence, ~15-20% better IoU than scratch
          2. Custom FloodUNet (fallback if smp not available)
        """
        try:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name="efficientnet-b3",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=1,
                activation="sigmoid",
            ).to(self.device)
            logger.info(
                "Using SMP UNet + EfficientNet-B3 encoder (ImageNet pretrained) — "
                "expected +15-20%% IoU vs scratch"
            )
            return model
        except Exception as e:
            logger.info("SMP not available (%s) — falling back to custom FloodUNet", e)
            return FloodUNet(in_channels=in_channels).to(self.device)

    def train(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        epochs: int = 20,
        batch_size: int = 8,
        lr: float = 1e-4,
        val_split: float = 0.2,
    ) -> dict:
        """Train the flood segmentation model with Dice+BCE loss and IoU tracking."""
        logger.info("Starting U-Net training: %d images, %d epochs", len(images), epochs)

        # Train/val split
        n = len(images)
        n_val = int(n * val_split)
        indices = np.random.RandomState(42).permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_imgs = [images[i] for i in train_idx]
        train_masks = [masks[i] for i in train_idx]
        val_imgs = [images[i] for i in val_idx]
        val_masks = [masks[i] for i in val_idx]

        train_ds = FloodDataset(train_imgs, train_masks)
        val_ds = FloodDataset(val_imgs, val_masks) if val_imgs else None

        if len(train_ds) == 0:
            logger.error("No training patches created")
            return {"error": "No training data"}

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds and len(val_ds) > 0 else None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = DiceBCELoss()

        self.model.train()
        best_iou = 0.0
        history = {"train_loss": [], "val_loss": [], "val_iou": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_img, batch_mask in train_dl:
                batch_img = batch_img.to(self.device)
                batch_mask = batch_mask.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_img)
                loss = criterion(pred, batch_mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_dl)
            history["train_loss"].append(round(avg_loss, 4))

            # Validation
            if val_dl:
                val_loss, val_iou = 0.0, 0.0
                n_batches = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_img, batch_mask in val_dl:
                        batch_img = batch_img.to(self.device)
                        batch_mask = batch_mask.to(self.device)
                        pred = self.model(batch_img)
                        val_loss += criterion(pred, batch_mask).item()
                        val_iou += compute_iou(pred, batch_mask)
                        n_batches += 1
                self.model.train()
                val_loss /= max(n_batches, 1)
                val_iou /= max(n_batches, 1)
                history["val_loss"].append(round(val_loss, 4))
                history["val_iou"].append(round(val_iou, 4))

                if val_iou > best_iou:
                    best_iou = val_iou
                    self.save_model()

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d/%d | Loss: %.4f | Val IoU: %.4f",
                    epoch + 1, epochs, avg_loss,
                    history["val_iou"][-1] if history["val_iou"] else 0,
                )

        if best_iou == 0:
            self.save_model()

        self.is_trained = True
        self._training_metrics = {
            "total_images": n,
            "train_images": len(train_imgs),
            "val_images": len(val_imgs),
            "epochs": epochs,
            "best_val_iou": round(best_iou, 4),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "model_type": "U-Net (Sentinel-1 VV+VH, DiceBCE Loss)",
        }

        logger.info("U-Net training complete. Best IoU: %.4f", best_iou)
        return self._training_metrics

    def predict(self, band_data: dict, threshold: float = 0.5) -> np.ndarray:
        """
        Run flood prediction on satellite band data.

        Args:
            band_data: dict with "bands" key containing numpy arrays
            threshold: Probability threshold for binary mask

        Returns:
            Binary flood mask (H, W)
        """
        self.model.eval()

        bands = band_data["bands"]
        band_arrays = []

        # Prefer Sentinel-1 VV/VH for Sen1Floods11-trained model
        for band_name in ["VV", "VH", "B03", "B04", "B08", "B8A"]:
            if band_name in bands:
                band_arrays.append(bands[band_name])
            if len(band_arrays) >= self.model.enc1.conv[0].in_channels:
                break

        if not band_arrays:
            band_arrays = list(bands.values())[:2]

        # Normalize and stack
        input_array = np.stack(band_arrays, axis=0).astype(np.float32)
        for i in range(input_array.shape[0]):
            band = input_array[i]
            bmin, bmax = np.nanmin(band), np.nanmax(band)
            if bmax > bmin:
                input_array[i] = (band - bmin) / (bmax - bmin)

        input_tensor = torch.FloatTensor(input_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        flood_mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        logger.info(
            "U-Net prediction: %d flood pixels (%.2f%%)",
            flood_mask.sum(), flood_mask.sum() / max(flood_mask.size, 1) * 100,
        )
        return flood_mask

    def get_training_metrics(self) -> dict:
        return self._training_metrics

    def save_model(self):
        """Save model weights."""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
