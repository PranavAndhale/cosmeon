"""
Phase 9: Predictive Flood Risk Model.

Uses historical flood data and external factors to forecast
potential flood risk for upcoming periods.

Training data sources (in order of preference):
  1. Real historical weather + river discharge data (Open-Meteo APIs)
  2. Synthetic data as fallback if APIs are unreachable

Improvements over v1:
  - 13 features instead of 7
  - Train/test split for honest evaluation
  - 5-fold cross-validation
  - Model persistence (save/load from disk)
  - Better hyperparameters with regularisation
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from config.settings import PROCESSED_DIR

logger = logging.getLogger("cosmeon.processing.predictor")

MODEL_PATH = PROCESSED_DIR / "flood_predictor.joblib"
SCALER_PATH = PROCESSED_DIR / "flood_scaler.joblib"


@dataclass
class FloodPrediction:
    """Prediction of future flood risk."""
    region_name: str
    prediction_date: str
    predicted_risk_level: str
    flood_probability: float
    confidence: float
    contributing_factors: dict
    model_version: str = "gbm_v2"
    predicted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return {
            "region_name": self.region_name,
            "prediction_date": self.prediction_date,
            "predicted_risk_level": self.predicted_risk_level,
            "flood_probability": self.flood_probability,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "model_version": self.model_version,
            "predicted_at": self.predicted_at,
        }


class FloodPredictor:
    """
    Predicts flood risk using an ensemble of:
      - Gradient Boosting Classifier (13 tabular features) — weight 0.6
      - LSTM (30-day weather sequences) — weight 0.4 (if trained)

    Falls back to GBM-only if LSTM is not available.
    """

    FEATURE_NAMES = [
        "mean_flood_pct", "max_flood_pct", "trend",
        "rainfall_mm", "elevation_m", "month", "risk_multiplier",
        "precip_7d", "precip_30d", "max_daily_rain_7d",
        "precip_anomaly", "soil_moisture", "temperature",
    ]

    GBM_WEIGHT = 0.6
    LSTM_WEIGHT = 0.4

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            subsample=0.8,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        self._training_metrics = {}

        # LSTM model (optional ensemble enhancement)
        self._lstm_manager = None
        try:
            from ml.lstm_model import LSTMFloodManager
            self._lstm_manager = LSTMFloodManager()
            logger.info("LSTM model loaded (ensemble mode available)")
        except Exception as e:
            logger.info("LSTM not available: %s (GBM-only mode)", e)

        # Try loading a persisted model on init
        if self._load_persisted_model():
            logger.info("Loaded persisted predictor model from disk")
        else:
            logger.info("FloodPredictor initialized (no persisted model found)")

    # ── Feature extraction ──

    def _extract_features(
        self,
        flood_history: list[dict],
        external_factors: dict = None,
    ) -> np.ndarray:
        """
        Extract 13 features from historical data for prediction.

        Features 0-6 (original):
          mean flood %, max flood %, trend, rainfall, elevation, month, risk_mult
        Features 7-12 (new):
          precip_7d, precip_30d, max_daily_rain_7d, precip_anomaly, soil_moisture, temperature
        """
        if not flood_history:
            return np.zeros((1, len(self.FEATURE_NAMES)))

        # Extract flood percentages
        flood_pcts = [h.get("flood_percentage", 0) for h in flood_history[-5:]]
        mean_pct = np.mean(flood_pcts) if flood_pcts else 0
        max_pct = np.max(flood_pcts) if flood_pcts else 0

        # Trend (positive = increasing flood risk)
        if len(flood_pcts) >= 2:
            trend = np.polyfit(range(len(flood_pcts)), flood_pcts, 1)[0]
        else:
            trend = 0

        # External factors (original 3)
        ext = external_factors or {}
        rainfall = ext.get("rainfall_mm", 0)
        elevation = ext.get("elevation_mean_m", 100)
        risk_mult = ext.get("risk_multiplier", 1.0)

        # Seasonal (month)
        month = datetime.utcnow().month

        # New features from raw_features (enriched training data)
        precip_7d = ext.get("precip_7d", rainfall)           # fallback to rainfall
        precip_30d = ext.get("precip_30d", rainfall * 4)     # rough estimate
        max_daily_rain = ext.get("max_daily_rain_7d", rainfall / 7 if rainfall else 0)
        precip_anomaly = ext.get("precip_anomaly", 0)
        soil_moisture = ext.get("soil_moisture", 0)
        temperature = ext.get("temperature", 25)

        features = np.array([[
            mean_pct, max_pct, trend,
            rainfall, elevation, month, risk_mult,
            precip_7d, precip_30d, max_daily_rain,
            precip_anomaly, soil_moisture, temperature,
        ]])

        return features

    # ── Training ──

    def train_on_real_data(self, regions: list[dict] = None) -> dict:
        """
        Train the model on REAL historical weather + GloFAS discharge data.

        Labels come from actual river discharge anomalies (ground truth),
        not precipitation heuristics.

        Args:
            regions: list of {"name": str, "lat": float, "lon": float, "elevation": float}
                     If None, uses default flood-prone regions.

        Returns:
            Training metrics dict
        """
        if regions is None:
            regions = [
                {"name": "Bihar, India", "lat": 26.0, "lon": 85.5, "elevation": 55},
                {"name": "Jakarta, Indonesia", "lat": -6.2, "lon": 106.8, "elevation": 8},
                {"name": "Bremen, Germany", "lat": 53.1, "lon": 8.8, "elevation": 12},
                {"name": "Navi Mumbai, India", "lat": 19.1, "lon": 73.0, "elevation": 14},
                {"name": "São Paulo, Brazil", "lat": -23.55, "lon": -46.6, "elevation": 760},
                {"name": "Dhaka, Bangladesh", "lat": 23.75, "lon": 90.4, "elevation": 8},
            ]

        try:
            from processing.real_data_trainer import RealDataTrainer
            trainer = RealDataTrainer()
            real_data = trainer.build_multi_region_training_data(regions)

            if len(real_data) >= 50:
                logger.info("Training on %d REAL data samples (GloFAS ground truth)", len(real_data))
                result = self.train(real_data)
                result["data_source"] = "real_glofas_discharge_ground_truth"
                result["regions_used"] = len(regions)
                return result
            else:
                logger.warning("Only got %d real samples. Falling back to synthetic.", len(real_data))
        except Exception as e:
            logger.error("Failed to train on real data: %s. Falling back to synthetic.", e)

        # Fallback to synthetic
        return self.train(self._generate_synthetic_data(500))

    def train(self, training_data: list[dict]) -> dict:
        """
        Train the predictive model on provided training data.

        Performs:
          - Feature extraction (13 features)
          - 80/20 train/test split
          - 5-fold cross-validation on training set
          - Full model training on training set
          - Evaluation on held-out test set
          - Model persistence to disk

        Args:
            training_data: list of dicts with keys:
                - flood_history: list of past assessments
                - external_factors: dict of external data
                - label: actual risk level that occurred
                - raw_features: (optional) enriched feature dict

        Returns:
            Training metrics including train_acc, test_acc, cv_score, classification_report
        """
        if len(training_data) < 10:
            logger.warning("Insufficient training data (%d samples). Generating synthetic data.", len(training_data))
            training_data = self._generate_synthetic_data(500)

        X = []
        y = []

        for sample in training_data:
            # Merge raw_features into external_factors so predictor can access them
            ext = dict(sample.get("external_factors", {}))
            raw = sample.get("raw_features", {})
            ext.update(raw)  # raw_features override/enrich external_factors

            features = self._extract_features(
                sample.get("flood_history", []),
                ext,
            )
            X.append(features[0])

            label = sample.get("label", "LOW")
            y.append(self.risk_labels.index(label) if label in self.risk_labels else 0)

        X = np.array(X)
        y = np.array(y)

        # ── Train/test split ──
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── Cross-validation on training set ──
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        logger.info(
            "5-fold CV: mean=%.3f (±%.3f) | per-fold=%s",
            cv_scores.mean(), cv_scores.std(),
            [round(s, 3) for s in cv_scores],
        )

        # ── Train on full training set with class weighting ──
        # Upweight rare classes (HIGH/CRITICAL) so model doesn't just predict LOW
        sample_weights = compute_sample_weight('balanced', y_train)
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        self.is_trained = True

        # ── Evaluate ──
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        # Classification report
        y_pred_test = self.model.predict(X_test_scaled)
        report = classification_report(
            y_test, y_pred_test,
            target_names=self.risk_labels,
            output_dict=True,
            zero_division=0,
        )
        report_str = classification_report(
            y_test, y_pred_test,
            target_names=self.risk_labels,
            zero_division=0,
        )

        logger.info(
            "Training complete: %d samples | train_acc=%.3f | test_acc=%.3f | cv=%.3f±%.3f",
            len(X), train_acc, test_acc, cv_scores.mean(), cv_scores.std(),
        )
        logger.info("Classification Report:\n%s", report_str)

        # ── Feature importances ──
        importances = {
            name: round(float(imp), 4)
            for name, imp in zip(self.FEATURE_NAMES, self.model.feature_importances_)
        }
        logger.info("Feature importances: %s", importances)

        # ── Persist model ──
        self._save_persisted_model()

        metrics = {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": round(train_acc, 3),
            "test_accuracy": round(test_acc, 3),
            "cv_mean": round(cv_scores.mean(), 3),
            "cv_std": round(cv_scores.std(), 3),
            "cv_scores": [round(s, 3) for s in cv_scores],
            "feature_importances": importances,
            "classification_report": report,
            "label_distribution": {
                self.risk_labels[i]: int(np.sum(y == i))
                for i in range(len(self.risk_labels))
            },
        }
        self._training_metrics = metrics
        return metrics

    # ── Prediction ──

    def predict(
        self,
        flood_history: list[dict],
        external_factors: dict = None,
        region_name: str = "Unknown",
    ) -> FloodPrediction:
        """
        Predict flood risk using GBM + LSTM ensemble (if LSTM trained).

        Args:
            flood_history: list of recent risk assessment dicts
            external_factors: dict from ExternalDataIntegrator
            region_name: Name of the region
            lat: optional latitude for LSTM sequence fetching
            lon: optional longitude for LSTM sequence fetching

        Returns:
            FloodPrediction with risk forecast
        """
        if not self.is_trained:
            logger.info("Model not trained. Attempting training with real data...")
            self.train_on_real_data()

        features = self._extract_features(flood_history, external_factors)
        features_scaled = self.scaler.transform(features)

        # GBM prediction
        gbm_proba = self.model.predict_proba(features_scaled)[0]

        # LSTM ensemble (if available and trained)
        model_version = "gbm_v2"
        lstm_proba = None
        lat = external_factors.get("_lat") if external_factors else None
        lon = external_factors.get("_lon") if external_factors else None

        if self._lstm_manager and self._lstm_manager.is_trained and lat and lon:
            try:
                from processing.lstm_trainer import LSTMDataBuilder
                builder = LSTMDataBuilder()
                sequence = builder.build_sequence_for_prediction(lat, lon, days=30)
                lstm_proba = self._lstm_manager.predict_proba(sequence)
                model_version = "ensemble_gbm_lstm"
                logger.info("LSTM proba: %s", [round(p, 3) for p in lstm_proba])
            except Exception as e:
                logger.warning("LSTM prediction failed, using GBM only: %s", e)

        # Ensemble combination
        if lstm_proba is not None:
            # Ensure both have 4 classes
            if len(gbm_proba) == 4 and len(lstm_proba) == 4:
                final_proba = self.GBM_WEIGHT * gbm_proba + self.LSTM_WEIGHT * lstm_proba
            else:
                final_proba = gbm_proba
        else:
            final_proba = gbm_proba

        pred_class = int(np.argmax(final_proba))
        predicted_level = self.risk_labels[pred_class]
        flood_probability = float(final_proba[pred_class])

        sorted_proba = sorted(final_proba, reverse=True)
        confidence = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else sorted_proba[0]

        importances = self.model.feature_importances_
        contributing = {
            name: round(float(imp), 3)
            for name, imp in zip(self.FEATURE_NAMES, importances)
        }

        prediction = FloodPrediction(
            region_name=region_name,
            prediction_date=datetime.utcnow().isoformat(),
            predicted_risk_level=predicted_level,
            flood_probability=round(flood_probability, 3),
            confidence=round(float(confidence), 3),
            contributing_factors=contributing,
            model_version=model_version,
        )

        logger.info(
            "Prediction: region=%s | risk=%s | prob=%.3f | conf=%.3f | model=%s",
            region_name, predicted_level, flood_probability, confidence, model_version,
        )
        return prediction

    def predict_by_coords(
        self,
        lat: float,
        lon: float,
        name: str = "Custom Location",
    ) -> FloodPrediction:
        """
        Predict flood risk for arbitrary coordinates (no database required).

        Fetches external factors on the fly and runs the ensemble.
        """
        from processing.external_data import ExternalDataIntegrator
        integrator = ExternalDataIntegrator()
        factors = integrator.get_risk_factors_by_coords(lat, lon)
        factors_dict = factors.to_dict()
        factors_dict["_lat"] = lat
        factors_dict["_lon"] = lon

        return self.predict(
            flood_history=[],
            external_factors=factors_dict,
            region_name=name,
        )

    def get_training_metrics(self) -> dict:
        """Return the most recent training metrics."""
        return self._training_metrics

    def get_lstm_metrics(self) -> dict:
        """Return LSTM training metrics if available."""
        if self._lstm_manager:
            return self._lstm_manager.get_training_metrics()
        return {}

    # ── Explainability ──

    def explain_prediction(
        self,
        flood_history: list[dict],
        external_factors: dict = None,
        region_name: str = "Unknown",
    ) -> dict:
        """
        Generate a fully explainable prediction with feature-level reasoning.

        Returns the prediction PLUS:
          - Raw feature values used (proving no discharge data)
          - Which features drove the prediction most
          - Human-readable explanation text
          - Data source proof (weather-only inputs)
        """
        if not self.is_trained:
            self.train_on_real_data()

        # Extract raw features (these are the 13 weather/terrain inputs)
        features = self._extract_features(flood_history, external_factors)
        features_scaled = self.scaler.transform(features)

        # Get prediction
        pred_class = self.model.predict(features_scaled)[0]
        pred_proba = self.model.predict_proba(features_scaled)[0]
        predicted_level = self.risk_labels[pred_class]
        flood_probability = float(pred_proba[pred_class])

        sorted_proba = sorted(pred_proba, reverse=True)
        confidence = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else sorted_proba[0]

        # Build feature values dict (raw, unscaled)
        raw_values = features[0]
        feature_values = {
            name: round(float(val), 4)
            for name, val in zip(self.FEATURE_NAMES, raw_values)
        }

        # Feature importances from the trained model
        importances = self.model.feature_importances_

        # Build per-feature drivers with human-readable influence
        drivers = []
        for i, name in enumerate(self.FEATURE_NAMES):
            val = float(raw_values[i])
            imp = float(importances[i])
            influence = self._describe_feature_influence(name, val, predicted_level)
            drivers.append({
                "feature": name,
                "value": round(val, 4),
                "importance": round(imp, 4),
                "influence": influence,
            })

        # Sort by importance (highest first)
        drivers.sort(key=lambda d: d["importance"], reverse=True)

        # Generate human-readable explanation
        explanation = self._generate_explanation_text(
            predicted_level, flood_probability, confidence, drivers[:5], feature_values
        )

        # Class probabilities
        class_probabilities = {
            self.risk_labels[i]: round(float(pred_proba[i]), 4)
            for i in range(len(self.risk_labels))
            if i < len(pred_proba)
        }

        return {
            "risk_level": predicted_level,
            "probability": round(flood_probability, 4),
            "confidence": round(float(confidence), 4),
            "class_probabilities": class_probabilities,
            "feature_values": feature_values,
            "top_drivers": drivers[:6],
            "all_drivers": drivers,
            "explanation": explanation,
            "model_inputs_source": (
                "13 weather and terrain features ONLY. "
                "No river discharge data is used as input. "
                "Features: precipitation (7d, 30d, max daily, anomaly), "
                "soil moisture, temperature, elevation, season, historical flood trends."
            ),
            "model_version": "gbm_v2",
        }

    def _describe_feature_influence(self, feature: str, value: float, predicted_level: str) -> str:
        """Generate human-readable influence description for a single feature."""
        descriptions = {
            "precip_7d": lambda v: (
                f"Very heavy rainfall ({v:.1f} mm in 7 days) — increases flood risk"
                if v > 100 else
                f"Heavy rainfall ({v:.1f} mm in 7 days) — elevates risk"
                if v > 50 else
                f"Moderate rainfall ({v:.1f} mm in 7 days)"
                if v > 20 else
                f"Minimal rainfall ({v:.1f} mm in 7 days) — pushes toward LOW"
            ),
            "precip_30d": lambda v: (
                f"Very high 30-day precipitation ({v:.1f} mm) — saturated conditions"
                if v > 300 else
                f"Elevated 30-day precipitation ({v:.1f} mm)"
                if v > 100 else
                f"Low 30-day precipitation ({v:.1f} mm) — dry conditions"
            ),
            "max_daily_rain_7d": lambda v: (
                f"Extreme daily rainfall peak ({v:.1f} mm) — flash flood risk"
                if v > 100 else
                f"Notable daily peak ({v:.1f} mm)"
                if v > 50 else
                f"Low daily rainfall peak ({v:.1f} mm) — no extreme events"
            ),
            "precip_anomaly": lambda v: (
                f"Rainfall far above normal ({v:+.2f} sigma) — unusual wet conditions"
                if v > 1.5 else
                f"Rainfall above normal ({v:+.2f} sigma)"
                if v > 0.5 else
                f"Near-normal or below-normal rainfall ({v:+.2f} sigma)"
            ),
            "soil_moisture": lambda v: (
                f"Very high soil moisture ({v:.3f}) — ground saturated, runoff likely"
                if v > 0.4 else
                f"Elevated soil moisture ({v:.3f})"
                if v > 0.3 else
                f"Moderate soil moisture ({v:.3f}) — ground can absorb more water"
                if v > 0.15 else
                f"Low soil moisture ({v:.3f}) — dry ground"
            ),
            "temperature": lambda v: (
                f"High temperature ({v:.1f} C) — may increase evaporation"
                if v > 35 else
                f"Moderate temperature ({v:.1f} C)"
                if v > 20 else
                f"Cool temperature ({v:.1f} C) — possible snowmelt factor"
            ),
            "elevation_m": lambda v: (
                f"Very low elevation ({v:.0f} m) — highly vulnerable to flooding"
                if v < 10 else
                f"Low elevation ({v:.0f} m) — flood-prone terrain"
                if v < 50 else
                f"Moderate elevation ({v:.0f} m)"
                if v < 200 else
                f"High elevation ({v:.0f} m) — less flood-vulnerable"
            ),
            "month": lambda v: (
                f"Peak monsoon season (month {int(v)}) — highest seasonal flood risk"
                if int(v) in [7, 8] else
                f"Monsoon season (month {int(v)}) — elevated seasonal risk"
                if int(v) in [6, 9] else
                f"Pre/post monsoon (month {int(v)}) — moderate seasonal risk"
                if int(v) in [5, 10] else
                f"Dry season (month {int(v)}) — low seasonal flood risk"
            ),
            "mean_flood_pct": lambda v: (
                f"High historical flood average ({v*100:.2f}%) — region has flood history"
                if v > 0.1 else
                f"Low historical flood average ({v*100:.2f}%)"
            ),
            "max_flood_pct": lambda v: (
                f"High historical flood peak ({v*100:.2f}%)"
                if v > 0.15 else
                f"Low historical flood peak ({v*100:.2f}%)"
            ),
            "trend": lambda v: (
                f"Increasing flood trend ({v:+.4f}) — risk is rising"
                if v > 0.01 else
                f"Decreasing flood trend ({v:+.4f}) — risk is declining"
                if v < -0.01 else
                f"Stable flood trend ({v:+.4f})"
            ),
            "rainfall_mm": lambda v: (
                f"Significant recent rainfall ({v:.1f} mm)"
                if v > 50 else
                f"Minimal recent rainfall ({v:.1f} mm)"
            ),
            "risk_multiplier": lambda v: (
                f"High seasonal risk multiplier ({v:.2f})"
                if v > 1.3 else
                f"Normal risk multiplier ({v:.2f})"
            ),
        }

        fn = descriptions.get(feature)
        if fn:
            return fn(value)
        return f"{feature} = {value:.4f}"

    def _generate_explanation_text(
        self,
        risk_level: str,
        probability: float,
        confidence: float,
        top_drivers: list[dict],
        feature_values: dict,
    ) -> str:
        """Generate a human-readable explanation paragraph."""
        level_desc = {
            "LOW": "LOW risk (normal conditions)",
            "MEDIUM": "MEDIUM risk (elevated indicators)",
            "HIGH": "HIGH risk (significant flood indicators)",
            "CRITICAL": "CRITICAL risk (extreme flood conditions)",
        }

        parts = [
            f"The ML model predicts {level_desc.get(risk_level, risk_level)} "
            f"with {probability*100:.1f}% probability and {confidence*100:.1f}% confidence."
        ]

        # Add top 3 reasons
        reasons = []
        for d in top_drivers[:3]:
            reasons.append(d["influence"])

        if reasons:
            parts.append("Key factors: " + "; ".join(reasons) + ".")

        # Add data source note
        parts.append(
            "This prediction uses ONLY weather and terrain data "
            "(precipitation, soil moisture, temperature, elevation, season). "
            "River discharge (GloFAS) data is NOT used as an input."
        )

        return " ".join(parts)

    # ── Model persistence ──

    def _save_persisted_model(self):
        """Save trained model and scaler to disk using joblib."""
        try:
            import joblib
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info("Persisted model to %s", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to persist model: %s", e)

    def _load_persisted_model(self) -> bool:
        """Load a previously trained model from disk. Returns True if successful."""
        try:
            if MODEL_PATH.exists() and SCALER_PATH.exists():
                import joblib
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                return True
        except Exception as e:
            logger.error("Failed to load persisted model: %s", e)
        return False

    # ── Synthetic data ──

    def _generate_synthetic_data(self, n_samples: int) -> list[dict]:
        """Generate synthetic training data with all 13 features."""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            # Random flood scenario
            base_flood = np.random.beta(2, 10)  # Most regions have low flood %
            rainfall = np.random.exponential(30)
            elevation = np.random.uniform(0, 500)
            month = np.random.randint(1, 13)

            # Derived weather features
            precip_7d = rainfall * np.random.uniform(0.5, 1.5)
            precip_30d = precip_7d * np.random.uniform(3, 5)
            max_daily_rain = precip_7d * np.random.uniform(0.3, 0.8)
            precip_anomaly = np.random.normal(0, 1)
            soil_moisture = np.random.uniform(0.05, 0.5)
            temperature = np.random.uniform(15, 40)

            # Monsoon months increase flood risk
            seasonal_factor = 1.5 if month in [6, 7, 8, 9] else 1.0

            # Determine true risk level using multiple factors
            risk_score = base_flood * seasonal_factor
            if rainfall > 100:
                risk_score *= 1.5
            if elevation < 50:
                risk_score *= 1.3
            if precip_anomaly > 1.5:
                risk_score *= 1.3
            if soil_moisture > 0.35:
                risk_score *= 1.2

            if risk_score > 0.25:
                label = "CRITICAL"
            elif risk_score > 0.15:
                label = "HIGH"
            elif risk_score > 0.05:
                label = "MEDIUM"
            else:
                label = "LOW"

            # Create history
            history = [{"flood_percentage": base_flood + np.random.normal(0, 0.02)} for _ in range(5)]

            data.append({
                "flood_history": history,
                "external_factors": {
                    "rainfall_mm": rainfall,
                    "elevation_mean_m": elevation,
                    "risk_multiplier": seasonal_factor,
                },
                "raw_features": {
                    "precip_7d": precip_7d,
                    "precip_30d": precip_30d,
                    "max_daily_rain_7d": max_daily_rain,
                    "precip_anomaly": precip_anomaly,
                    "soil_moisture": soil_moisture,
                    "temperature": temperature,
                    "month": month,
                    "elevation": elevation,
                },
                "label": label,
            })

        return data
