"""
LSTM Training Data Builder — Constructs sequential training data.

Fetches 2 years of daily weather from Open-Meteo Archive API and
GloFAS river discharge for ground-truth labels, then creates
sliding-window sequences for LSTM training.

Each 30-day window has 8 features per day:
  1. precipitation_sum (mm)
  2. soil_moisture (0-7cm mean)
  3. temp_max (C)
  4. temp_min (C)
  5. et0 (evapotranspiration mm)
  6. discharge_ratio (current / mean discharge)
  7. month_sin
  8. month_cos
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import requests

logger = logging.getLogger("cosmeon.processing.lstm_trainer")

WEATHER_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"

SEQ_LEN = 30
RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

DEFAULT_REGIONS = [
    # South Asia — monsoon-driven, excellent GloFAS coverage
    {"name": "Bihar, India",          "lat": 26.0,   "lon": 85.5,    "elevation": 55},
    {"name": "Dhaka, Bangladesh",     "lat": 23.75,  "lon": 90.4,    "elevation": 8},
    {"name": "Navi Mumbai, India",    "lat": 19.1,   "lon": 73.0,    "elevation": 14},
    {"name": "Kolkata, India",        "lat": 22.6,   "lon": 88.4,    "elevation": 6},
    {"name": "Assam, India",          "lat": 26.2,   "lon": 92.5,    "elevation": 55},
    {"name": "Sylhet, Bangladesh",    "lat": 24.9,   "lon": 91.9,    "elevation": 15},
    # Southeast Asia
    {"name": "Jakarta, Indonesia",    "lat": -6.2,   "lon": 106.8,   "elevation": 8},
    {"name": "Bangkok, Thailand",     "lat": 13.75,  "lon": 100.5,   "elevation": 2},
    {"name": "Ho Chi Minh City",      "lat": 10.8,   "lon": 106.7,   "elevation": 5},
    {"name": "Manila, Philippines",   "lat": 14.6,   "lon": 121.0,   "elevation": 15},
    # Europe — Rhine/Danube basin
    {"name": "Bremen, Germany",       "lat": 53.1,   "lon": 8.8,     "elevation": 12},
    {"name": "Rotterdam, Netherlands","lat": 51.9,   "lon": 4.5,     "elevation": 0},
    {"name": "Budapest, Hungary",     "lat": 47.5,   "lon": 19.0,    "elevation": 105},
    {"name": "Venice, Italy",         "lat": 45.4,   "lon": 12.3,    "elevation": 1},
    # Americas
    {"name": "Sao Paulo, Brazil",     "lat": -23.55, "lon": -46.6,   "elevation": 760},
    {"name": "Manaus, Brazil",        "lat": -3.1,   "lon": -60.0,   "elevation": 92},
    {"name": "New Orleans, USA",      "lat": 29.95,  "lon": -90.07,  "elevation": 0},
    {"name": "Houston, USA",          "lat": 29.76,  "lon": -95.37,  "elevation": 13},
    # East Asia — Yangtze basin
    {"name": "Wuhan, China",          "lat": 30.6,   "lon": 114.3,   "elevation": 23},
    {"name": "Chongqing, China",      "lat": 29.6,   "lon": 106.5,   "elevation": 259},
    # Africa
    {"name": "Khartoum, Sudan",       "lat": 15.55,  "lon": 32.53,   "elevation": 380},
    {"name": "Lagos, Nigeria",        "lat": 6.45,   "lon": 3.4,     "elevation": 2},
]


class LSTMDataBuilder:
    """Builds sliding-window sequential data for LSTM training."""

    def __init__(self):
        logger.info("LSTMDataBuilder initialized")

    def build_all_regions(
        self, regions: list[dict] = None, days_history: int = 730
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build training sequences across all regions.

        Returns:
            sequences: (N, 30, 8) array
            labels:    (N,) int array
        """
        regions = regions or DEFAULT_REGIONS
        all_seqs, all_labels = [], []

        for region in regions:
            try:
                seqs, labs = self.build_region(
                    lat=region["lat"],
                    lon=region["lon"],
                    days_history=days_history,
                )
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(labs)
                    logger.info(
                        "%s: %d sequences built", region["name"], len(seqs)
                    )
            except Exception as e:
                logger.error("Failed for %s: %s", region["name"], e)

        if not all_seqs:
            logger.warning("No sequences built. Returning empty arrays.")
            return np.empty((0, SEQ_LEN, 8)), np.empty((0,), dtype=int)

        sequences = np.concatenate(all_seqs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logger.info(
            "Total: %d sequences across %d regions", len(sequences), len(all_seqs)
        )
        return sequences, labels

    def build_region(
        self, lat: float, lon: float, days_history: int = 730
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sliding-window sequences for a single region.

        Returns:
            sequences: (N, 30, 8) array
            labels:    (N,) int array
        """
        # Fetch weather data (2 years)
        weather = self._fetch_weather(lat, lon, days_history)
        if not weather["dates"] or len(weather["dates"]) < SEQ_LEN + 1:
            return np.empty((0, SEQ_LEN, 8)), np.empty((0,), dtype=int)

        # Fetch discharge data (max 210 days for labeling)
        discharge = self._fetch_discharge(lat, lon, min(days_history, 210))

        # Build daily feature matrix
        n_days = len(weather["dates"])
        features = np.zeros((n_days, 8))
        features[:, 0] = weather["precip"]
        features[:, 1] = weather["soil_moisture"]
        features[:, 2] = weather["temp_max"]
        features[:, 3] = weather["temp_min"]
        features[:, 4] = weather["et0"]

        # Discharge ratio (normalized by mean)
        if discharge["discharge"]:
            discharge_lookup = dict(zip(discharge["dates"], discharge["discharge"]))
            valid_discharge = [d for d in discharge["discharge"] if d > 0]
            mean_discharge = np.mean(valid_discharge) if valid_discharge else 1.0
        else:
            discharge_lookup = {}
            mean_discharge = 1.0

        for i, date_str in enumerate(weather["dates"]):
            d = discharge_lookup.get(date_str, 0)
            features[i, 5] = d / max(mean_discharge, 0.01) if d > 0 else 0

            # Month encoding (cyclical)
            try:
                month = datetime.strptime(date_str, "%Y-%m-%d").month
            except ValueError:
                month = 6
            features[i, 6] = np.sin(2 * np.pi * month / 12)
            features[i, 7] = np.cos(2 * np.pi * month / 12)

        # Build labels from discharge (where available) or weather heuristic
        labels_daily = self._build_daily_labels(
            weather, discharge, features, mean_discharge
        )

        # Create sliding windows
        sequences, labels = [], []
        for i in range(SEQ_LEN, n_days):
            seq = features[i - SEQ_LEN: i]  # (30, 8)
            label = labels_daily[i]
            sequences.append(seq)
            labels.append(label)

        if not sequences:
            return np.empty((0, SEQ_LEN, 8)), np.empty((0,), dtype=int)

        return np.array(sequences), np.array(labels)

    def _build_daily_labels(
        self, weather: dict, discharge: dict, features: np.ndarray, mean_discharge: float
    ) -> np.ndarray:
        """Assign a risk label to each day from discharge or weather heuristic."""
        n_days = len(weather["dates"])
        labels = np.zeros(n_days, dtype=int)  # default LOW

        if discharge["discharge"]:
            discharge_lookup = dict(zip(discharge["dates"], discharge["discharge"]))
            valid = [d for d in discharge["discharge"] if d > 0]
            if valid:
                d_mean = np.mean(valid)
                d_std = np.std(valid) if len(valid) > 1 else d_mean * 0.3
                d_p90 = np.percentile(valid, 90)
                d_p95 = np.percentile(valid, 95)
        else:
            discharge_lookup = {}
            valid = []

        for i, date_str in enumerate(weather["dates"]):
            d = discharge_lookup.get(date_str)

            if d is not None and d > 0 and valid:
                # Ground-truth label from discharge
                anomaly = (d - d_mean) / max(d_std, 0.01)
                ratio = d / max(d_mean, 0.01)
                if anomaly > 2.5 or ratio > 3.0 or d > d_p95:
                    labels[i] = 3  # CRITICAL
                elif anomaly > 1.5 or ratio > 2.0 or d > d_p90:
                    labels[i] = 2  # HIGH
                elif anomaly > 0.8 or ratio > 1.3:
                    labels[i] = 1  # MEDIUM
                else:
                    labels[i] = 0  # LOW
            else:
                # Weather-based heuristic fallback
                precip = features[i, 0]
                soil = features[i, 1]
                score = 0.0
                if precip > 50:
                    score += 2.0
                elif precip > 20:
                    score += 1.0
                if soil > 0.4:
                    score += 1.5
                elif soil > 0.3:
                    score += 0.5
                # 7-day lookback
                if i >= 7:
                    precip_7d = features[max(0, i - 7):i, 0].sum()
                    if precip_7d > 200:
                        score += 3.0
                    elif precip_7d > 100:
                        score += 1.5

                if score > 4.0:
                    labels[i] = 3
                elif score > 2.5:
                    labels[i] = 2
                elif score > 1.0:
                    labels[i] = 1

        return labels

    # ─── API Fetchers ───

    def _fetch_weather(self, lat: float, lon: float, days: int) -> dict:
        """Fetch daily weather from Open-Meteo Archive API."""
        try:
            end_dt = datetime.utcnow() - timedelta(days=5)
            start_dt = end_dt - timedelta(days=days)
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
                "daily": (
                    "precipitation_sum,soil_moisture_0_to_7cm_mean,"
                    "temperature_2m_max,temperature_2m_min,"
                    "et0_fao_evapotranspiration"
                ),
            }
            resp = requests.get(WEATHER_ARCHIVE_API, params=params, timeout=30)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})

            def clean(arr):
                return [v if v is not None else 0 for v in arr]

            return {
                "dates": daily.get("time", []),
                "precip": clean(daily.get("precipitation_sum", [])),
                "soil_moisture": clean(daily.get("soil_moisture_0_to_7cm_mean", [])),
                "temp_max": clean(daily.get("temperature_2m_max", [])),
                "temp_min": clean(daily.get("temperature_2m_min", [])),
                "et0": clean(daily.get("et0_fao_evapotranspiration", [])),
            }
        except Exception as e:
            logger.error("Weather fetch failed: %s", e)
            return {"dates": [], "precip": [], "soil_moisture": [], "temp_max": [], "temp_min": [], "et0": []}

    def _fetch_discharge(self, lat: float, lon: float, days: int) -> dict:
        """Fetch GloFAS river discharge data."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "river_discharge",
                "past_days": min(days, 210),
                "forecast_days": 0,
            }
            resp = requests.get(FLOOD_API, params=params, timeout=20)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            discharge = [d if d is not None else 0 for d in daily.get("river_discharge", [])]
            return {
                "dates": daily.get("time", []),
                "discharge": discharge,
            }
        except Exception as e:
            logger.error("Discharge fetch failed: %s", e)
            return {"dates": [], "discharge": []}

    def build_sequence_for_prediction(
        self, lat: float, lon: float, days: int = 30
    ) -> np.ndarray:
        """
        Build a single recent 30-day sequence for inference.

        Returns:
            (30, 8) feature array for the most recent 30 days
        """
        weather = self._fetch_weather(lat, lon, days + 5)
        discharge = self._fetch_discharge(lat, lon, min(days + 5, 210))

        if not weather["dates"] or len(weather["dates"]) < days:
            logger.warning("Insufficient data for prediction sequence")
            return np.zeros((SEQ_LEN, 8))

        n = len(weather["dates"])
        features = np.zeros((n, 8))
        features[:, 0] = weather["precip"][:n]
        features[:, 1] = weather["soil_moisture"][:n]
        features[:, 2] = weather["temp_max"][:n]
        features[:, 3] = weather["temp_min"][:n]
        features[:, 4] = weather["et0"][:n]

        if discharge["discharge"]:
            d_lookup = dict(zip(discharge["dates"], discharge["discharge"]))
            valid = [d for d in discharge["discharge"] if d > 0]
            d_mean = np.mean(valid) if valid else 1.0
        else:
            d_lookup = {}
            d_mean = 1.0

        for i, date_str in enumerate(weather["dates"]):
            d = d_lookup.get(date_str, 0)
            features[i, 5] = d / max(d_mean, 0.01) if d > 0 else 0
            try:
                month = datetime.strptime(date_str, "%Y-%m-%d").month
            except ValueError:
                month = 6
            features[i, 6] = np.sin(2 * np.pi * month / 12)
            features[i, 7] = np.cos(2 * np.pi * month / 12)

        # Return last 30 days
        return features[-SEQ_LEN:]
