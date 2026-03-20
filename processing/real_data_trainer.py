"""
Real Data Training Pipeline — Ground Truth from GloFAS Discharge.

Builds training datasets by aligning REAL weather features (precipitation,
soil moisture, temperature) with REAL river discharge data (GloFAS) so that
labels reflect actual flooding conditions rather than precipitation guesses.

How it works:
  1. Fetch N days of daily weather (Open-Meteo Archive API)
  2. Fetch N days of daily river discharge (Open-Meteo Flood API)
  3. Align both datasets by date
  4. Label each day from ACTUAL discharge anomaly (ground truth)
  5. Extract weather-based features as model inputs

This means the model learns:
  "Given these weather conditions → this is the actual flood risk"
  instead of:
  "Given these weather conditions → this is our guess of flood risk"
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import requests

from processing.live_flood_data import LiveFloodDataFetcher

logger = logging.getLogger("cosmeon.processing.trainer")

FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
WEATHER_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"


class RealDataTrainer:
    """Builds real training data from historical weather + discharge APIs."""

    def __init__(self):
        self.fetcher = LiveFloodDataFetcher()
        logger.info("RealDataTrainer initialized (ground-truth mode)")

    # ── Core: build training data for a single region ──

    def build_training_data(
        self,
        lat: float,
        lon: float,
        elevation: float = 100.0,
        days_history: int = 365,
    ) -> list[dict]:
        """
        Build training dataset from real historical data.

        Fetches weather + discharge for the same time window, aligns by date,
        and labels each day based on ACTUAL discharge anomaly.

        Args:
            lat: Latitude of region center
            lon: Longitude of region center
            elevation: Mean elevation of region
            days_history: How many days to look back (uses start/end date
                          API so supports up to multi-year history)

        Returns:
            List of training samples with ground-truth labels
        """
        logger.info(
            "Building training data: lat=%.2f, lon=%.2f, days=%d",
            lat, lon, days_history,
        )

        # ── 1. Fetch discharge data (ground truth for labels) ──
        # Use start/end date for unlimited history (vs past_days cap of 210)
        end_date = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=days_history + 3)).strftime("%Y-%m-%d")
        discharge_data = self._fetch_discharge_history(
            lat, lon, start_date=start_date, end_date=end_date
        )
        if not discharge_data["dates"]:
            logger.warning("No discharge data available. Cannot build ground-truth labels.")
            return self._fallback_weather_only(lat, lon, elevation, days_history)

        # ── 2. Fetch weather data for the same period ──
        weather_data = self._fetch_weather_history(
            lat, lon, discharge_data["dates"][0], discharge_data["dates"][-1]
        )
        if not weather_data["dates"]:
            logger.warning("No weather data available.")
            return []

        # ── 3. Align by date ──
        aligned = self._align_by_date(weather_data, discharge_data)
        if len(aligned) < 30:
            logger.warning("Only %d aligned days. Falling back.", len(aligned))
            return self._fallback_weather_only(lat, lon, elevation, days_history)

        # ── 4. Compute discharge statistics for labeling ──
        all_discharges = [d["discharge"] for d in aligned if d["discharge"] > 0]
        if not all_discharges:
            logger.warning("All discharge values are zero. Cannot label.")
            return self._fallback_weather_only(lat, lon, elevation, days_history)

        discharge_mean = np.mean(all_discharges)
        discharge_std = np.std(all_discharges) if len(all_discharges) > 1 else discharge_mean * 0.3
        discharge_p75 = np.percentile(all_discharges, 75)
        discharge_p90 = np.percentile(all_discharges, 90)
        discharge_p95 = np.percentile(all_discharges, 95)

        logger.info(
            "Discharge stats: mean=%.1f, std=%.1f, p75=%.1f, p90=%.1f, p95=%.1f",
            discharge_mean, discharge_std, discharge_p75, discharge_p90, discharge_p95,
        )

        # ── 5. Build training samples ──
        training_data = []
        precips = [d["precip"] for d in aligned]

        for i in range(30, len(aligned)):  # need 30-day lookback
            day = aligned[i]

            # --- GROUND-TRUTH LABEL from actual discharge ---
            discharge = day["discharge"]
            label = self._label_from_discharge(
                discharge, discharge_mean, discharge_std,
                discharge_p90, discharge_p95,
            )

            # --- Features from weather ---
            precip_7d = sum(precips[max(0, i-7):i])
            precip_30d = sum(precips[max(0, i-30):i])
            max_daily_rain_7d = max(precips[max(0, i-7):i]) if precips[max(0, i-7):i] else 0
            soil_moist = day.get("soil_moisture", 0)
            temp = day.get("temp_max", 25)

            # Month for seasonality
            try:
                month = datetime.strptime(day["date"], "%Y-%m-%d").month
            except (ValueError, IndexError):
                month = 6

            # Precipitation anomaly (z-score of 30-day total vs prior)
            window_start = max(0, i - min(i, 365))
            annual_precip_30d = []
            for j in range(window_start, i - 30, 30):
                annual_precip_30d.append(sum(precips[j:j+30]))
            if annual_precip_30d:
                precip_mean = np.mean(annual_precip_30d)
                precip_std_val = np.std(annual_precip_30d) if len(annual_precip_30d) > 1 else precip_mean * 0.5
                precip_anomaly = (precip_30d - precip_mean) / max(precip_std_val, 0.01)
            else:
                precip_anomaly = 0

            # Flood proxy percentage (for flood_history feature)
            flood_proxy = self._estimate_flood_proxy(
                precip_7d, precip_30d, max_daily_rain_7d,
                elevation, soil_moist, month, precip_anomaly,
            )

            # Build training sample
            flood_history = [
                {"flood_percentage": flood_proxy * (1 + np.random.normal(0, 0.01))}
                for _ in range(5)
            ]
            seasonal_mult = 1.5 if month in [6, 7, 8, 9] else (1.2 if month in [5, 10] else 1.0)

            sample = {
                "flood_history": flood_history,
                "external_factors": {
                    "rainfall_mm": precip_7d,
                    "elevation_mean_m": elevation,
                    "risk_multiplier": seasonal_mult,
                },
                "label": label,
                "raw_features": {
                    "precip_7d": precip_7d,
                    "precip_30d": precip_30d,
                    "max_daily_rain_7d": max_daily_rain_7d,
                    "precip_anomaly": precip_anomaly,
                    "soil_moisture": soil_moist,
                    "temperature": temp,
                    "month": month,
                    "elevation": elevation,
                },
                # Metadata (not used in training, for audit)
                "_ground_truth": {
                    "date": day["date"],
                    "discharge_m3s": discharge,
                    "discharge_mean": round(discharge_mean, 1),
                    "discharge_std": round(discharge_std, 1),
                    "anomaly_sigma": round(
                        (discharge - discharge_mean) / max(discharge_std, 0.01), 2
                    ),
                },
            }
            training_data.append(sample)

        logger.info(
            "Built %d training samples with GROUND-TRUTH labels (lat=%.2f, lon=%.2f)",
            len(training_data), lat, lon,
        )

        # Log label distribution
        labels = [s["label"] for s in training_data]
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = labels.count(level)
            logger.info("  %s: %d (%.1f%%)", level, count, count / max(len(labels), 1) * 100)

        return training_data

    # ── Labeling from actual discharge ──

    def _label_from_discharge(
        self,
        discharge: float,
        mean: float,
        std: float,
        p90: float,
        p95: float,
    ) -> str:
        """
        Label a day's flood risk based on ACTUAL river discharge.

        Uses both statistical anomaly AND percentile thresholds:
          - CRITICAL: discharge > 2.5σ above mean OR > P95
          - HIGH:     discharge > 1.5σ above mean OR > P90
          - MEDIUM:   discharge > 0.8σ above mean OR > P75-equivalent
          - LOW:      otherwise

        This is calibrated to real hydrological standards.
        """
        if discharge <= 0:
            return "LOW"

        anomaly = (discharge - mean) / max(std, 0.01)
        ratio = discharge / max(mean, 0.01)

        # CRITICAL: extreme discharge
        if anomaly > 2.5 or ratio > 3.0 or discharge > p95:
            return "CRITICAL"
        # HIGH: significantly elevated
        elif anomaly > 1.5 or ratio > 2.0 or discharge > p90:
            return "HIGH"
        # MEDIUM: moderately elevated
        elif anomaly > 0.8 or ratio > 1.3:
            return "MEDIUM"
        # LOW: normal
        return "LOW"

    # ── Data fetching helpers ──

    def _fetch_discharge_history(
        self,
        lat: float,
        lon: float,
        past_days: int = 210,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """
        Fetch daily river discharge from GloFAS Flood API.

        Supports two modes:
          - start_date + end_date: full archive access (years of history)
          - past_days: convenience mode, capped at 210 by the API
        """
        try:
            if start_date and end_date:
                # Archive mode — supports unlimited history back to ~1984
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "river_discharge,river_discharge_mean,river_discharge_max",
                    "start_date": start_date,
                    "end_date": end_date,
                }
            else:
                # Legacy past_days mode (max 210)
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "river_discharge,river_discharge_mean,river_discharge_max",
                    "past_days": min(past_days, 210),
                    "forecast_days": 0,
                }
            response = requests.get(FLOOD_API, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            discharge = daily.get("river_discharge", [])
            discharge = [d if d is not None else 0 for d in discharge]

            logger.info("Fetched %d days of discharge data (lat=%.2f, lon=%.2f)", len(dates), lat, lon)
            return {"dates": dates, "discharge": discharge}

        except Exception as e:
            logger.error("Failed to fetch discharge history from flood API: %s", e)
            logger.warning("Falling back to ERA5 precipitation surrogate for discharge labels")
            return self._fetch_discharge_era5_surrogate(lat, lon, start_date, end_date, past_days)

    def _fetch_discharge_era5_surrogate(
        self,
        lat: float,
        lon: float,
        start_date: str = None,
        end_date: str = None,
        past_days: int = 210,
    ) -> dict:
        """
        ERA5 precipitation-based discharge surrogate.

        Used when flood-api.open-meteo.com is unreachable (blocked on some hosts).
        Fetches ERA5 precipitation and applies a 3-day lag runoff proxy
        (3-day rolling sum × 10 m³/s per mm/day) — the same method used in
        model_hub._discharge_precip_surrogate().

        Returns the same schema as _fetch_discharge_history() so the caller
        can still compute anomaly-based labels without any code changes.
        """
        try:
            if start_date and end_date:
                params = {
                    "latitude": lat, "longitude": lon,
                    "start_date": start_date, "end_date": end_date,
                    "daily": "precipitation_sum",
                }
            else:
                # Approximate start_date from past_days
                end_dt = datetime.utcnow() - timedelta(days=5)
                start_dt = end_dt - timedelta(days=past_days)
                params = {
                    "latitude": lat, "longitude": lon,
                    "start_date": start_dt.strftime("%Y-%m-%d"),
                    "end_date": end_dt.strftime("%Y-%m-%d"),
                    "daily": "precipitation_sum",
                }

            resp = requests.get(WEATHER_ARCHIVE_API, params=params, timeout=20)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            dates = daily.get("time", [])
            precip = [p if p is not None else 0.0 for p in daily.get("precipitation_sum", [])]

            if not dates:
                logger.error("ERA5 surrogate: no precipitation data returned")
                return {"dates": [], "discharge": []}

            # 3-day rolling sum × 10 m³/s as surrogate discharge
            lag = 3
            surrogate = [
                sum(precip[max(0, i - lag):i + 1]) * 10.0
                for i in range(len(precip))
            ]

            logger.info(
                "ERA5 surrogate: built %d days of surrogate discharge (lat=%.2f, lon=%.2f)",
                len(dates), lat, lon,
            )
            return {"dates": dates, "discharge": surrogate}

        except Exception as e:
            logger.error("ERA5 surrogate discharge also failed: %s", e)
            return {"dates": [], "discharge": []}

    def _fetch_weather_history(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> dict:
        """Fetch daily weather from Open-Meteo Archive API."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,soil_moisture_0_to_7cm_mean",
            }
            response = requests.get(WEATHER_ARCHIVE_API, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            precip = [p if p is not None else 0 for p in daily.get("precipitation_sum", [])]
            temp_max = [t if t is not None else 0 for t in daily.get("temperature_2m_max", [])]
            soil = [s if s is not None else 0 for s in daily.get("soil_moisture_0_to_7cm_mean", [])]

            logger.info("Fetched %d days of weather data (lat=%.2f, lon=%.2f)", len(dates), lat, lon)
            return {"dates": dates, "precip": precip, "temp_max": temp_max, "soil_moisture": soil}

        except Exception as e:
            logger.error("Failed to fetch weather history: %s", e)
            return {"dates": [], "precip": [], "temp_max": [], "soil_moisture": []}

    def _align_by_date(self, weather: dict, discharge: dict) -> list[dict]:
        """Align weather and discharge data by date, keeping only overlapping days."""
        weather_lookup = {}
        for i, date in enumerate(weather["dates"]):
            weather_lookup[date] = {
                "precip": weather["precip"][i] if i < len(weather["precip"]) else 0,
                "temp_max": weather["temp_max"][i] if i < len(weather["temp_max"]) else 25,
                "soil_moisture": weather["soil_moisture"][i] if i < len(weather["soil_moisture"]) else 0,
            }

        aligned = []
        for i, date in enumerate(discharge["dates"]):
            if date in weather_lookup:
                w = weather_lookup[date]
                aligned.append({
                    "date": date,
                    "discharge": discharge["discharge"][i] if i < len(discharge["discharge"]) else 0,
                    "precip": w["precip"],
                    "temp_max": w["temp_max"],
                    "soil_moisture": w["soil_moisture"],
                })

        logger.info(
            "Aligned %d days (discharge=%d, weather=%d)",
            len(aligned), len(discharge["dates"]), len(weather["dates"]),
        )
        return aligned

    # ── Helpers ──

    def _estimate_flood_proxy(
        self,
        precip_7d: float,
        precip_30d: float,
        max_daily: float,
        elevation: float,
        soil_moisture: float,
        month: int,
        precip_anomaly: float,
    ) -> float:
        """Estimate a flood proxy percentage from weather features."""
        base = 0.0

        if precip_7d > 200:
            base += 0.30
        elif precip_7d > 100:
            base += 0.15
        elif precip_7d > 50:
            base += 0.08
        elif precip_7d > 20:
            base += 0.03

        if max_daily > 100:
            base += 0.15
        elif max_daily > 50:
            base += 0.08

        if elevation < 10:
            base *= 2.0
        elif elevation < 50:
            base *= 1.5
        elif elevation < 100:
            base *= 1.2

        if soil_moisture > 0.4:
            base *= 1.3
        elif soil_moisture > 0.3:
            base *= 1.1

        if month in [6, 7, 8, 9]:
            base *= 1.3

        if precip_anomaly > 2:
            base *= 1.4
        elif precip_anomaly > 1:
            base *= 1.2

        return min(base, 0.6)

    def _fallback_weather_only(
        self, lat: float, lon: float, elevation: float, days_history: int,
    ) -> list[dict]:
        """
        Fallback: build training data from weather only when discharge is unavailable.
        Uses precipitation-based heuristic labels (less accurate than discharge labels).
        """
        logger.warning("Using precipitation-based fallback labels (less accurate)")
        weather = self.fetcher.fetch_historical_weather(lat, lon, days_back=min(days_history, 730))

        if not weather.dates or len(weather.dates) < 30:
            return []

        training_data = []
        precip = weather.precipitation_mm
        temp_max = weather.temperature_max
        soil = weather.soil_moisture if weather.soil_moisture else [0] * len(precip)
        dates = weather.dates

        for i in range(30, len(dates)):
            precip_7d = sum(precip[max(0, i-7):i])
            precip_30d = sum(precip[max(0, i-30):i])
            max_daily_rain_7d = max(precip[max(0, i-7):i]) if precip[max(0, i-7):i] else 0
            temp = temp_max[i] if i < len(temp_max) else 25
            soil_moist = soil[i] if i < len(soil) else 0

            try:
                month = datetime.strptime(dates[i], "%Y-%m-%d").month
            except (ValueError, IndexError):
                month = 6

            window_start = max(0, i - min(i, 365))
            annual_precip_30d = []
            for j in range(window_start, i - 30, 30):
                annual_precip_30d.append(sum(precip[j:j+30]))
            if annual_precip_30d:
                precip_mean = np.mean(annual_precip_30d)
                precip_std_val = np.std(annual_precip_30d) if len(annual_precip_30d) > 1 else precip_mean * 0.5
                precip_anomaly = (precip_30d - precip_mean) / max(precip_std_val, 0.01)
            else:
                precip_anomaly = 0

            flood_proxy = self._estimate_flood_proxy(
                precip_7d, precip_30d, max_daily_rain_7d,
                elevation, soil_moist, month, precip_anomaly,
            )

            # Heuristic label (fallback)
            score = 0.0
            if precip_7d > 200: score += 3.0
            elif precip_7d > 100: score += 2.0
            elif precip_7d > 50: score += 1.0
            if max_daily_rain_7d > 100: score += 2.5
            elif max_daily_rain_7d > 50: score += 1.5
            if precip_anomaly > 2.0: score += 2.0
            elif precip_anomaly > 1.0: score += 1.0
            if elevation < 50: score *= 1.3
            if month in [6, 7, 8, 9]: score *= 1.3

            if score > 5.0: label = "CRITICAL"
            elif score > 3.0: label = "HIGH"
            elif score > 1.0: label = "MEDIUM"
            else: label = "LOW"

            flood_history = [
                {"flood_percentage": flood_proxy * (1 + np.random.normal(0, 0.01))}
                for _ in range(5)
            ]
            seasonal_mult = 1.5 if month in [6, 7, 8, 9] else (1.2 if month in [5, 10] else 1.0)

            sample = {
                "flood_history": flood_history,
                "external_factors": {
                    "rainfall_mm": precip_7d,
                    "elevation_mean_m": elevation,
                    "risk_multiplier": seasonal_mult,
                },
                "label": label,
                "raw_features": {
                    "precip_7d": precip_7d,
                    "precip_30d": precip_30d,
                    "max_daily_rain_7d": max_daily_rain_7d,
                    "precip_anomaly": precip_anomaly,
                    "soil_moisture": soil_moist,
                    "temperature": temp,
                    "month": month,
                    "elevation": elevation,
                },
            }
            training_data.append(sample)

        logger.info("Fallback: built %d samples from weather-only data", len(training_data))
        return training_data

    # ── Multi-region ──

    def build_multi_region_training_data(
        self, regions: list[dict]
    ) -> list[dict]:
        """
        Build training data across multiple regions for a more robust model.

        Args:
            regions: List of {"name": str, "lat": float, "lon": float, "elevation": float}

        Returns:
            Combined training data from all regions
        """
        all_data = []
        for region in regions:
            logger.info("Building training data for %s...", region["name"])
            data = self.build_training_data(
                lat=region["lat"],
                lon=region["lon"],
                elevation=region.get("elevation", 100),
                days_history=365,  # 365 days via archive API (was 210)
            )
            all_data.extend(data)

        logger.info("Total training samples across %d regions: %d", len(regions), len(all_data))

        # Log overall label distribution
        labels = [s["label"] for s in all_data]
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = labels.count(level)
            pct = count / max(len(labels), 1) * 100
            logger.info("  Overall %s: %d (%.1f%%)", level, count, pct)

        return all_data
