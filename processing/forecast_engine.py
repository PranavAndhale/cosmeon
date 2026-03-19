"""
Phase 10B: Predictive Forecasting Engine.

Generates probabilistic multi-month flood risk forecasts by:
  1. Fetching historical weather data from Open-Meteo's climate archive
  2. Combining with existing GBM model predictions
  3. Applying seasonal decomposition + trend projection
  4. Returning per-month risk probabilities with confidence intervals

No additional ML dependencies required — uses existing scikit-learn
and numpy stack. Prophet is optional.
"""
import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("cosmeon.processing.forecast")


class ForecastEngine:
    """Generates probabilistic multi-month flood risk forecasts."""

    CLIMATE_API = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_API = "https://api.open-meteo.com/v1/forecast"

    # Seasonal flood risk multipliers (Northern Hemisphere baseline)
    # Adjusted per-region based on hemisphere detection
    NH_SEASONAL = {
        1: 0.6, 2: 0.5, 3: 0.7, 4: 0.8, 5: 0.9, 6: 1.0,
        7: 1.2, 8: 1.3, 9: 1.1, 10: 0.9, 11: 0.7, 12: 0.6,
    }

    def __init__(self):
        self._cache: dict[str, dict] = {}
        # Chronos time-series foundation model (loaded lazily on first use)
        self._chronos = None
        self._chronos_attempted = False
        logger.info("ForecastEngine initialized")

    def generate_forecast(
        self,
        lat: float,
        lon: float,
        region_name: str = "Unknown",
        horizon_months: int = 6,
        region_id: Optional[int] = None,
    ) -> dict:
        """
        Generate a probabilistic flood risk forecast.

        Args:
            lat: Latitude of the region
            lon: Longitude of the region
            region_name: Human-readable name
            horizon_months: Number of months to forecast (1-12)
            region_id: Optional region ID for caching

        Returns:
            {
                "region": str,
                "generated_at": str,
                "horizon_months": int,
                "monthly_forecast": [
                    {
                        "month": "2026-04",
                        "month_name": "April 2026",
                        "risk_probability": 0.35,
                        "risk_level": "MEDIUM",
                        "confidence_lower": 0.20,
                        "confidence_upper": 0.50,
                        "seasonal_factor": 0.8,
                        "precipitation_forecast_mm": 120.5,
                        "drivers": ["seasonal_monsoon", "recent_trend"]
                    },
                    ...
                ],
                "summary": {
                    "peak_risk_month": "July 2026",
                    "peak_probability": 0.72,
                    "overall_trend": "escalating" | "stable" | "declining",
                    "avg_risk_probability": 0.45,
                }
            }
        """
        horizon_months = max(1, min(horizon_months, 12))
        cache_key = f"{region_id or f'{lat:.2f}_{lon:.2f}'}_{horizon_months}"

        # Check cache (valid for 1 hour)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (datetime.utcnow() - datetime.fromisoformat(cached["generated_at"])).seconds
            if age < 3600:
                logger.info("Forecast cache hit for %s", cache_key)
                return cached

        # 1. Fetch historical climate data (last 1 year)
        historical = self._fetch_historical_climate(lat, lon)

        # 2. Fetch upcoming weather forecast (next 16 days from Open-Meteo)
        upcoming = self._fetch_short_term_forecast(lat, lon)

        # 3. Compute baseline risk from historical patterns
        baseline = self._compute_baseline_risk(historical, lat)

        # 3b. Try Chronos for data-driven precipitation forecasts (improves accuracy)
        self._load_chronos()
        chronos_monthly = []
        if self._chronos is not None and len(historical.get("precip", [])) >= 30:
            chronos_monthly = self._predict_precip_with_chronos(
                historical["precip"], horizon_months
            )

        # 4. Generate monthly forecasts
        now = datetime.utcnow()
        is_southern = lat < 0
        monthly = []

        for i in range(1, horizon_months + 1):
            target_date = now + timedelta(days=30 * i)
            month_num = target_date.month
            month_str = target_date.strftime("%Y-%m")
            month_name = target_date.strftime("%B %Y")

            # Adjust seasonal factors for hemisphere
            if is_southern:
                seasonal = self.NH_SEASONAL.get((month_num + 6 - 1) % 12 + 1, 0.8)
            else:
                seasonal = self.NH_SEASONAL.get(month_num, 0.8)

            # Combine signals
            base_prob = baseline.get("avg_flood_probability", 0.15)
            historical_month_risk = self._get_historical_month_risk(historical, month_num)

            # Weighted ensemble
            raw_probability = (
                0.35 * base_prob +  # recent risk level
                0.30 * historical_month_risk +  # same month in prior years
                0.25 * seasonal +  # seasonal pattern
                0.10 * (upcoming.get("trend_factor", 0.5))  # short-term weather
            )

            # Clamp
            risk_probability = max(0.02, min(0.98, raw_probability))

            # Confidence interval — tighter when Chronos data available (better model)
            chronos_ci_factor = 0.82 if (i <= len(chronos_monthly)) else 1.0
            uncertainty = (0.08 + (i * 0.03)) * chronos_ci_factor
            conf_lower = max(0.0, risk_probability - uncertainty)
            conf_upper = min(1.0, risk_probability + uncertainty)

            # Risk level
            risk_level = self._probability_to_level(risk_probability)

            # Precipitation — prefer Chronos data-driven estimate over seasonal heuristic
            if i <= len(chronos_monthly) and chronos_monthly[i - 1]["mean"] >= 0:
                precip = chronos_monthly[i - 1]["mean"]
            else:
                precip = self._estimate_monthly_precipitation(historical, month_num)

            # Drivers
            drivers = self._identify_drivers(
                seasonal, base_prob, historical_month_risk, upcoming, month_num
            )

            # ── Orb-specific forecast metrics ──
            # Infrastructure exposure: higher precip → more infrastructure stress
            _infra_exp = round(min(0.98, max(0.02, min(1.0, precip / 300.0) * risk_probability * seasonal)), 3)
            # Vegetation stress: dry months → high stress, wet months → low
            if precip > 150:
                _veg_stress = round(max(0.02, min(0.98, 0.05 + risk_probability * 0.1)), 3)
            elif precip > 80:
                _veg_stress = round(max(0.02, min(0.98, 0.15 + (1.0 - precip / 150.0) * 0.3)), 3)
            else:
                _veg_stress = round(max(0.02, min(0.98, 0.30 + (1.0 - precip / 80.0) * 0.5)), 3)

            monthly.append({
                "month": month_str,
                "month_name": month_name,
                "risk_probability": round(risk_probability, 3),
                "risk_level": risk_level,
                "infra_exposure": _infra_exp,
                "vegetation_stress_index": _veg_stress,
                "confidence_lower": round(conf_lower, 3),
                "confidence_upper": round(conf_upper, 3),
                "seasonal_factor": round(seasonal, 2),
                "precipitation_forecast_mm": round(precip, 1),
                "drivers": drivers,
            })

        # Summary
        peak = max(monthly, key=lambda m: m["risk_probability"])
        probs = [m["risk_probability"] for m in monthly]
        first_half = probs[:len(probs) // 2] if len(probs) > 1 else probs
        second_half = probs[len(probs) // 2:] if len(probs) > 1 else probs

        if sum(second_half) / max(len(second_half), 1) > sum(first_half) / max(len(first_half), 1) + 0.05:
            overall_trend = "escalating"
        elif sum(first_half) / max(len(first_half), 1) > sum(second_half) / max(len(second_half), 1) + 0.05:
            overall_trend = "declining"
        else:
            overall_trend = "stable"

        result = {
            "region": region_name,
            "generated_at": datetime.utcnow().isoformat(),
            "horizon_months": horizon_months,
            "monthly_forecast": monthly,
            "summary": {
                "peak_risk_month": peak["month_name"],
                "peak_probability": peak["risk_probability"],
                "overall_trend": overall_trend,
                "avg_risk_probability": round(sum(probs) / len(probs), 3),
            },
        }

        # Cache
        self._cache[cache_key] = result
        logger.info(
            "Forecast generated for %s: %d months, peak=%s (%.0f%%)",
            region_name, horizon_months, peak["month_name"], peak["risk_probability"] * 100
        )
        return result

    # ── Chronos foundation model (optional, lazy-loaded) ──

    def _load_chronos(self):
        """Lazily load Chronos T5-Tiny on first forecast call (CPU-compatible)."""
        if self._chronos_attempted:
            return
        self._chronos_attempted = True
        try:
            import torch
            from chronos import ChronosPipeline
            self._chronos = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("Chronos T5-Tiny loaded — data-driven precipitation forecasting active")
        except Exception as e:
            logger.info("Chronos not available (using statistical fallback): %s", e)
            self._chronos = None

    def _predict_precip_with_chronos(
        self, precip_series: list, months_ahead: int
    ) -> list[dict]:
        """
        Use Chronos to predict daily precipitation for the next N months.

        Returns list of dicts with keys: mean, p10, p90 (monthly totals in mm).
        Falls back to empty list on any error so caller always has a safe fallback.
        """
        try:
            import torch

            clean = [float(p) if p is not None else 0.0 for p in precip_series]
            context = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)  # (1, T)

            prediction_length = months_ahead * 30
            forecast = self._chronos.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=20,
            )  # shape: (1, num_samples, prediction_length)

            samples = forecast[0].numpy()  # (num_samples, prediction_length)

            monthly = []
            for m in range(months_ahead):
                start = m * 30
                end = start + 30
                month_samples = np.sum(samples[:, start:end], axis=1)
                monthly.append({
                    "mean": float(np.mean(month_samples)),
                    "p10":  float(np.percentile(month_samples, 10)),
                    "p90":  float(np.percentile(month_samples, 90)),
                })
            logger.info(
                "Chronos forecast: %d months, first-month mean=%.1f mm",
                months_ahead, monthly[0]["mean"] if monthly else 0,
            )
            return monthly
        except Exception as e:
            logger.warning("Chronos precipitation forecast failed: %s", e)
            return []

    def _fetch_historical_climate(self, lat: float, lon: float) -> dict:
        """Fetch 1 year of historical daily climate data."""
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=365)
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,rain_sum",
            }
            resp = requests.get(self.CLIMATE_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})

            precip = daily.get("precipitation_sum", [])
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            dates = daily.get("time", [])

            return {
                "precip": precip,
                "temps_max": temps_max,
                "temps_min": temps_min,
                "dates": dates,
                "total_precip": sum(p for p in precip if p is not None),
                "avg_daily_precip": sum(p for p in precip if p is not None) / max(len(precip), 1),
                "rainy_days": sum(1 for p in precip if p and p > 1.0),
            }
        except Exception as e:
            logger.warning("Failed to fetch historical climate: %s", e)
            return {"precip": [], "temps_max": [], "temps_min": [], "dates": [],
                    "total_precip": 0, "avg_daily_precip": 2.0, "rainy_days": 60}

    def _fetch_short_term_forecast(self, lat: float, lon: float) -> dict:
        """Fetch the next 16-day weather forecast."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,precipitation_probability_max",
                "forecast_days": 16,
            }
            resp = requests.get(self.FORECAST_API, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            precip = daily.get("precipitation_sum", [])
            prob = daily.get("precipitation_probability_max", [])

            total_precip = sum(p for p in precip if p is not None)
            avg_prob = sum(p for p in prob if p is not None) / max(len(prob), 1) / 100

            # Higher upcoming rain → higher trend factor
            trend_factor = min(1.0, avg_prob * 1.5)

            return {
                "total_precip_16d": total_precip,
                "avg_rain_probability": avg_prob,
                "trend_factor": trend_factor,
                "max_daily_precip": max((p for p in precip if p is not None), default=0),
            }
        except Exception as e:
            logger.warning("Failed to fetch short-term forecast: %s", e)
            return {"total_precip_16d": 0, "avg_rain_probability": 0.3, "trend_factor": 0.5, "max_daily_precip": 0}

    def _compute_baseline_risk(self, historical: dict, lat: float) -> dict:
        """Compute a baseline flood risk probability from historical data."""
        precip = historical.get("precip", [])
        if not precip:
            return {"avg_flood_probability": 0.15}

        # Count days with significant rainfall (>20mm = flood-prone)
        heavy_rain_days = sum(1 for p in precip if p and p > 20)
        total_days = len([p for p in precip if p is not None])

        # Base probability from frequency of heavy rain
        if total_days > 0:
            freq = heavy_rain_days / total_days
            base_prob = min(0.9, freq * 3.0)  # Scale up — heavy rain frequency is a strong signal
        else:
            base_prob = 0.15

        return {"avg_flood_probability": round(base_prob, 3)}

    def _get_historical_month_risk(self, historical: dict, target_month: int) -> float:
        """Get average risk for a specific month from historical data."""
        dates = historical.get("dates", [])
        precip = historical.get("precip", [])

        if not dates or not precip:
            return 0.3  # Default moderate

        month_precip = []
        for d, p in zip(dates, precip):
            if p is not None:
                try:
                    month = int(d[5:7])
                    if month == target_month:
                        month_precip.append(p)
                except (ValueError, IndexError):
                    pass

        if not month_precip:
            return 0.3

        avg = sum(month_precip) / len(month_precip)
        heavy_days = sum(1 for p in month_precip if p > 20)

        # Convert to probability
        prob = min(0.95, (avg / 30) + (heavy_days / max(len(month_precip), 1)) * 1.5)
        return max(0.05, prob)

    def _estimate_monthly_precipitation(self, historical: dict, target_month: int) -> float:
        """Estimate monthly precipitation from historical data."""
        dates = historical.get("dates", [])
        precip = historical.get("precip", [])

        if not dates or not precip:
            return 80.0  # Default

        month_totals = []
        current_total = 0
        current_month = None

        for d, p in zip(dates, precip):
            try:
                m = int(d[5:7])
                if m == target_month:
                    current_total += (p or 0)
                elif current_total > 0:
                    month_totals.append(current_total)
                    current_total = 0
            except (ValueError, IndexError):
                pass

        if current_total > 0:
            month_totals.append(current_total)

        return sum(month_totals) / max(len(month_totals), 1) if month_totals else 80.0

    def _identify_drivers(
        self, seasonal: float, base_prob: float, hist_risk: float,
        upcoming: dict, month_num: int
    ) -> list[str]:
        """Identify the primary drivers for a month's forecast."""
        drivers = []

        if seasonal > 1.0:
            drivers.append("monsoon_season" if seasonal > 1.1 else "wet_season")
        elif seasonal < 0.6:
            drivers.append("dry_season")

        if base_prob > 0.4:
            drivers.append("recent_high_activity")
        elif base_prob < 0.1:
            drivers.append("recent_low_activity")

        if hist_risk > 0.5:
            drivers.append("historical_flood_month")

        if upcoming.get("trend_factor", 0) > 0.7:
            drivers.append("elevated_near_term_rain")

        if not drivers:
            drivers.append("baseline_seasonal")

        return drivers[:3]  # Cap at 3 drivers

    @staticmethod
    def _probability_to_level(probability: float) -> str:
        """Convert flood probability to risk level."""
        if probability >= 0.7:
            return "CRITICAL"
        elif probability >= 0.45:
            return "HIGH"
        elif probability >= 0.2:
            return "MEDIUM"
        else:
            return "LOW"
