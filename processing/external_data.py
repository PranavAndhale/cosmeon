"""
Phase 9: External Data Integration.

Integrates rainfall, elevation (DEM), and population density data
to enhance flood risk modeling.
"""
import logging
from dataclasses import dataclass

import numpy as np
import requests

logger = logging.getLogger("cosmeon.processing.external")


@dataclass
class ExternalRiskFactors:
    """External data factors for risk enhancement."""
    rainfall_mm: float = 0.0
    elevation_mean_m: float = 0.0
    elevation_min_m: float = 0.0
    population_density: float = 0.0
    rainfall_anomaly: float = 0.0  # deviation from normal
    low_elevation_pct: float = 0.0  # % area below 10m
    risk_multiplier: float = 1.0

    def to_dict(self):
        return {
            "rainfall_mm": self.rainfall_mm,
            "elevation_mean_m": self.elevation_mean_m,
            "elevation_min_m": self.elevation_min_m,
            "population_density": self.population_density,
            "rainfall_anomaly": self.rainfall_anomaly,
            "low_elevation_pct": self.low_elevation_pct,
            "risk_multiplier": self.risk_multiplier,
        }


class ExternalDataIntegrator:
    """Fetches and integrates external datasets for enhanced risk modeling."""

    def __init__(self):
        # Open-Meteo API for weather/rainfall (free, no key needed)
        self.weather_api = "https://api.open-meteo.com/v1/forecast"
        self.elevation_api = "https://api.open-meteo.com/v1/elevation"
        logger.info("ExternalDataIntegrator initialized")

    def fetch_rainfall(self, lat: float, lon: float, days: int = 7) -> dict:
        """
        Fetch recent rainfall data from Open-Meteo API.

        Args:
            lat: Latitude of region center
            lon: Longitude of region center
            days: Number of days of rainfall history

        Returns:
            dict with rainfall data
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,rain_sum",
                "past_days": days,
                "forecast_days": 0,
            }
            response = requests.get(self.weather_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            precip = daily.get("precipitation_sum", [])
            total_rainfall = sum(p for p in precip if p is not None)
            avg_daily = total_rainfall / max(len(precip), 1)

            result = {
                "total_rainfall_mm": round(total_rainfall, 1),
                "avg_daily_mm": round(avg_daily, 1),
                "max_daily_mm": round(max((p for p in precip if p is not None), default=0), 1),
                "rainy_days": sum(1 for p in precip if p and p > 1.0),
                "daily_values": precip,
            }

            logger.info(
                "Rainfall data: total=%.1fmm, avg=%.1fmm/day, max=%.1fmm",
                result["total_rainfall_mm"], result["avg_daily_mm"], result["max_daily_mm"],
            )
            return result

        except Exception as e:
            logger.error("Failed to fetch rainfall data: %s", e)
            return {"total_rainfall_mm": 0, "avg_daily_mm": 0, "max_daily_mm": 0, "rainy_days": 0}

    def fetch_elevation(self, lat: float, lon: float) -> dict:
        """Fetch elevation data from Open-Meteo API."""
        try:
            # Sample multiple points in the region
            offsets = [-0.5, -0.25, 0, 0.25, 0.5]
            lats = [lat + o for o in offsets]
            lons = [lon + o for o in offsets]

            params = {
                "latitude": ",".join(str(l) for l in lats),
                "longitude": ",".join(str(l) for l in lons),
            }
            response = requests.get(self.elevation_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            elevations = data.get("elevation", [0])
            if isinstance(elevations, (int, float)):
                elevations = [elevations]

            result = {
                "mean_elevation_m": round(np.mean(elevations), 1),
                "min_elevation_m": round(min(elevations), 1),
                "max_elevation_m": round(max(elevations), 1),
                "low_elevation_pct": round(sum(1 for e in elevations if e < 10) / len(elevations), 2),
            }

            logger.info(
                "Elevation data: mean=%.1fm, min=%.1fm, low_elev_pct=%.0f%%",
                result["mean_elevation_m"], result["min_elevation_m"],
                result["low_elevation_pct"] * 100,
            )
            return result

        except Exception as e:
            logger.error("Failed to fetch elevation data: %s", e)
            return {"mean_elevation_m": 0, "min_elevation_m": 0, "max_elevation_m": 0, "low_elevation_pct": 0}

    def compute_risk_multiplier(self, rainfall: dict, elevation: dict) -> float:
        """
        Compute a risk multiplier based on external factors.

        Factors:
          - High rainfall increases risk
          - Low elevation increases risk
          - Recent heavy rain (>50mm/day) is a strong indicator

        Returns:
            Multiplier (1.0 = baseline, >1.0 = increased risk)
        """
        multiplier = 1.0

        # Rainfall factor
        total_rain = rainfall.get("total_rainfall_mm", 0)
        max_daily = rainfall.get("max_daily_mm", 0)

        if total_rain > 200:
            multiplier *= 1.5
        elif total_rain > 100:
            multiplier *= 1.3
        elif total_rain > 50:
            multiplier *= 1.1

        if max_daily > 100:
            multiplier *= 1.4
        elif max_daily > 50:
            multiplier *= 1.2

        # Elevation factor
        mean_elev = elevation.get("mean_elevation_m", 100)
        low_pct = elevation.get("low_elevation_pct", 0)

        if mean_elev < 10:
            multiplier *= 1.5
        elif mean_elev < 50:
            multiplier *= 1.2

        if low_pct > 0.5:
            multiplier *= 1.3

        return round(multiplier, 2)

    def get_risk_factors(self, bbox: list[float]) -> ExternalRiskFactors:
        """
        Get all external risk factors for a bounding box.

        Args:
            bbox: [west, south, east, north]

        Returns:
            ExternalRiskFactors with all external data
        """
        # Center of bbox
        lat = (bbox[1] + bbox[3]) / 2
        lon = (bbox[0] + bbox[2]) / 2

        logger.info("Fetching external risk factors for lat=%.2f, lon=%.2f", lat, lon)

        rainfall = self.fetch_rainfall(lat, lon)
        elevation = self.fetch_elevation(lat, lon)
        risk_multiplier = self.compute_risk_multiplier(rainfall, elevation)

        factors = ExternalRiskFactors(
            rainfall_mm=rainfall.get("total_rainfall_mm", 0),
            elevation_mean_m=elevation.get("mean_elevation_m", 0),
            elevation_min_m=elevation.get("min_elevation_m", 0),
            rainfall_anomaly=0,  # Would need historical normals
            low_elevation_pct=elevation.get("low_elevation_pct", 0),
            risk_multiplier=risk_multiplier,
        )

        logger.info("External risk factors: multiplier=%.2f", risk_multiplier)
        return factors
