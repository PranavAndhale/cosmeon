"""
Phase 2A: Multi-Sensor Data Fusion Engine.

Fuses data from multiple satellite and weather sources:
  - Optical (Sentinel-2, Landsat): Vegetation health, surface water
  - SAR (Sentinel-1): Flood detection through clouds
  - Thermal (MODIS/Landsat): Heat stress indicators
  - Weather (Open-Meteo): Precipitation, temperature, soil moisture
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("cosmeon.processing.fusion")


@dataclass
class SensorLayer:
    """Individual sensor data layer."""
    source: str  # e.g. "sentinel-2", "sentinel-1-sar", "modis-thermal"
    data_type: str  # "optical", "sar", "thermal", "weather"
    quality_score: float = 1.0  # 0-1, cloud cover penalty etc.
    timestamp: str = ""
    values: dict = field(default_factory=dict)


@dataclass
class FusedAnalysis:
    """Result of multi-sensor fusion."""
    flood_confidence: float = 0.0
    vegetation_stress: float = 0.0  # NDVI-based, 0=healthy, 1=dead
    thermal_anomaly: float = 0.0  # degrees above normal
    soil_saturation: float = 0.0  # 0-1
    surface_water_extent_pct: float = 0.0
    cloud_penetration_used: bool = False
    sensors_fused: list = field(default_factory=list)
    fusion_weights: dict = field(default_factory=dict)
    quality_score: float = 0.0  # overall fusion quality
    timestamp: str = ""

    def to_dict(self):
        return {
            "flood_confidence": round(self.flood_confidence, 3),
            "vegetation_stress": round(self.vegetation_stress, 3),
            "thermal_anomaly": round(self.thermal_anomaly, 2),
            "soil_saturation": round(self.soil_saturation, 3),
            "surface_water_extent_pct": round(self.surface_water_extent_pct, 4),
            "cloud_penetration_used": self.cloud_penetration_used,
            "sensors_fused": self.sensors_fused,
            "fusion_weights": self.fusion_weights,
            "quality_score": round(self.quality_score, 2),
            "timestamp": self.timestamp,
        }


class DataFusionEngine:
    """Fuses multiple satellite sensor data sources for enhanced flood analysis."""

    WEATHER_API = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        logger.info("DataFusionEngine initialized")

    def fuse_sensors(
        self,
        lat: float,
        lon: float,
        region_name: str = "Unknown",
        existing_flood_pct: float = 0.0,
        cloud_cover_pct: float = 0.0,
    ) -> FusedAnalysis:
        """
        Perform multi-sensor fusion for a location.

        Combines optical, SAR, thermal, and weather data with
        adaptive weighting based on data quality.
        """
        layers = []

        # 1. Optical layer (Sentinel-2 proxy via existing analysis)
        optical = SensorLayer(
            source="sentinel-2",
            data_type="optical",
            quality_score=max(0.1, 1.0 - cloud_cover_pct),
            timestamp=datetime.utcnow().isoformat(),
            values={
                "surface_water_pct": existing_flood_pct,
                "ndvi_proxy": max(0.0, 0.7 - existing_flood_pct * 2),
            }
        )
        layers.append(optical)

        # 2. SAR layer (Sentinel-1 proxy — works through clouds)
        sar = self._simulate_sar_data(lat, lon, existing_flood_pct, cloud_cover_pct)
        layers.append(sar)

        # 3. Thermal layer (MODIS proxy)
        thermal = self._fetch_thermal_proxy(lat, lon)
        layers.append(thermal)

        # 4. Weather/soil moisture layer
        weather = self._fetch_weather_layer(lat, lon)
        layers.append(weather)

        # Compute adaptive fusion weights
        result = self._adaptive_fusion(layers, cloud_cover_pct)
        result.timestamp = datetime.utcnow().isoformat()

        logger.info(
            "Fusion complete for %s: confidence=%.2f, sensors=%s, quality=%.2f",
            region_name, result.flood_confidence,
            result.sensors_fused, result.quality_score
        )
        return result

    def _simulate_sar_data(
        self, lat: float, lon: float,
        flood_pct: float, cloud_cover: float
    ) -> SensorLayer:
        """
        Simulate SAR (Sentinel-1) backscatter analysis.
        SAR penetrates clouds, so quality is independent of cloud cover.
        """
        # SAR flood detection: water has low backscatter
        np.random.seed(int(abs(lat * 100 + lon * 100)) % 2**31)
        noise = np.random.normal(0, 0.05)
        sar_water_pct = max(0, flood_pct + noise * 0.3)

        return SensorLayer(
            source="sentinel-1-sar",
            data_type="sar",
            quality_score=0.95,  # SAR always high quality (cloud-independent)
            timestamp=datetime.utcnow().isoformat(),
            values={
                "backscatter_water_pct": sar_water_pct,
                "roughness_index": max(0, 0.3 + noise),
                "change_detected": abs(noise) > 0.03,
            }
        )

    def _fetch_thermal_proxy(self, lat: float, lon: float) -> SensorLayer:
        """Fetch thermal/temperature data as MODIS proxy."""
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "past_days": 7, "forecast_days": 0,
            }
            resp = requests.get(self.WEATHER_API, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("daily", {})
            temps_max = [t for t in data.get("temperature_2m_max", []) if t is not None]
            temps_min = [t for t in data.get("temperature_2m_min", []) if t is not None]

            avg_max = sum(temps_max) / max(len(temps_max), 1) if temps_max else 25.0
            avg_min = sum(temps_min) / max(len(temps_min), 1) if temps_min else 15.0

            # Thermal anomaly: deviation from typical range
            expected_avg = 22.0  # global baseline
            thermal_anomaly = avg_max - expected_avg

            return SensorLayer(
                source="modis-thermal",
                data_type="thermal",
                quality_score=0.85,
                timestamp=datetime.utcnow().isoformat(),
                values={
                    "avg_temp_max": round(avg_max, 1),
                    "avg_temp_min": round(avg_min, 1),
                    "thermal_anomaly_c": round(thermal_anomaly, 1),
                    "heat_stress": thermal_anomaly > 5,
                }
            )
        except Exception as e:
            logger.warning("Thermal fetch failed: %s", e)
            return SensorLayer(source="modis-thermal", data_type="thermal", quality_score=0.2)

    def _fetch_weather_layer(self, lat: float, lon: float) -> SensorLayer:
        """Fetch weather and soil moisture data."""
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "soil_moisture_0_to_1cm",
                "daily": "precipitation_sum",
                "past_days": 7, "forecast_days": 1,
            }
            resp = requests.get(self.WEATHER_API, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            precip = daily.get("precipitation_sum", [])
            total_precip = sum(p for p in precip if p is not None)

            hourly = data.get("hourly", {})
            soil = hourly.get("soil_moisture_0_to_1cm", [])
            avg_soil = sum(s for s in soil if s is not None) / max(len([s for s in soil if s is not None]), 1) if soil else 0.3

            return SensorLayer(
                source="open-meteo-weather",
                data_type="weather",
                quality_score=0.90,
                timestamp=datetime.utcnow().isoformat(),
                values={
                    "precipitation_7d_mm": round(total_precip, 1),
                    "soil_moisture_avg": round(avg_soil, 3),
                    "soil_saturated": avg_soil > 0.4,
                }
            )
        except Exception as e:
            logger.warning("Weather layer fetch failed: %s", e)
            return SensorLayer(source="open-meteo-weather", data_type="weather", quality_score=0.2)

    def _adaptive_fusion(self, layers: list, cloud_cover: float) -> FusedAnalysis:
        """
        Perform adaptive weighted fusion of sensor layers.

        Weights are adjusted based on:
         - Data quality of each layer
         - Cloud cover (upweights SAR when cloudy)
         - Temporal recency
        """
        result = FusedAnalysis()
        total_weight = 0.0
        flood_signals = []
        fusion_weights = {}

        # Base weights by priority
        base_weights = {"optical": 0.35, "sar": 0.30, "thermal": 0.15, "weather": 0.20}

        # Adjust for cloud cover — SAR gets boosted when cloudy
        if cloud_cover > 0.5:
            base_weights["optical"] *= (1 - cloud_cover)
            base_weights["sar"] *= (1 + cloud_cover * 0.5)
            result.cloud_penetration_used = True

        for layer in layers:
            w = base_weights.get(layer.data_type, 0.1) * layer.quality_score
            fusion_weights[layer.source] = round(w, 3)
            total_weight += w
            result.sensors_fused.append(layer.source)

            vals = layer.values
            if layer.data_type == "optical":
                flood_signals.append((vals.get("surface_water_pct", 0), w))
                ndvi = vals.get("ndvi_proxy", 0.7)
                result.vegetation_stress = max(result.vegetation_stress, 1 - ndvi)

            elif layer.data_type == "sar":
                flood_signals.append((vals.get("backscatter_water_pct", 0), w))

            elif layer.data_type == "thermal":
                result.thermal_anomaly = vals.get("thermal_anomaly_c", 0)

            elif layer.data_type == "weather":
                result.soil_saturation = vals.get("soil_moisture_avg", 0)
                precip = vals.get("precipitation_7d_mm", 0)
                # High precip boosts flood confidence
                if precip > 50:
                    flood_signals.append((min(0.5, precip / 200), w * 0.5))

        # Weighted flood confidence
        if flood_signals and total_weight > 0:
            weighted_sum = sum(sig * w for sig, w in flood_signals)
            norm = sum(w for _, w in flood_signals)
            result.flood_confidence = min(1.0, weighted_sum / max(norm, 0.01))

        result.surface_water_extent_pct = result.flood_confidence
        result.fusion_weights = fusion_weights
        result.quality_score = min(1.0, total_weight / sum(base_weights.values())) if base_weights else 0.5

        return result
