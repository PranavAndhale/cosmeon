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
            response = requests.get(self.weather_api, params=params, timeout=30)
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
            response = requests.get(self.elevation_api, params=params, timeout=30)
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

        # Elevation factor (only apply when we have a real non-zero value;
        # elevation == 0 means the API failed, not that terrain is at sea level)
        mean_elev = elevation.get("mean_elevation_m", 100)
        low_pct = elevation.get("low_elevation_pct", 0)

        if mean_elev > 0:
            if mean_elev < 10:
                multiplier *= 1.5
            elif mean_elev < 50:
                multiplier *= 1.2

        if low_pct > 0:
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
        lat = (bbox[1] + bbox[3]) / 2
        lon = (bbox[0] + bbox[2]) / 2
        return self.get_risk_factors_by_coords(lat, lon)

    def fetch_temperature_anomaly(self, lat: float, lon: float) -> dict:
        """
        Fetch current 7-day average temperature and estimate anomaly vs seasonal baseline.

        Uses the Open-Meteo forecast API for current temps, and a latitude-based
        seasonal baseline so we don't need a second archive call.

        Returns:
            {"current_avg_c": float, "baseline_c": float, "anomaly_c": float}
        """
        try:
            from datetime import datetime
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "past_days": 7,
                "forecast_days": 0,
            }
            response = requests.get(self.weather_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            daily = data.get("daily", {})
            maxes = daily.get("temperature_2m_max", [])
            mins = daily.get("temperature_2m_min", [])
            # Mean of daily averages
            avgs = [
                (mx + mn) / 2
                for mx, mn in zip(maxes, mins)
                if mx is not None and mn is not None
            ]
            current_avg = sum(avgs) / max(len(avgs), 1)

            # Latitude-based seasonal baseline (Northern Hemisphere biased, flipped for SH)
            month = datetime.utcnow().month
            abs_lat = abs(lat)
            is_southern = lat < 0
            # Estimate climatological mean temperature from latitude
            if abs_lat < 15:
                clim_base = 28.0   # Tropical
            elif abs_lat < 30:
                clim_base = 25.0   # Subtropical
            elif abs_lat < 50:
                # Temperate: peaks in summer (NH: Jul/Aug, SH: Jan/Feb)
                summer_months = [6, 7, 8] if not is_southern else [12, 1, 2]
                winter_months = [12, 1, 2] if not is_southern else [6, 7, 8]
                if month in summer_months:
                    clim_base = 22.0
                elif month in winter_months:
                    clim_base = 5.0
                else:
                    clim_base = 14.0
            else:
                clim_base = 5.0    # High latitude

            anomaly = round(current_avg - clim_base, 1)
            logger.info(
                "Temp anomaly lat=%.2f: current=%.1f°C baseline=%.1f°C anomaly=%+.1f°C",
                lat, current_avg, clim_base, anomaly,
            )
            return {
                "current_avg_c": round(current_avg, 1),
                "baseline_c": round(clim_base, 1),
                "anomaly_c": anomaly,
            }
        except Exception as e:
            logger.warning("fetch_temperature_anomaly failed: %s", e)
            return {"current_avg_c": 25.0, "baseline_c": 25.0, "anomaly_c": 0.0}

    def fetch_soil_moisture(self, lat: float, lon: float) -> dict:
        """
        Fetch direct soil moisture (0-7 cm) from Open-Meteo ERA5 hourly data.
        This is a physically measured quantity — not a computed proxy.

        Returns:
            {"volumetric_0to7cm": float (m³/m³), "saturation_fraction": float (0-1), "source": str}
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "soil_moisture_0_to_7cm",
                "past_days": 7,
                "forecast_days": 0,
            }
            response = requests.get(self.weather_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            sm_vals = [v for v in hourly.get("soil_moisture_0_to_7cm", []) if v is not None]

            if not sm_vals:
                return {"volumetric_0to7cm": 0.3, "saturation_fraction": 0.5, "source": "fallback"}

            # Average of last 24 hours for current conditions
            recent = sm_vals[-24:] if len(sm_vals) >= 24 else sm_vals
            avg_sm = sum(recent) / max(len(recent), 1)

            # Field capacity for typical loam soil: 0.35–0.40 m³/m³
            # Values near/above 0.40 indicate near-saturation
            FIELD_CAPACITY = 0.40
            saturation = min(1.0, avg_sm / FIELD_CAPACITY)

            logger.info(
                "Soil moisture lat=%.2f lon=%.2f: %.4f m³/m³ → %.0f%% saturation",
                lat, lon, avg_sm, saturation * 100,
            )
            return {
                "volumetric_0to7cm": round(avg_sm, 4),
                "saturation_fraction": round(saturation, 3),
                "source": "open-meteo-era5",
            }
        except Exception as e:
            logger.warning("fetch_soil_moisture failed: %s", e)
            return {"volumetric_0to7cm": 0.3, "saturation_fraction": 0.5, "source": "fallback"}

    def fetch_vegetation_stress(self, lat: float, lon: float) -> dict:
        """
        Estimate vegetation stress using the Penman-Monteith ET0 water balance.

        Logic: water_balance = precipitation − ET0 over 14 days.
        A negative balance (more ET demand than rain) = drought = high stress.
        A positive balance (rain exceeds ET demand) = adequate moisture = low stress.

        This approach is used in agrometeorology (FAO-56 method).

        Returns:
            {"et0_mm_day": float, "precip_mm_day": float, "stress_index": float (0-1), "source": str}
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "et0_fao_evapotranspiration,precipitation_sum",
                "past_days": 14,
                "forecast_days": 0,
            }
            response = requests.get(self.weather_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            et0_vals  = [v for v in daily.get("et0_fao_evapotranspiration", []) if v is not None]
            rain_vals = [v for v in daily.get("precipitation_sum", [])          if v is not None]

            if not et0_vals:
                return {"et0_mm_day": 4.0, "precip_mm_day": 2.0, "stress_index": 0.3, "source": "fallback"}

            avg_et0  = sum(et0_vals)  / max(len(et0_vals),  1)
            avg_rain = sum(rain_vals) / max(len(rain_vals), 1) if rain_vals else 0.0

            # Water balance per day (mm). Negative = moisture deficit.
            water_balance_mm_day = avg_rain - avg_et0

            # Normalise deficit to 0-1 stress range.
            # A deficit of ≥5 mm/day is considered severe drought.
            SEVERE_DEFICIT = 5.0
            if water_balance_mm_day >= 0:
                stress = 0.0   # surplus water — no drought stress
            else:
                stress = min(1.0, -water_balance_mm_day / SEVERE_DEFICIT)

            logger.info(
                "Veg stress lat=%.2f lon=%.2f: ET0=%.2f rain=%.2f balance=%.2f → stress=%.2f",
                lat, lon, avg_et0, avg_rain, water_balance_mm_day, stress,
            )
            return {
                "et0_mm_day":    round(avg_et0,  2),
                "precip_mm_day": round(avg_rain, 2),
                "stress_index":  round(stress,   3),
                "source": "open-meteo-fao56",
            }
        except Exception as e:
            logger.warning("fetch_vegetation_stress failed: %s", e)
            return {"et0_mm_day": 4.0, "precip_mm_day": 2.0, "stress_index": 0.3, "source": "fallback"}

    def fetch_country_gdp_pop(self, lat: float, lon: float) -> dict:
        """
        Fetch country-level GDP (USD) and population density (people/km²) from
        the World Bank Open Data API, using Nominatim to derive the ISO-2 country code.

        Provides authoritative economic data for financial impact calculations instead
        of a hardcoded city lookup table.

        Returns:
            {"gdp_usd": float, "pop_density_km2": float, "country_code": str,
             "country_name": str, "source": str}
        """
        try:
            # Step 1: Reverse geocode to get ISO-2 country code
            rev_url = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"?lat={lat}&lon={lon}&format=json&zoom=3"
            )
            rev_resp = requests.get(rev_url, headers={"User-Agent": "COSMEON/1.0"}, timeout=6)
            rev_data = rev_resp.json()
            country_code = rev_data.get("address", {}).get("country_code", "").upper()
            country_name = rev_data.get("address", {}).get("country", "Unknown")

            if not country_code or len(country_code) != 2:
                raise ValueError(f"Invalid country code: {country_code!r}")

            wb_base = "https://api.worldbank.org/v2/country"

            # Step 2: GDP (current USD) — indicator NY.GDP.MKTP.CD
            gdp_url = (
                f"{wb_base}/{country_code}/indicator/NY.GDP.MKTP.CD"
                f"?mrv=1&format=json&per_page=1"
            )
            gdp_resp = requests.get(gdp_url, headers={"User-Agent": "COSMEON/1.0"}, timeout=8)
            gdp_data = gdp_resp.json()
            gdp_usd: float = 50_000_000_000  # fallback $50B
            if (isinstance(gdp_data, list) and len(gdp_data) >= 2
                    and gdp_data[1] and gdp_data[1][0].get("value") is not None):
                gdp_usd = float(gdp_data[1][0]["value"])

            # Step 3: Population density (people/km²) — indicator EN.POP.DNST
            pop_url = (
                f"{wb_base}/{country_code}/indicator/EN.POP.DNST"
                f"?mrv=1&format=json&per_page=1"
            )
            pop_resp = requests.get(pop_url, headers={"User-Agent": "COSMEON/1.0"}, timeout=8)
            pop_data = pop_resp.json()
            national_density: float = 100.0  # fallback 100 people/km²
            if (isinstance(pop_data, list) and len(pop_data) >= 2
                    and pop_data[1] and pop_data[1][0].get("value") is not None):
                national_density = float(pop_data[1][0]["value"])

            # Urban areas typically run 5-8× the national average density.
            # Use 6× as the urban multiplier so estimates are city-scale.
            urban_density = national_density * 6.0

            logger.info(
                "World Bank data for %s (%s): GDP=$%.0fB, nat_density=%.0f/km², "
                "urban_est=%.0f/km²",
                country_name, country_code,
                gdp_usd / 1e9, national_density, urban_density,
            )
            return {
                "gdp_usd":        gdp_usd,
                "pop_density_km2": round(urban_density, 0),
                "country_code":   country_code,
                "country_name":   country_name,
                "source": "world-bank",
            }

        except Exception as e:
            logger.warning("fetch_country_gdp_pop failed: %s", e)
            return {
                "gdp_usd":        50_000_000_000,
                "pop_density_km2": 500.0,
                "country_code":   "XX",
                "country_name":   "Unknown",
                "source": "fallback",
            }

    def get_risk_factors_by_coords(self, lat: float, lon: float) -> ExternalRiskFactors:
        """
        Get all external risk factors for arbitrary coordinates.

        Enables ad-hoc analysis of any location on Earth without
        requiring a pre-registered region in the database.
        """
        logger.info("Fetching external risk factors for lat=%.2f, lon=%.2f", lat, lon)

        rainfall = self.fetch_rainfall(lat, lon)
        elevation = self.fetch_elevation(lat, lon)
        risk_multiplier = self.compute_risk_multiplier(rainfall, elevation)

        factors = ExternalRiskFactors(
            rainfall_mm=rainfall.get("total_rainfall_mm", 0),
            elevation_mean_m=elevation.get("mean_elevation_m", 0),
            elevation_min_m=elevation.get("min_elevation_m", 0),
            rainfall_anomaly=0,
            low_elevation_pct=elevation.get("low_elevation_pct", 0),
            risk_multiplier=risk_multiplier,
        )

        logger.info("External risk factors: multiplier=%.2f", risk_multiplier)
        return factors
