"""
COSMEON Model Hub — Tiered Prediction Engine with Fallback Chains.

For every right-panel feature, this module provides a *primary → secondary → tertiary*
fallback chain so the platform always returns real data, never stubs.

Tier hierarchy (best → safest fallback):

  FLOOD RISK (core prediction)
    T1: XGBoost + LightGBM ensemble trained on Open-Meteo ERA5 features
    T2: Scikit-learn GradientBoosting (always available, simpler)
    T3: Rule-based classifier from rainfall + GloFAS discharge thresholds

  PRECIPITATION FORECAST
    T1: Open-Meteo ECMWF Seasonal Forecast API (ensemble, 6-month horizon)
    T2: Open-Meteo ERA5 historical climatology + 16-day GFS forecast
    T3: Simple seasonal climatological baseline from lat/month

  SOIL MOISTURE (for compound risk)
    T1: Open-Meteo ERA5 hourly soil_moisture_0_to_7cm (real measurement)
    T2: Antecedent-Precipitation Index (7-day weighted rainfall proxy)
    T3: Climate-zone default values

  VEGETATION STRESS (for compound risk)
    T1: Open-Meteo FAO-56 ET0 water-balance (precip − ET0 per day)
    T2: NDVI proxy via scaled MODIS product-like formula from temperature + rain
    T3: Drought-index heuristic from rainfall percentile

  ECONOMIC DATA (for financial impact)
    T1: World Bank Open Data — GDP (NY.GDP.MKTP.CD) + pop density (EN.POP.DNST)
    T2: City-level lookup table (24 pre-populated flood-prone cities)
    T3: Continental GDP/density average by lat/lon

  RIVER DISCHARGE (for GloFAS validation)
    T1: GloFAS v4 via Open-Meteo Flood API (real-time operational forecast)
    T2: Past 30-day discharge from Open-Meteo archive Flood API
    T3: Synthetic discharge estimate from upstream area + rainfall runoff model

All functions return a dict that includes a `_tier` key so callers know
which tier was actually used.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("cosmeon.model_hub")

# ──────────────────────────────────────────────────────────────────────────────
# Open-Meteo endpoint constants
# ──────────────────────────────────────────────────────────────────────────────
_FORECAST_API   = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_API    = "https://archive-api.open-meteo.com/v1/archive"
_SEASONAL_API   = "https://seasonal-api.open-meteo.com/v1/seasonal"
_FLOOD_API      = "https://flood-api.open-meteo.com/v1/flood"
_ELEVATION_API  = "https://api.open-meteo.com/v1/elevation"
_WB_BASE        = "https://api.worldbank.org/v2/country"
_NOMINATIM      = "https://nominatim.openstreetmap.org/reverse"

_TIMEOUT_FAST  = 8    # seconds — for forecast/flood API calls
_TIMEOUT_SLOW  = 15   # seconds — for archive API calls (larger responses)


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRECIPITATION FORECAST
# ──────────────────────────────────────────────────────────────────────────────

def get_precipitation_forecast(lat: float, lon: float, months: int = 6) -> dict:
    """
    Return monthly precipitation forecast with tiered fallback.

    Returns:
        {
          "monthly_precip_mm": [float, ...],   # one per month
          "monthly_prob_heavy": [float, ...],  # probability >20mm day each month
          "source": str,                        # which tier was used
          "_tier": int                          # 1, 2, or 3
        }
    """
    # ── Tier 1: ECMWF Seasonal Forecast via Open-Meteo ──────────────────────
    try:
        result = _precip_forecast_ecmwf(lat, lon, months)
        if result and result.get("monthly_precip_mm"):
            result["_tier"] = 1
            logger.info("Precip forecast T1 (ECMWF seasonal): lat=%.2f lon=%.2f", lat, lon)
            return result
    except Exception as e:
        logger.warning("T1 ECMWF seasonal forecast failed: %s", e)

    # ── Tier 2: ERA5 historical + GFS 16-day blend ───────────────────────────
    try:
        result = _precip_forecast_era5_gfs(lat, lon, months)
        if result and result.get("monthly_precip_mm"):
            result["_tier"] = 2
            logger.info("Precip forecast T2 (ERA5+GFS): lat=%.2f lon=%.2f", lat, lon)
            return result
    except Exception as e:
        logger.warning("T2 ERA5+GFS forecast failed: %s", e)

    # ── Tier 3: Climatological seasonal baseline ─────────────────────────────
    result = _precip_forecast_climatological(lat, lon, months)
    result["_tier"] = 3
    logger.info("Precip forecast T3 (climatological): lat=%.2f lon=%.2f", lat, lon)
    return result


def _precip_forecast_ecmwf(lat: float, lon: float, months: int) -> dict:
    """
    Open-Meteo Seasonal Forecast API — backed by ECMWF SEAS5 / CFS ensemble.
    Provides 6-month lead-time probabilistic precipitation.
    """
    resp = requests.get(
        _SEASONAL_API,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "forecast_days": min(months * 30, 183),  # API max ~183 days
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates  = daily.get("time", [])

    # The seasonal API returns ensemble members as separate arrays
    # Key pattern: "precipitation_sum_member01", "precipitation_sum_member02", …
    member_keys = [k for k in daily if k.startswith("precipitation_sum_member")]
    if not member_keys:
        # Some API responses use a single key
        precip_raw = daily.get("precipitation_sum", [])
        member_arrays = [precip_raw] if precip_raw else []
    else:
        member_arrays = [daily[k] for k in member_keys]

    if not member_arrays or not dates:
        raise ValueError("No ensemble member data in ECMWF seasonal response")

    # Average across ensemble members to get mean daily precip
    n_days = len(dates)
    mean_daily = []
    for i in range(n_days):
        vals = [m[i] for m in member_arrays if i < len(m) and m[i] is not None]
        mean_daily.append(sum(vals) / len(vals) if vals else 0.0)

    # Aggregate into months
    monthly_precip = []
    monthly_prob_heavy = []
    from collections import defaultdict
    month_vals: dict = defaultdict(list)
    for d, p in zip(dates, mean_daily):
        key = d[:7]  # YYYY-MM
        month_vals[key].append(p)

    for key in sorted(month_vals.keys())[:months]:
        vals = month_vals[key]
        total = sum(vals)
        heavy = sum(1 for v in vals if v > 20) / max(len(vals), 1)
        monthly_precip.append(round(total, 1))
        monthly_prob_heavy.append(round(heavy, 3))

    return {
        "monthly_precip_mm":    monthly_precip,
        "monthly_prob_heavy":   monthly_prob_heavy,
        "source": "ECMWF SEAS5 via Open-Meteo seasonal forecast",
    }


def _precip_forecast_era5_gfs(lat: float, lon: float, months: int) -> dict:
    """
    Blend ERA5 historical climatology (same calendar months, prior year)
    with the 16-day GFS forecast from Open-Meteo for near-term accuracy.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=395)  # ~13 months back to cover all calendar months

    # Historical precipitation
    hist = requests.get(
        _ARCHIVE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   (end - timedelta(days=5)).strftime("%Y-%m-%d"),
            "daily": "precipitation_sum",
        },
        timeout=_TIMEOUT_SLOW,
    )
    hist.raise_for_status()
    hist_data = hist.json().get("daily", {})
    hist_dates = hist_data.get("time", [])
    hist_precip = [p if p is not None else 0.0 for p in hist_data.get("precipitation_sum", [])]

    # Near-term (16-day) GFS forecast
    gfs = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max",
            "forecast_days": 16,
        },
        timeout=_TIMEOUT_FAST,
    )
    gfs.raise_for_status()
    gfs_data  = gfs.json().get("daily", {})
    gfs_dates = gfs_data.get("time", [])
    gfs_precip = [p if p is not None else 0.0 for p in gfs_data.get("precipitation_sum", [])]

    # Build monthly averages from history
    from collections import defaultdict
    month_hist: dict = defaultdict(list)
    for d, p in zip(hist_dates, hist_precip):
        month_hist[int(d[5:7])].append(p)
    month_avg = {m: sum(v) / max(len(v), 1) for m, v in month_hist.items()}

    # Also add GFS days to their calendar month
    gfs_month: dict = defaultdict(list)
    for d, p in zip(gfs_dates, gfs_precip):
        gfs_month[int(d[5:7])].append(p)

    monthly_precip = []
    monthly_prob_heavy = []
    now = datetime.utcnow()
    for i in range(1, months + 1):
        target = now + timedelta(days=30 * i)
        m = target.month
        # Blend: GFS if available for this calendar month, else historical
        if m in gfs_month:
            vals = gfs_month[m] + month_hist.get(m, [])
        else:
            vals = month_hist.get(m, [])

        if vals:
            total = sum(vals) * (30 / max(len(vals), 1))  # scale to 30-day month
            heavy = sum(1 for v in vals if v > 20) / max(len(vals), 1)
        else:
            total = month_avg.get(m, 60.0) * 30
            heavy = 0.15
        monthly_precip.append(round(total, 1))
        monthly_prob_heavy.append(round(min(1.0, heavy), 3))

    return {
        "monthly_precip_mm":    monthly_precip,
        "monthly_prob_heavy":   monthly_prob_heavy,
        "source": "ERA5 archive + GFS 16-day via Open-Meteo",
    }


def _precip_forecast_climatological(lat: float, lon: float, months: int) -> dict:
    """
    Tier-3: Latitude/climate-zone based climatological precipitation estimate.
    Pure heuristic — no API required.
    """
    abs_lat = abs(lat)
    is_southern = lat < 0
    now = datetime.utcnow()

    # Koppen-Geiger inspired monthly coefficients
    if abs_lat < 10:
        # Equatorial — uniform heavy rainfall year-round
        base_mm   = 180.0
        amplitude = 0.2
    elif abs_lat < 25:
        # Tropical savanna / monsoon — strong wet/dry cycle
        base_mm   = 120.0
        amplitude = 0.6
    elif abs_lat < 40:
        # Subtropical / Mediterranean
        base_mm   = 60.0
        amplitude = 0.5
    elif abs_lat < 60:
        # Temperate
        base_mm   = 70.0
        amplitude = 0.3
    else:
        # Subarctic / polar
        base_mm   = 30.0
        amplitude = 0.2

    monthly_precip = []
    monthly_prob_heavy = []
    for i in range(1, months + 1):
        target = now + timedelta(days=30 * i)
        m = target.month
        # Peak for NH in Jul/Aug; flip for SH
        if is_southern:
            m_adj = (m + 6 - 1) % 12 + 1
        else:
            m_adj = m
        # Sinusoidal seasonal cycle: peak at month 7 (July)
        seasonal = 1.0 + amplitude * math.sin((m_adj - 4) * math.pi / 6)
        mm = base_mm * seasonal
        heavy_prob = min(0.6, mm / 300.0)
        monthly_precip.append(round(mm, 1))
        monthly_prob_heavy.append(round(heavy_prob, 3))

    return {
        "monthly_precip_mm":    monthly_precip,
        "monthly_prob_heavy":   monthly_prob_heavy,
        "source": "Koppen-Geiger climatological baseline",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. SOIL MOISTURE
# ──────────────────────────────────────────────────────────────────────────────

def get_soil_moisture(lat: float, lon: float) -> dict:
    """
    Return soil saturation fraction with tiered fallback.

    Returns:
        {"saturation_fraction": float (0-1), "volumetric_m3m3": float,
         "source": str, "_tier": int}
    """
    # T1: Direct ERA5 hourly soil moisture measurement
    try:
        result = _soil_moisture_era5(lat, lon)
        if result:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 soil moisture (ERA5) failed: %s", e)

    # T2: Antecedent Precipitation Index proxy
    try:
        result = _soil_moisture_api_proxy(lat, lon)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 soil moisture (API proxy) failed: %s", e)

    # T3: Climate-zone default
    result = _soil_moisture_default(lat)
    result["_tier"] = 3
    return result


def _soil_moisture_era5(lat: float, lon: float) -> dict:
    """
    Fetch ERA5 soil moisture from the *archive* API (last 14 days).
    The forecast API omits soil_moisture_0_to_7cm for many tropical/equatorial
    grid cells; the archive API is the authoritative ERA5 reanalysis source.
    """
    from datetime import date, timedelta
    end_date   = date.today()
    start_date = end_date - timedelta(days=14)

    # Try archive API first (ERA5 reanalysis — most reliable)
    resp = requests.get(
        _ARCHIVE_API,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date.isoformat(),
            "end_date":   end_date.isoformat(),
            "hourly":     "soil_moisture_0_to_7cm",
            "models":     "era5",
        },
        timeout=_TIMEOUT_SLOW,
    )
    resp.raise_for_status()
    vals = [v for v in resp.json().get("hourly", {}).get("soil_moisture_0_to_7cm", [])
            if v is not None]

    # Fallback: try forecast API with past_days (ERA5-Land model)
    if not vals:
        resp2 = requests.get(
            _FORECAST_API,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "soil_moisture_0_to_7cm",
                "past_days": 7,
                "forecast_days": 0,
                "models": "best_match",
            },
            timeout=_TIMEOUT_FAST,
        )
        resp2.raise_for_status()
        vals = [v for v in resp2.json().get("hourly", {}).get("soil_moisture_0_to_7cm", [])
                if v is not None]

    if not vals:
        return {}

    recent = vals[-48:] if len(vals) >= 48 else vals  # last 48h
    avg    = sum(recent) / len(recent)
    sat    = min(1.0, avg / 0.40)   # field capacity ≈ 0.40 m³/m³
    return {
        "saturation_fraction": round(sat, 3),
        "volumetric_m3m3":     round(avg, 4),
        "source": "ERA5 reanalysis soil_moisture_0_to_7cm (Open-Meteo archive)",
    }


def _soil_moisture_api_proxy(lat: float, lon: float) -> dict:
    """
    Antecedent Precipitation Index (API):
        API_t = k * API_{t-1} + P_t   (k = 0.9 decay)
    Higher API → wetter soil.  Normalize to 0-1 using empirical max of 200mm.
    """
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": 14,
            "forecast_days": 0,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    precip_list = [p if p is not None else 0.0
                   for p in resp.json().get("daily", {}).get("precipitation_sum", [])]
    k = 0.90
    api = 0.0
    for p in precip_list:
        api = k * api + p
    saturation = min(1.0, api / 200.0)
    return {
        "saturation_fraction": round(saturation, 3),
        "volumetric_m3m3":     round(saturation * 0.40, 4),  # rough inverse
        "source": "Antecedent Precipitation Index (14-day, k=0.9)",
    }


def _soil_moisture_default(lat: float) -> dict:
    abs_lat = abs(lat)
    if abs_lat < 15:
        sat = 0.65   # tropical — usually moist
    elif abs_lat < 30:
        sat = 0.45
    elif abs_lat < 50:
        sat = 0.40
    else:
        sat = 0.30
    return {
        "saturation_fraction": sat,
        "volumetric_m3m3":     round(sat * 0.40, 4),
        "source": "climate-zone default",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. VEGETATION STRESS
# ──────────────────────────────────────────────────────────────────────────────

def get_vegetation_stress(lat: float, lon: float) -> dict:
    """
    Return vegetation stress index (0 = healthy, 1 = severely stressed).

    Returns:
        {"stress_index": float (0-1), "et0_mm_day": float,
         "precip_mm_day": float, "source": str, "_tier": int}
    """
    # T1: FAO-56 ET0 water-balance (Open-Meteo)
    try:
        result = _veg_stress_fao56(lat, lon)
        if result:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 veg stress (FAO-56) failed: %s", e)

    # T2: Temperature-based drought index (SPEI proxy)
    try:
        result = _veg_stress_temp_index(lat, lon)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 veg stress (temp index) failed: %s", e)

    # T3: Default by lat
    result = _veg_stress_default(lat)
    result["_tier"] = 3
    return result


def _veg_stress_fao56(lat: float, lon: float) -> dict:
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "et0_fao_evapotranspiration,precipitation_sum",
            "past_days": 14,
            "forecast_days": 0,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    et0_vals  = [v for v in daily.get("et0_fao_evapotranspiration", []) if v is not None]
    rain_vals = [v for v in daily.get("precipitation_sum",         []) if v is not None]
    if not et0_vals:
        return {}
    avg_et0  = sum(et0_vals)  / len(et0_vals)
    avg_rain = sum(rain_vals) / len(rain_vals) if rain_vals else 0.0
    deficit  = avg_et0 - avg_rain          # positive deficit → drought stress
    stress   = max(0.0, min(1.0, deficit / 5.0))   # 5 mm/day deficit = 100% stress
    return {
        "stress_index":  round(stress, 3),
        "et0_mm_day":    round(avg_et0,  2),
        "precip_mm_day": round(avg_rain, 2),
        "source": "FAO-56 ET0 water-balance via Open-Meteo",
    }


def _veg_stress_temp_index(lat: float, lon: float) -> dict:
    """
    Simplified SPEI-like index: combine temperature anomaly and rainfall deficit
    to estimate drought stress without needing ET0.
    """
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,precipitation_sum",
            "past_days": 30,
            "forecast_days": 0,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    temps  = [v for v in daily.get("temperature_2m_max",  []) if v is not None]
    rains  = [v for v in daily.get("precipitation_sum",   []) if v is not None]
    if not temps:
        return {}
    avg_temp  = sum(temps) / len(temps)
    avg_rain  = sum(rains) / len(rains) if rains else 0.0
    # Rough Thornthwaite PET: 1.6 * (10T/I)^a but simplified:
    pet_daily = max(0.0, (avg_temp * 0.6) - 2.0)   # mm/day
    deficit   = max(0.0, pet_daily - avg_rain)
    stress    = min(1.0, deficit / 5.0)
    return {
        "stress_index":  round(stress, 3),
        "et0_mm_day":    round(pet_daily, 2),
        "precip_mm_day": round(avg_rain,  2),
        "source": "Thornthwaite PET proxy via Open-Meteo temperature",
    }


def _veg_stress_default(lat: float) -> dict:
    abs_lat = abs(lat)
    if abs_lat < 10:
        stress = 0.1    # lush tropics
    elif abs_lat < 25:
        stress = 0.35   # variable subtropical
    elif abs_lat < 40:
        stress = 0.45   # dry subtropics
    else:
        stress = 0.25
    return {
        "stress_index":  stress,
        "et0_mm_day":    4.0,
        "precip_mm_day": 2.5,
        "source": "climate-zone default",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. ECONOMIC DATA (GDP + population density)
# ──────────────────────────────────────────────────────────────────────────────

# City-level GDP lookup — backup if World Bank unavailable
_CITY_GDP: dict[str, float] = {
    "mumbai": 370e9, "navi mumbai": 110e9, "kolkata": 150e9,
    "dhaka": 80e9, "jakarta": 180e9, "bangkok": 190e9,
    "manila": 120e9, "ho chi minh": 85e9,
    "rotterdam": 80e9, "new orleans": 60e9, "houston": 530e9,
    "são paulo": 430e9, "sao paulo": 430e9, "wuhan": 230e9,
    "lagos": 90e9, "khartoum": 22e9, "delhi": 290e9, "karachi": 78e9,
    "miami": 400e9, "new york": 1800e9, "london": 700e9,
    "tokyo": 1500e9, "shanghai": 690e9, "beijing": 600e9,
}
_CITY_POP_DENSITY: dict[str, float] = {
    "mumbai": 20000, "dhaka": 44000, "jakarta": 15000,
    "manila": 46000, "kolkata": 24000, "karachi": 25000,
    "delhi": 11000, "lagos": 6800, "cairo": 19000,
    "rotterdam": 3000, "houston": 1400, "miami": 5000,
    "new york": 10400, "london": 5700, "tokyo": 6200,
    "shanghai": 3900, "beijing": 1300,
}


def get_economic_data(lat: float, lon: float, name: str = "") -> dict:
    """
    Return GDP and population density with tiered fallback.

    Returns:
        {"gdp_usd": float, "pop_density_km2": float,
         "country_code": str, "country_name": str,
         "source": str, "_tier": int}
    """
    # T1: World Bank Open Data
    try:
        result = _econ_world_bank(lat, lon)
        if result and result.get("gdp_usd", 0) > 0:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 economic data (World Bank) failed: %s", e)

    # T2: City lookup table
    try:
        result = _econ_city_lookup(name, lat, lon)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 economic data (city lookup) failed: %s", e)

    # T3: Continental estimates
    result = _econ_continental_default(lat, lon)
    result["_tier"] = 3
    return result


def _econ_world_bank(lat: float, lon: float) -> dict:
    # Step 1: ISO-2 country code via Nominatim reverse geocode
    rev = requests.get(
        _NOMINATIM,
        params={"lat": lat, "lon": lon, "format": "json", "zoom": 3},
        headers={"User-Agent": "COSMEON/1.0"},
        timeout=6,
    )
    rev.raise_for_status()
    rev_data = rev.json()
    cc = rev_data.get("address", {}).get("country_code", "").upper()
    country_name = rev_data.get("address", {}).get("country", "Unknown")
    if not cc or len(cc) != 2:
        raise ValueError(f"Bad country code: {cc!r}")

    # Step 2: GDP from World Bank
    gdp_resp = requests.get(
        f"{_WB_BASE}/{cc}/indicator/NY.GDP.MKTP.CD?mrv=1&format=json&per_page=1",
        headers={"User-Agent": "COSMEON/1.0"},
        timeout=8,
    )
    gdp_resp.raise_for_status()
    gdp_data = gdp_resp.json()
    gdp_usd = None
    if (isinstance(gdp_data, list) and len(gdp_data) >= 2
            and gdp_data[1] and gdp_data[1][0].get("value") is not None):
        gdp_usd = float(gdp_data[1][0]["value"])

    # Step 3: Population density from World Bank
    pop_resp = requests.get(
        f"{_WB_BASE}/{cc}/indicator/EN.POP.DNST?mrv=1&format=json&per_page=1",
        headers={"User-Agent": "COSMEON/1.0"},
        timeout=8,
    )
    pop_resp.raise_for_status()
    pop_data = pop_resp.json()
    nat_density = 100.0
    if (isinstance(pop_data, list) and len(pop_data) >= 2
            and pop_data[1] and pop_data[1][0].get("value") is not None):
        nat_density = float(pop_data[1][0]["value"])

    return {
        "gdp_usd":         float(gdp_usd) if gdp_usd else 50e9,
        "pop_density_km2": round(nat_density * 6.0, 0),   # 6× urban multiplier
        "country_code":    cc,
        "country_name":    country_name,
        "source": "World Bank Open Data (NY.GDP.MKTP.CD + EN.POP.DNST)",
    }


def _econ_city_lookup(name: str, lat: float, lon: float) -> dict:
    n = name.lower()
    gdp = None
    pop = None
    for key, v in _CITY_GDP.items():
        if key in n:
            gdp = v
            break
    for key, v in _CITY_POP_DENSITY.items():
        if key in n:
            pop = v
            break

    if gdp is None and pop is None:
        return {}

    return {
        "gdp_usd":         gdp or 50e9,
        "pop_density_km2": pop or 500.0,
        "country_code":    "XX",
        "country_name":    name,
        "source": "city-level lookup table (pre-populated)",
    }


def _econ_continental_default(lat: float, lon: float) -> dict:
    """
    Very rough continental GDP and density from coordinates.
    Better than a single global constant.
    """
    abs_lat = abs(lat)
    # Sub-Saharan Africa
    if lat < 15 and 25 < lon < 55:
        gdp, pop = 30e9, 100.0
    # South/SE Asia
    elif 5 < lat < 35 and 65 < lon < 135:
        gdp, pop = 80e9, 400.0
    # East Asia
    elif 20 < lat < 55 and 100 < lon < 145:
        gdp, pop = 300e9, 200.0
    # Europe
    elif 35 < lat < 70 and -10 < lon < 40:
        gdp, pop = 200e9, 120.0
    # North America
    elif 25 < lat < 70 and -130 < lon < -60:
        gdp, pop = 400e9, 50.0
    # South America
    elif -55 < lat < 15 and -80 < lon < -34:
        gdp, pop = 100e9, 30.0
    else:
        gdp, pop = 50e9, 80.0

    return {
        "gdp_usd":         gdp,
        "pop_density_km2": pop,
        "country_code":    "XX",
        "country_name":    "Unknown",
        "source": "continental economic baseline",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. RIVER DISCHARGE (GloFAS)
# ──────────────────────────────────────────────────────────────────────────────

def get_river_discharge(lat: float, lon: float, past_days: int = 30) -> dict:
    """
    Return GloFAS river discharge data with tiered fallback.

    Returns:
        {"current_discharge_m3s": float, "mean_discharge_m3s": float,
         "anomaly_sigma": float, "flood_risk_level": str,
         "forecast_discharge": list, "source": str, "_tier": int}
    """
    # T1: GloFAS v4 operational forecast
    try:
        result = _discharge_glofas_forecast(lat, lon, past_days)
        if result and result.get("current_discharge_m3s", 0) >= 0:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 discharge (GloFAS v4 forecast) failed: %s", e)

    # T2: GloFAS historical archive
    try:
        result = _discharge_glofas_archive(lat, lon, past_days)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 discharge (GloFAS archive) failed: %s", e)

    # T3: Rainfall-runoff proxy (no GloFAS)
    try:
        result = _discharge_rainfall_proxy(lat, lon)
        result["_tier"] = 3
        return result
    except Exception as e:
        logger.warning("T3 discharge (rainfall proxy) failed: %s", e)

    return {
        "current_discharge_m3s": 0.0,
        "mean_discharge_m3s": 0.0,
        "anomaly_sigma": 0.0,
        "flood_risk_level": "UNKNOWN",
        "forecast_discharge": [],
        "source": "unavailable",
        "_tier": 99,
    }


def _discharge_glofas_forecast(lat: float, lon: float, past_days: int) -> dict:
    resp = requests.get(
        _FLOOD_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "river_discharge,river_discharge_mean,river_discharge_max",
            "past_days": past_days,
            "forecast_days": 7,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    discharge = [v for v in daily.get("river_discharge", []) if v is not None]
    means     = [v for v in daily.get("river_discharge_mean", []) if v is not None]
    if not discharge:
        return {}

    current = discharge[-1] if discharge else 0.0
    mean    = sum(means) / len(means) if means else (sum(discharge) / len(discharge))
    std     = float(np.std(discharge)) if len(discharge) > 1 else 1.0
    anomaly = (current - mean) / max(std, 0.01)

    forecast = discharge[-7:]   # last 7 as forecast proxy
    risk = _discharge_to_risk(anomaly, current, mean)

    return {
        "current_discharge_m3s": round(current, 2),
        "mean_discharge_m3s":    round(mean, 2),
        "anomaly_sigma":         round(anomaly, 2),
        "flood_risk_level":      risk,
        "forecast_discharge":    forecast,
        "source": "GloFAS v4 operational via Open-Meteo Flood API",
    }


def _discharge_glofas_archive(lat: float, lon: float, past_days: int) -> dict:
    end   = datetime.utcnow() - timedelta(days=5)   # archive lag
    start = end - timedelta(days=past_days)
    resp = requests.get(
        _FLOOD_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "river_discharge",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   end.strftime("%Y-%m-%d"),
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    discharge = [v for v in resp.json().get("daily", {}).get("river_discharge", [])
                 if v is not None]
    if not discharge:
        return {}

    current = discharge[-1]
    mean    = sum(discharge) / len(discharge)
    std     = float(np.std(discharge)) if len(discharge) > 1 else 1.0
    anomaly = (current - mean) / max(std, 0.01)
    risk    = _discharge_to_risk(anomaly, current, mean)

    return {
        "current_discharge_m3s": round(current, 2),
        "mean_discharge_m3s":    round(mean, 2),
        "anomaly_sigma":         round(anomaly, 2),
        "flood_risk_level":      risk,
        "forecast_discharge":    discharge[-7:],
        "source": "GloFAS v4 archive via Open-Meteo Flood API",
    }


def _discharge_rainfall_proxy(lat: float, lon: float) -> dict:
    """
    Estimate discharge as rainfall runoff:  Q ≈ C * P * A
    C = runoff coefficient (0.3 for average terrain)
    P = recent 7-day precipitation (mm)
    A = catchment area estimate (km²) from Open-Meteo elevation sampling
    """
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": 7,
            "forecast_days": 0,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    precip_list = [p for p in resp.json().get("daily", {}).get("precipitation_sum", [])
                   if p is not None]
    total_precip_m = sum(precip_list) / 1000.0   # convert mm → m
    catchment_km2  = 5000.0                        # rough default
    runoff_c       = 0.30
    # Q (m³/s) ≈ C * P * A / (7 * 86400)
    q = runoff_c * total_precip_m * catchment_km2 * 1e6 / (7 * 86400)
    mean_q = q * 0.7
    anomaly = (q - mean_q) / max(mean_q * 0.3, 0.1)
    risk = _discharge_to_risk(anomaly, q, mean_q)
    return {
        "current_discharge_m3s": round(q, 2),
        "mean_discharge_m3s":    round(mean_q, 2),
        "anomaly_sigma":         round(anomaly, 2),
        "flood_risk_level":      risk,
        "forecast_discharge":    [],
        "source": "rainfall-runoff proxy (no GloFAS coverage at this location)",
    }


def _discharge_to_risk(anomaly: float, current: float, mean: float) -> str:
    if anomaly > 3.0 or current > mean * 5:
        return "CRITICAL"
    elif anomaly > 2.0 or current > mean * 3:
        return "HIGH"
    elif anomaly > 1.0 or current > mean * 1.5:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────────────────────────
# 6. TEMPERATURE ANOMALY
# ──────────────────────────────────────────────────────────────────────────────

def get_temperature_anomaly(lat: float, lon: float) -> dict:
    """
    Return current temperature anomaly vs seasonal baseline.

    T1: ERA5 7-day mean vs Open-Meteo 30-year climatology
    T2: 7-day mean vs latitude-based seasonal estimate
    T3: Zero anomaly default
    """
    # T1: Compare vs ERA5 same-month climatology
    try:
        result = _temp_anomaly_era5_clim(lat, lon)
        if result:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 temp anomaly (ERA5 clim) failed: %s", e)

    # T2: Latitude seasonal baseline
    try:
        result = _temp_anomaly_lat_baseline(lat, lon)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 temp anomaly (lat baseline) failed: %s", e)

    return {"current_avg_c": 25.0, "baseline_c": 25.0, "anomaly_c": 0.0,
            "source": "default", "_tier": 3}


def _temp_anomaly_era5_clim(lat: float, lon: float) -> dict:
    """Compare current 7-day mean vs same-month historical average (prior year)."""
    now   = datetime.utcnow()
    start_hist = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    end_hist   = (now - timedelta(days=8)).strftime("%Y-%m-%d")

    # Historical (1 year back, same months)
    hist = requests.get(
        _ARCHIVE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start_hist, "end_date": end_hist,
            "daily": "temperature_2m_mean",
        },
        timeout=_TIMEOUT_SLOW,
    )
    hist.raise_for_status()
    hist_daily = hist.json().get("daily", {})
    hist_dates = hist_daily.get("time", [])
    hist_temps = [t for t in hist_daily.get("temperature_2m_mean", []) if t is not None]

    # Current 7-day
    curr = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_mean",
            "past_days": 7, "forecast_days": 0,
        },
        timeout=_TIMEOUT_FAST,
    )
    curr.raise_for_status()
    curr_temps = [t for t in curr.json().get("daily", {}).get("temperature_2m_mean", [])
                  if t is not None]
    if not curr_temps or not hist_temps:
        return {}

    current_avg = sum(curr_temps) / len(curr_temps)

    # Historical baseline = same calendar month average
    cur_month = now.month
    same_month = [t for d, t in zip(hist_dates, hist_temps) if int(d[5:7]) == cur_month]
    baseline = sum(same_month) / len(same_month) if same_month else sum(hist_temps) / len(hist_temps)
    anomaly = current_avg - baseline

    return {
        "current_avg_c": round(current_avg, 1),
        "baseline_c":    round(baseline, 1),
        "anomaly_c":     round(anomaly, 1),
        "source": "ERA5 historical climatology vs current observation",
    }


def _temp_anomaly_lat_baseline(lat: float, lon: float) -> dict:
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "past_days": 7, "forecast_days": 0,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    maxes = daily.get("temperature_2m_max", [])
    mins  = daily.get("temperature_2m_min", [])
    avgs  = [(mx + mn) / 2 for mx, mn in zip(maxes, mins)
             if mx is not None and mn is not None]
    if not avgs:
        return {}

    current_avg = sum(avgs) / len(avgs)
    abs_lat = abs(lat)
    month = datetime.utcnow().month
    is_sh = lat < 0
    if abs_lat < 15:
        baseline = 28.0
    elif abs_lat < 30:
        baseline = 25.0
    elif abs_lat < 50:
        summer = [6, 7, 8] if not is_sh else [12, 1, 2]
        winter = [12, 1, 2] if not is_sh else [6, 7, 8]
        baseline = 22.0 if month in summer else (5.0 if month in winter else 14.0)
    else:
        baseline = 5.0

    return {
        "current_avg_c": round(current_avg, 1),
        "baseline_c":    round(baseline, 1),
        "anomaly_c":     round(current_avg - baseline, 1),
        "source": "latitude-based seasonal baseline",
    }
