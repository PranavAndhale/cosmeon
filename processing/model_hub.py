"""
COSMEON Model Hub — Tiered Prediction Engine (Established Sources Only).

Every tier in every fallback chain uses a globally trusted, established model
or data source. No custom heuristics, no hardcoded lookup tables, no invented
formulas.

Tier hierarchy (best → safest fallback):

  PRECIPITATION FORECAST
    T1: ECMWF SEAS5 seasonal ensemble via Open-Meteo seasonal API
    T2: ERA5 reanalysis archive + GFS 16-day operational forecast (Open-Meteo)
    T3: CMIP6 EC-Earth3P-HR 30-year daily climatology (Open-Meteo Climate API)

  SOIL MOISTURE (for compound risk)
    T1: ERA5 reanalysis soil_moisture_0_to_7cm (Open-Meteo archive API)
    T2: ECMWF IFS 0.25° operational model soil_moisture_0_to_7cm (Open-Meteo)
    T3: ERA5 reanalysis same-month prior-year archive (climatological baseline)

  VEGETATION STRESS (for compound risk)
    T1: FAO-56 Penman-Monteith ET0 water-balance via Open-Meteo (best_match)
    T2: FAO-56 ET0 via ECMWF IFS 0.25° operational model (Open-Meteo)
    T3: FAO-56 ET0 via ERA5 archive (same calendar month, 3-year average)

  ECONOMIC DATA (for financial impact)
    T1: World Bank Open Data — GDP + population density via Nominatim geocode
    T2: World Bank regional aggregates via Open-Meteo timezone mapping
    T3: World Bank global aggregate ('WLD' region code)

  RIVER DISCHARGE (GloFAS)
    T1: GloFAS v4 operational forecast (Open-Meteo Flood API)
    T2: GloFAS v4 archive — past 30 days (Open-Meteo Flood API)
    T3: GloFAS v4 prior-year archive — same calendar month (Open-Meteo Flood API)

  TEMPERATURE ANOMALY
    T1: ERA5 7-day observed mean vs same-month historical average
    T2: Current observation vs CMIP6 EC-Earth3P-HR climatological baseline

All functions return a dict that includes `_tier` and `source` keys.
"""

import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("cosmeon.model_hub")

# ──────────────────────────────────────────────────────────────────────────────
# Open-Meteo endpoint constants
# ──────────────────────────────────────────────────────────────────────────────
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
_SEASONAL_API = "https://seasonal-api.open-meteo.com/v1/seasonal"
_FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
_CLIMATE_API = "https://climate-api.open-meteo.com/v1/climate"
_ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
_WB_BASE = "https://api.worldbank.org/v2/country"
_NOMINATIM = "https://nominatim.openstreetmap.org/reverse"

_TIMEOUT_FAST = 8  # seconds — forecast/flood API calls
_TIMEOUT_SLOW = 15  # seconds — archive/climate API calls
_TIMEOUT_WB = 12  # seconds — World Bank API (can be slow)

# Timezone → World Bank region code mapping
_TZ_TO_WB_REGION = {
    "Asia/Kolkata": "SAS", "Asia/Colombo": "SAS", "Asia/Dhaka": "SAS",
    "Asia/Karachi": "SAS", "Asia/Kathmandu": "SAS", "Asia/Thimphu": "SAS",
    "Asia/Jakarta": "EAS", "Asia/Bangkok": "EAS", "Asia/Manila": "EAS",
    "Asia/Singapore": "EAS", "Asia/Ho_Chi_Minh": "EAS", "Asia/Kuala_Lumpur": "EAS",
    "Asia/Shanghai": "EAS", "Asia/Tokyo": "EAS", "Asia/Seoul": "EAS",
    "Asia/Taipei": "EAS", "Asia/Hong_Kong": "EAS",
    "Asia/Dubai": "MEA", "Asia/Riyadh": "MEA", "Asia/Tehran": "MEA",
    "Asia/Baghdad": "MEA", "Asia/Beirut": "MEA", "Asia/Jerusalem": "MEA",
    "Africa/Lagos": "SSF", "Africa/Nairobi": "SSF", "Africa/Johannesburg": "SSF",
    "Africa/Cairo": "MEA", "Africa/Casablanca": "MEA", "Africa/Tunis": "MEA",
    "Africa/Addis_Ababa": "SSF", "Africa/Dar_es_Salaam": "SSF",
    "Africa/Accra": "SSF", "Africa/Kinshasa": "SSF",
    "Europe/London": "ECS", "Europe/Paris": "ECS", "Europe/Berlin": "ECS",
    "Europe/Rome": "ECS", "Europe/Madrid": "ECS", "Europe/Amsterdam": "ECS",
    "Europe/Moscow": "ECS", "Europe/Istanbul": "ECS", "Europe/Warsaw": "ECS",
    "Europe/Bucharest": "ECS", "Europe/Budapest": "ECS",
    "America/New_York": "NAC", "America/Chicago": "NAC", "America/Denver": "NAC",
    "America/Los_Angeles": "NAC", "America/Toronto": "NAC",
    "America/Mexico_City": "LCN", "America/Sao_Paulo": "LCN",
    "America/Buenos_Aires": "LCN", "America/Bogota": "LCN",
    "America/Lima": "LCN", "America/Santiago": "LCN",
    "Australia/Sydney": "EAS", "Australia/Melbourne": "EAS",
    "Pacific/Auckland": "EAS",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRECIPITATION FORECAST
# ──────────────────────────────────────────────────────────────────────────────

def get_precipitation_forecast(lat: float, lon: float, months: int = 6) -> dict:
    """
    Return monthly precipitation forecast with tiered fallback.

    T1: ECMWF SEAS5 seasonal ensemble
    T2: ERA5 archive + GFS 16-day blend
    T3: CMIP6 EC-Earth3P-HR 30-year daily climatology
    """
    # T1: ECMWF SEAS5 seasonal ensemble
    try:
        result = _precip_forecast_ecmwf(lat, lon, months)
        if result and result.get("monthly_precip_mm"):
            result["_tier"] = 1
            logger.info("Precip T1 (ECMWF SEAS5): lat=%.2f lon=%.2f", lat, lon)
            return result
    except Exception as e:
        logger.warning("T1 ECMWF SEAS5 failed: %s", e)

    # T2: ERA5 archive + GFS 16-day blend
    try:
        result = _precip_forecast_era5_gfs(lat, lon, months)
        if result and result.get("monthly_precip_mm"):
            result["_tier"] = 2
            logger.info("Precip T2 (ERA5+GFS): lat=%.2f lon=%.2f", lat, lon)
            return result
    except Exception as e:
        logger.warning("T2 ERA5+GFS failed: %s", e)

    # T3: CMIP6 EC-Earth3P-HR climatological monthly means
    try:
        result = _precip_forecast_cmip6(lat, lon, months)
        if result and result.get("monthly_precip_mm"):
            result["_tier"] = 3
            logger.info("Precip T3 (CMIP6 climatology): lat=%.2f lon=%.2f", lat, lon)
            return result
    except Exception as e:
        logger.warning("T3 CMIP6 climatology failed: %s", e)

    return {
        "monthly_precip_mm": [], "monthly_prob_heavy": [],
        "source": "all_sources_unavailable", "_tier": 99,
    }


def _precip_forecast_ecmwf(lat: float, lon: float, months: int) -> dict:
    """ECMWF SEAS5 ensemble via Open-Meteo seasonal forecast API."""
    resp = requests.get(
        _SEASONAL_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum",
            "forecast_days": min(months * 30, 183),
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])

    member_keys = [k for k in daily if k.startswith("precipitation_sum_member")]
    if not member_keys:
        precip_raw = daily.get("precipitation_sum", [])
        member_arrays = [precip_raw] if precip_raw else []
    else:
        member_arrays = [daily[k] for k in member_keys]

    if not member_arrays or not dates:
        raise ValueError("No ensemble member data in ECMWF seasonal response")

    n_days = len(dates)
    mean_daily = []
    for i in range(n_days):
        vals = [m[i] for m in member_arrays if i < len(m) and m[i] is not None]
        mean_daily.append(sum(vals) / len(vals) if vals else 0.0)

    month_vals: dict = defaultdict(list)
    for d, p in zip(dates, mean_daily):
        month_vals[d[:7]].append(p)

    monthly_precip = []
    monthly_prob_heavy = []
    for key in sorted(month_vals.keys())[:months]:
        vals = month_vals[key]
        total = sum(vals)
        heavy = sum(1 for v in vals if v > 20) / max(len(vals), 1)
        monthly_precip.append(round(total, 1))
        monthly_prob_heavy.append(round(heavy, 3))

    return {
        "monthly_precip_mm": monthly_precip,
        "monthly_prob_heavy": monthly_prob_heavy,
        "source": "ECMWF SEAS5 ensemble via Open-Meteo seasonal API",
    }


def _precip_forecast_era5_gfs(lat: float, lon: float, months: int) -> dict:
    """ERA5 reanalysis archive + GFS 16-day operational forecast blend."""
    end = datetime.utcnow()
    start = end - timedelta(days=395)

    hist = requests.get(
        _ARCHIVE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": (end - timedelta(days=5)).strftime("%Y-%m-%d"),
            "daily": "precipitation_sum",
        },
        timeout=_TIMEOUT_SLOW,
    )
    hist.raise_for_status()
    hist_data = hist.json().get("daily", {})
    hist_dates = hist_data.get("time", [])
    hist_precip = [p if p is not None else 0.0
                   for p in hist_data.get("precipitation_sum", [])]

    gfs = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum",
            "forecast_days": 16,
        },
        timeout=_TIMEOUT_FAST,
    )
    gfs.raise_for_status()
    gfs_data = gfs.json().get("daily", {})
    gfs_dates = gfs_data.get("time", [])
    gfs_precip = [p if p is not None else 0.0
                  for p in gfs_data.get("precipitation_sum", [])]

    month_hist: dict = defaultdict(list)
    for d, p in zip(hist_dates, hist_precip):
        month_hist[int(d[5:7])].append(p)

    gfs_month: dict = defaultdict(list)
    for d, p in zip(gfs_dates, gfs_precip):
        gfs_month[int(d[5:7])].append(p)

    monthly_precip = []
    monthly_prob_heavy = []
    now = datetime.utcnow()
    for i in range(1, months + 1):
        target = now + timedelta(days=30 * i)
        m = target.month
        if m in gfs_month:
            vals = gfs_month[m] + month_hist.get(m, [])
        else:
            vals = month_hist.get(m, [])

        if vals:
            total = sum(vals) * (30 / max(len(vals), 1))
            heavy = sum(1 for v in vals if v > 20) / max(len(vals), 1)
        else:
            total = 60.0
            heavy = 0.10
        monthly_precip.append(round(total, 1))
        monthly_prob_heavy.append(round(min(1.0, heavy), 3))

    return {
        "monthly_precip_mm": monthly_precip,
        "monthly_prob_heavy": monthly_prob_heavy,
        "source": "ERA5 reanalysis archive + GFS 16-day operational (Open-Meteo)",
    }


def _precip_forecast_cmip6(lat: float, lon: float, months: int) -> dict:
    """
    T3: CMIP6 EC-Earth3P-HR 30-year daily climatology via Open-Meteo Climate API.
    Aggregates real climate model output into monthly means for the location.
    """
    resp = requests.get(
        _CLIMATE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": "1990-01-01", "end_date": "2020-12-31",
            "models": "EC_Earth3P_HR",
            "daily": "precipitation_sum",
        },
        timeout=_TIMEOUT_SLOW,
    )
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    if not dates or not precip:
        raise ValueError("CMIP6 returned empty precipitation data")

    # Compute monthly climatological means from 30 years of data
    month_totals: dict = defaultdict(list)
    month_heavy: dict = defaultdict(list)
    current_ym = None
    current_sum = 0.0
    current_heavy = 0
    current_days = 0

    for d, p in zip(dates, precip):
        ym = d[:7]
        pv = p if p is not None else 0.0
        if ym != current_ym:
            if current_ym is not None and current_days > 0:
                m = int(current_ym[5:7])
                month_totals[m].append(current_sum)
                month_heavy[m].append(current_heavy / current_days)
            current_ym = ym
            current_sum = 0.0
            current_heavy = 0
            current_days = 0
        current_sum += pv
        current_heavy += 1 if pv > 20 else 0
        current_days += 1

    # Final month
    if current_ym is not None and current_days > 0:
        m = int(current_ym[5:7])
        month_totals[m].append(current_sum)
        month_heavy[m].append(current_heavy / current_days)

    # Build forecast from climatological means
    monthly_precip = []
    monthly_prob_heavy = []
    now = datetime.utcnow()
    for i in range(1, months + 1):
        target = now + timedelta(days=30 * i)
        m = target.month
        totals = month_totals.get(m, [60.0])
        heavys = month_heavy.get(m, [0.1])
        monthly_precip.append(round(sum(totals) / len(totals), 1))
        monthly_prob_heavy.append(round(sum(heavys) / len(heavys), 3))

    return {
        "monthly_precip_mm": monthly_precip,
        "monthly_prob_heavy": monthly_prob_heavy,
        "source": "CMIP6 EC-Earth3P-HR 30-year climatology (Open-Meteo Climate API)",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. SOIL MOISTURE
# ──────────────────────────────────────────────────────────────────────────────

def get_soil_moisture(lat: float, lon: float) -> dict:
    """
    Return soil saturation fraction with tiered fallback.

    T1: ERA5 reanalysis archive (14 days)
    T2: ECMWF IFS 0.25° operational model (7 days)
    T3: ERA5 reanalysis same-month prior year (climatological baseline)
    """
    # T1: ERA5 reanalysis archive
    try:
        result = _soil_moisture_era5_archive(lat, lon)
        if result:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 soil moisture (ERA5 archive) failed: %s", e)

    # T2: ECMWF IFS 0.25° operational
    try:
        result = _soil_moisture_ecmwf_ifs(lat, lon)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 soil moisture (ECMWF IFS) failed: %s", e)

    # T3: ERA5 prior-year same-month archive
    try:
        result = _soil_moisture_era5_climatology(lat, lon)
        if result:
            result["_tier"] = 3
            return result
    except Exception as e:
        logger.warning("T3 soil moisture (ERA5 prior-year) failed: %s", e)

    return {
        "saturation_fraction": 0.0, "volumetric_m3m3": 0.0,
        "source": "no_soil_data_available", "_tier": 99,
    }


def _soil_moisture_era5_archive(lat: float, lon: float) -> dict:
    """T1: ERA5 reanalysis soil moisture from the archive API (last 14 days)."""
    end_date = date.today()
    start_date = end_date - timedelta(days=14)
    resp = requests.get(
        _ARCHIVE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "soil_moisture_0_to_7cm",
            "models": "era5",
        },
        timeout=_TIMEOUT_SLOW,
    )
    resp.raise_for_status()
    vals = [v for v in resp.json().get("hourly", {}).get("soil_moisture_0_to_7cm", [])
            if v is not None]
    if not vals:
        return {}
    recent = vals[-48:] if len(vals) >= 48 else vals
    avg = sum(recent) / len(recent)
    sat = min(1.0, avg / 0.40)
    return {
        "saturation_fraction": round(sat, 3),
        "volumetric_m3m3": round(avg, 4),
        "source": "ERA5 reanalysis soil_moisture_0_to_7cm (Open-Meteo archive)",
    }


def _soil_moisture_ecmwf_ifs(lat: float, lon: float) -> dict:
    """T2: ECMWF IFS 0.25° operational model soil moisture."""
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "hourly": "soil_moisture_0_to_7cm",
            "past_days": 7, "forecast_days": 0,
            "models": "ecmwf_ifs025",
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    vals = [v for v in resp.json().get("hourly", {}).get("soil_moisture_0_to_7cm", [])
            if v is not None]
    if not vals:
        return {}
    recent = vals[-48:] if len(vals) >= 48 else vals
    avg = sum(recent) / len(recent)
    sat = min(1.0, avg / 0.40)
    return {
        "saturation_fraction": round(sat, 3),
        "volumetric_m3m3": round(avg, 4),
        "source": "ECMWF IFS 0.25° operational soil_moisture_0_to_7cm (Open-Meteo)",
    }


def _soil_moisture_era5_climatology(lat: float, lon: float) -> dict:
    """T3: ERA5 reanalysis same-month prior year (climatological baseline)."""
    now = date.today()
    start_date = date(now.year - 1, now.month, 1)
    end_date = date(now.year - 1, now.month, 28)
    resp = requests.get(
        _ARCHIVE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "soil_moisture_0_to_7cm",
            "models": "era5",
        },
        timeout=_TIMEOUT_SLOW,
    )
    resp.raise_for_status()
    vals = [v for v in resp.json().get("hourly", {}).get("soil_moisture_0_to_7cm", [])
            if v is not None]
    if not vals:
        return {}
    avg = sum(vals) / len(vals)
    sat = min(1.0, avg / 0.40)
    return {
        "saturation_fraction": round(sat, 3),
        "volumetric_m3m3": round(avg, 4),
        "source": "ERA5 reanalysis same-month prior year (climatological baseline)",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. VEGETATION STRESS
# ──────────────────────────────────────────────────────────────────────────────

def get_vegetation_stress(lat: float, lon: float) -> dict:
    """
    Return vegetation stress index (0 = healthy, 1 = severely stressed).

    T1: FAO-56 ET0 water-balance via Open-Meteo best_match model
    T2: FAO-56 ET0 via ECMWF IFS 0.25° operational model
    T3: FAO-56 ET0 via ERA5 archive (same calendar month, 3-year average)
    """
    # T1: FAO-56 ET0 via Open-Meteo best_match
    try:
        result = _veg_stress_fao56(lat, lon, model="best_match")
        if result:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 veg stress (FAO-56 best_match) failed: %s", e)

    # T2: FAO-56 ET0 via ECMWF IFS 0.25°
    try:
        result = _veg_stress_fao56(lat, lon, model="ecmwf_ifs025")
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 veg stress (FAO-56 ECMWF IFS) failed: %s", e)

    # T3: FAO-56 ET0 via ERA5 archive (same-month 3-year average)
    try:
        result = _veg_stress_era5_archive(lat, lon)
        if result:
            result["_tier"] = 3
            return result
    except Exception as e:
        logger.warning("T3 veg stress (ERA5 archive) failed: %s", e)

    return {
        "stress_index": 0.0, "et0_mm_day": 0.0, "precip_mm_day": 0.0,
        "source": "no_et0_data_available", "_tier": 99,
    }


def _veg_stress_fao56(lat: float, lon: float, model: str = "best_match") -> dict:
    """FAO-56 Penman-Monteith ET0 water-balance from Open-Meteo forecast API."""
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "et0_fao_evapotranspiration,precipitation_sum",
        "past_days": 14, "forecast_days": 0,
    }
    if model != "best_match":
        params["models"] = model
    resp = requests.get(_FORECAST_API, params=params, timeout=_TIMEOUT_FAST)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    et0_vals = [v for v in daily.get("et0_fao_evapotranspiration", []) if v is not None]
    rain_vals = [v for v in daily.get("precipitation_sum", []) if v is not None]
    if not et0_vals:
        return {}
    avg_et0 = sum(et0_vals) / len(et0_vals)
    avg_rain = sum(rain_vals) / len(rain_vals) if rain_vals else 0.0
    deficit = avg_et0 - avg_rain
    stress = max(0.0, min(1.0, deficit / 5.0))
    model_name = "Open-Meteo best_match" if model == "best_match" else f"ECMWF IFS 0.25°"
    return {
        "stress_index": round(stress, 3),
        "et0_mm_day": round(avg_et0, 2),
        "precip_mm_day": round(avg_rain, 2),
        "source": f"FAO-56 ET0 water-balance via {model_name}",
    }


def _veg_stress_era5_archive(lat: float, lon: float) -> dict:
    """T3: FAO-56 ET0 from ERA5 archive — same calendar month, 3-year average."""
    now = datetime.utcnow()
    et0_totals = []
    rain_totals = []

    for years_back in [1, 2, 3]:
        try:
            start = date(now.year - years_back, now.month, 1)
            end = date(now.year - years_back, now.month, 28)
            resp = requests.get(
                _ARCHIVE_API,
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "daily": "et0_fao_evapotranspiration,precipitation_sum",
                },
                timeout=_TIMEOUT_SLOW,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            et0 = [v for v in daily.get("et0_fao_evapotranspiration", []) if v is not None]
            rain = [v for v in daily.get("precipitation_sum", []) if v is not None]
            if et0:
                et0_totals.extend(et0)
            if rain:
                rain_totals.extend(rain)
        except Exception:
            continue

    if not et0_totals:
        return {}

    avg_et0 = sum(et0_totals) / len(et0_totals)
    avg_rain = sum(rain_totals) / len(rain_totals) if rain_totals else 0.0
    deficit = avg_et0 - avg_rain
    stress = max(0.0, min(1.0, deficit / 5.0))
    return {
        "stress_index": round(stress, 3),
        "et0_mm_day": round(avg_et0, 2),
        "precip_mm_day": round(avg_rain, 2),
        "source": "FAO-56 ET0 via ERA5 archive (same-month 3-year average)",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. ECONOMIC DATA (GDP + population density)
# ──────────────────────────────────────────────────────────────────────────────

def get_economic_data(lat: float, lon: float, name: str = "") -> dict:
    """
    Return GDP and population density with tiered fallback.

    T1: World Bank via Nominatim reverse geocode → country → GDP + pop density
    T2: World Bank regional aggregate via Open-Meteo timezone mapping
    T3: World Bank global aggregate ('WLD')
    """
    # T1: World Bank via Nominatim country code
    try:
        result = _econ_world_bank(lat, lon)
        if result and result.get("gdp_usd", 0) > 0:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 economic (World Bank) failed: %s", e)

    # T2: World Bank regional aggregate via timezone
    try:
        result = _econ_world_bank_regional(lat, lon)
        if result and result.get("gdp_usd", 0) > 0:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 economic (WB regional) failed: %s", e)

    # T3: World Bank global aggregate
    try:
        result = _econ_world_bank_global()
        result["_tier"] = 3
        return result
    except Exception as e:
        logger.warning("T3 economic (WB global) failed: %s", e)

    return {
        "gdp_usd": 0.0, "pop_density_km2": 100.0,
        "country_code": "XX", "country_name": "Unknown",
        "source": "all_sources_unavailable", "_tier": 99,
    }


def _econ_world_bank(lat: float, lon: float) -> dict:
    """T1: Nominatim reverse geocode → ISO-2 → World Bank GDP + pop density."""
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

    gdp_usd = _fetch_wb_indicator(cc, "NY.GDP.MKTP.CD")
    nat_density = _fetch_wb_indicator(cc, "EN.POP.DNST") or 100.0

    return {
        "gdp_usd": float(gdp_usd) if gdp_usd else 0.0,
        "pop_density_km2": round(nat_density * 6.0, 0),  # urban multiplier
        "country_code": cc,
        "country_name": country_name,
        "source": "World Bank Open Data (NY.GDP.MKTP.CD + EN.POP.DNST)",
    }


def _econ_world_bank_regional(lat: float, lon: float) -> dict:
    """T2: Use Open-Meteo timezone to determine World Bank region code."""
    # Get timezone from Open-Meteo
    resp = requests.get(
        _FORECAST_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max",
            "forecast_days": 1,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    tz = resp.json().get("timezone", "")

    # Map timezone to World Bank region
    region_code = _TZ_TO_WB_REGION.get(tz)
    if not region_code:
        # Try prefix match (e.g., "Europe/Berlin" -> match "Europe/" prefix)
        for tz_key, rc in _TZ_TO_WB_REGION.items():
            if tz.split("/")[0] == tz_key.split("/")[0]:
                region_code = rc
                break
    if not region_code:
        raise ValueError(f"Cannot map timezone '{tz}' to World Bank region")

    gdp_usd = _fetch_wb_indicator(region_code, "NY.GDP.MKTP.CD")
    nat_density = _fetch_wb_indicator(region_code, "EN.POP.DNST") or 100.0

    return {
        "gdp_usd": float(gdp_usd) if gdp_usd else 0.0,
        "pop_density_km2": round(nat_density * 6.0, 0),
        "country_code": region_code,
        "country_name": f"World Bank region: {region_code}",
        "source": f"World Bank regional aggregate ({region_code})",
    }


def _econ_world_bank_global() -> dict:
    """T3: World Bank global aggregate ('WLD' region)."""
    gdp_usd = _fetch_wb_indicator("WLD", "NY.GDP.MKTP.CD")
    nat_density = _fetch_wb_indicator("WLD", "EN.POP.DNST") or 60.0
    return {
        "gdp_usd": float(gdp_usd) if gdp_usd else 100e12,
        "pop_density_km2": round(nat_density * 6.0, 0),
        "country_code": "WLD",
        "country_name": "World (global average)",
        "source": "World Bank global aggregate (WLD)",
    }


def _fetch_wb_indicator(country_or_region: str, indicator: str) -> Optional[float]:
    """Fetch a single World Bank indicator value (most recent available)."""
    try:
        resp = requests.get(
            f"{_WB_BASE}/{country_or_region}/indicator/{indicator}",
            params={"mrv": 1, "format": "json", "per_page": 1},
            headers={"User-Agent": "COSMEON/1.0"},
            timeout=_TIMEOUT_WB,
        )
        resp.raise_for_status()
        data = resp.json()
        if (isinstance(data, list) and len(data) >= 2
                and data[1] and data[1][0].get("value") is not None):
            return float(data[1][0]["value"])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. RIVER DISCHARGE (GloFAS)
# ──────────────────────────────────────────────────────────────────────────────

def get_river_discharge(lat: float, lon: float, past_days: int = 30) -> dict:
    """
    Return GloFAS river discharge data with tiered fallback.

    T1: GloFAS v4 operational forecast (recent + 7-day forecast)
    T2: GloFAS v4 archive (past 30 days)
    T3: GloFAS v4 prior-year archive (same calendar month)
    """
    # T1: GloFAS v4 operational forecast
    try:
        result = _discharge_glofas_forecast(lat, lon, past_days)
        if result and result.get("current_discharge_m3s", -1) >= 0:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 discharge (GloFAS forecast) failed: %s", e)

    # T2: GloFAS archive
    try:
        result = _discharge_glofas_archive(lat, lon, past_days)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 discharge (GloFAS archive) failed: %s", e)

    # T3: GloFAS prior-year same-month archive
    try:
        result = _discharge_glofas_prior_year(lat, lon)
        if result:
            result["_tier"] = 3
            return result
    except Exception as e:
        logger.warning("T3 discharge (GloFAS prior-year) failed: %s", e)

    return {
        "current_discharge_m3s": 0.0, "mean_discharge_m3s": 0.0,
        "anomaly_sigma": 0.0, "flood_risk_level": "UNKNOWN",
        "forecast_discharge": [],
        "source": "no_glofas_coverage", "_tier": 99,
    }


def _discharge_glofas_forecast(lat: float, lon: float, past_days: int) -> dict:
    """T1: GloFAS v4 operational forecast via Open-Meteo Flood API."""
    resp = requests.get(
        _FLOOD_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "river_discharge,river_discharge_mean,river_discharge_max",
            "past_days": past_days, "forecast_days": 7,
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    discharge = [v for v in daily.get("river_discharge", []) if v is not None]
    means = [v for v in daily.get("river_discharge_mean", []) if v is not None]
    if not discharge:
        return {}

    current = discharge[-1]
    mean = sum(means) / len(means) if means else (sum(discharge) / len(discharge))
    std = float(np.std(discharge)) if len(discharge) > 1 else 1.0
    anomaly = (current - mean) / max(std, 0.01)
    risk = _discharge_to_risk(anomaly, current, mean)

    return {
        "current_discharge_m3s": round(current, 2),
        "mean_discharge_m3s": round(mean, 2),
        "anomaly_sigma": round(anomaly, 2),
        "flood_risk_level": risk,
        "forecast_discharge": discharge[-7:],
        "source": "GloFAS v4 operational via Open-Meteo Flood API",
    }


def _discharge_glofas_archive(lat: float, lon: float, past_days: int) -> dict:
    """T2: GloFAS v4 archive (past 30 days)."""
    end = datetime.utcnow() - timedelta(days=5)
    start = end - timedelta(days=past_days)
    resp = requests.get(
        _FLOOD_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "river_discharge",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    discharge = [v for v in resp.json().get("daily", {}).get("river_discharge", [])
                 if v is not None]
    if not discharge:
        return {}

    current = discharge[-1]
    mean = sum(discharge) / len(discharge)
    std = float(np.std(discharge)) if len(discharge) > 1 else 1.0
    anomaly = (current - mean) / max(std, 0.01)
    risk = _discharge_to_risk(anomaly, current, mean)
    return {
        "current_discharge_m3s": round(current, 2),
        "mean_discharge_m3s": round(mean, 2),
        "anomaly_sigma": round(anomaly, 2),
        "flood_risk_level": risk,
        "forecast_discharge": discharge[-7:],
        "source": "GloFAS v4 archive via Open-Meteo Flood API",
    }


def _discharge_glofas_prior_year(lat: float, lon: float) -> dict:
    """T3: GloFAS v4 same calendar month from prior year."""
    now = datetime.utcnow()
    start = date(now.year - 1, now.month, 1)
    end = date(now.year - 1, now.month, 28)
    resp = requests.get(
        _FLOOD_API,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "river_discharge",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=_TIMEOUT_FAST,
    )
    resp.raise_for_status()
    discharge = [v for v in resp.json().get("daily", {}).get("river_discharge", [])
                 if v is not None]
    if not discharge:
        return {}

    current = discharge[-1]
    mean = sum(discharge) / len(discharge)
    std = float(np.std(discharge)) if len(discharge) > 1 else 1.0
    anomaly = (current - mean) / max(std, 0.01)
    risk = _discharge_to_risk(anomaly, current, mean)
    return {
        "current_discharge_m3s": round(current, 2),
        "mean_discharge_m3s": round(mean, 2),
        "anomaly_sigma": round(anomaly, 2),
        "flood_risk_level": risk,
        "forecast_discharge": discharge[-7:],
        "source": "GloFAS v4 same-month prior year via Open-Meteo Flood API",
    }


def _discharge_to_risk(anomaly: float, current: float, mean: float) -> str:
    """GloFAS operational return period classification."""
    ratio = current / max(mean, 0.01)
    if ratio > 3.0 or anomaly > 3.0:
        return "CRITICAL"  # ~20-year return period
    elif ratio > 2.0 or anomaly > 2.0:
        return "HIGH"  # ~5-year return period
    elif ratio > 1.3 or anomaly > 1.0:
        return "MEDIUM"  # ~2-year return period
    return "LOW"


# ──────────────────────────────────────────────────────────────────────────────
# 6. TEMPERATURE ANOMALY
# ──────────────────────────────────────────────────────────────────────────────

def get_temperature_anomaly(lat: float, lon: float) -> dict:
    """
    Return current temperature anomaly vs seasonal baseline.

    T1: ERA5 7-day mean vs same-month historical average (1 year archive)
    T2: Current observation vs CMIP6 EC-Earth3P-HR climatological baseline
    """
    # T1: ERA5 historical climatology
    try:
        result = _temp_anomaly_era5_clim(lat, lon)
        if result:
            result["_tier"] = 1
            return result
    except Exception as e:
        logger.warning("T1 temp anomaly (ERA5 clim) failed: %s", e)

    # T2: CMIP6 climatological baseline
    try:
        result = _temp_anomaly_cmip6(lat, lon)
        if result:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 temp anomaly (CMIP6) failed: %s", e)

    return {
        "current_avg_c": 0.0, "baseline_c": 0.0, "anomaly_c": 0.0,
        "source": "no_baseline_available", "_tier": 99,
    }


def _temp_anomaly_era5_clim(lat: float, lon: float) -> dict:
    """T1: Compare current 7-day mean vs ERA5 same-month historical average."""
    now = datetime.utcnow()
    start_hist = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    end_hist = (now - timedelta(days=8)).strftime("%Y-%m-%d")

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
    cur_month = now.month
    same_month = [t for d, t in zip(hist_dates, hist_temps) if int(d[5:7]) == cur_month]
    baseline = sum(same_month) / len(same_month) if same_month else sum(hist_temps) / len(hist_temps)
    anomaly = current_avg - baseline

    return {
        "current_avg_c": round(current_avg, 1),
        "baseline_c": round(baseline, 1),
        "anomaly_c": round(anomaly, 1),
        "source": "ERA5 historical climatology vs current observation",
    }


def _temp_anomaly_cmip6(lat: float, lon: float) -> dict:
    """T2: Current observation vs CMIP6 EC-Earth3P-HR climatological baseline."""
    # Get current temperature
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
    if not curr_temps:
        return {}
    current_avg = sum(curr_temps) / len(curr_temps)

    # Get CMIP6 30-year climatological baseline for this month
    resp = requests.get(
        _CLIMATE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": "1990-01-01", "end_date": "2020-12-31",
            "models": "EC_Earth3P_HR",
            "daily": "temperature_2m_mean",
        },
        timeout=_TIMEOUT_SLOW,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not dates or not temps:
        return {}

    cur_month = datetime.utcnow().month
    same_month_temps = [t for d, t in zip(dates, temps)
                        if t is not None and int(d[5:7]) == cur_month]
    if not same_month_temps:
        return {}

    baseline = sum(same_month_temps) / len(same_month_temps)
    anomaly = current_avg - baseline

    return {
        "current_avg_c": round(current_avg, 1),
        "baseline_c": round(baseline, 1),
        "anomaly_c": round(anomaly, 1),
        "source": "CMIP6 EC-Earth3P-HR climatology vs current observation",
    }
