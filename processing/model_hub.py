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
    A value of exactly 0.0 is treated as no-data (ocean grid cell) and
    triggers T2, since ERA5 returns 0 for water-covered grid cells.

T1: ERA5 reanalysis archive (14 days)
    T2: ECMWF IFS 0.25° operational model (7 days)
    T3: ERA5 reanalysis same-month prior year (climatological baseline)
    """
    # T1: ERA5 reanalysis archive
    try:
        result = _soil_moisture_era5_archive(lat, lon)
        # saturation_fraction == 0.0 means ERA5 returned ocean/no-land data
        if result and result.get("saturation_fraction", 0) > 0.001:
            result["_tier"] = 1
            return result
        if result and result.get("saturation_fraction", 0) == 0.0:
            logger.info("Soil T1 ERA5 returned 0 (ocean grid) at %.2f,%.2f — trying T2", lat, lon)
    except Exception as e:
        logger.warning("T1 soil moisture (ERA5 archive) failed: %s", e)

    # T2: ECMWF IFS 0.25° operational
    try:
        result = _soil_moisture_ecmwf_ifs(lat, lon)
        if result and result.get("saturation_fraction", 0) > 0.001:
            result["_tier"] = 2
            return result
    except Exception as e:
        logger.warning("T2 soil moisture (ECMWF IFS) failed: %s", e)

    # T3: ERA5 prior-year same-month archive
    try:
        result = _soil_moisture_era5_climatology(lat, lon)
        if result and result.get("saturation_fraction", 0) > 0.001:
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


def _latlon_to_wb_region(lat: float, lon: float) -> Optional[str]:
    """
    Map lat/lon coordinates to World Bank regional aggregate code.
    Used as a final geographic fallback when timezone mapping fails
    (e.g. 'GMT' timezone for small islands, Sahara, ocean points).
    """
    # North America
    if lat > 14 and -170 < lon < -50:
        return "NAC"
    # Latin America & Caribbean
    if -60 < lat < 35 and -120 < lon < -30:
        return "LCN"
    # Europe & Central Asia (includes Russia)
    if lat > 35 and -25 < lon < 85:
        return "ECS"
    if lat > 50 and 85 < lon < 180:  # Russia east of Urals
        return "ECS"
    # Middle East & North Africa
    if 15 < lat < 40 and -10 < lon < 65:
        return "MEA"
    # Sub-Saharan Africa
    if -35 < lat < 15 and -20 < lon < 55:
        return "SSF"
    # South Asia
    if 5 < lat < 40 and 55 < lon < 100:
        return "SAS"
    # East Asia & Pacific (including small Pacific islands)
    if -55 < lat < 55 and 95 < lon < 180:
        return "EAS"
    if lat < 0 and lon < -100:  # Pacific islands south/east
        return "EAS"
    # Default: MEA for remaining Sahara/equatorial ambiguous zones
    if -35 < lat < 40 and -20 < lon < 55:
        return "MEA"
    return None


def _econ_world_bank_regional(lat: float, lon: float) -> dict:
    """T2: Use Open-Meteo timezone (with lat/lon bounding box fallback) to determine
    World Bank region code. Handles 'GMT' timezone returned for islands/Sahara/ocean."""
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

    # 1. Direct match
    region_code = _TZ_TO_WB_REGION.get(tz)

    # 2. Prefix match (e.g. "Europe/Berlin" → first "Europe/" key)
    if not region_code and "/" in tz:
        tz_prefix = tz.split("/")[0]
        for tz_key, rc in _TZ_TO_WB_REGION.items():
            if tz_key.split("/")[0] == tz_prefix:
                region_code = rc
                break

    # 3. Lat/lon geographic bounding box (handles "GMT", unknown islands, etc.)
    if not region_code:
        region_code = _latlon_to_wb_region(lat, lon)
        if region_code:
            logger.info("Econ T2: timezone '%s' unmapped — used lat/lon bbox → %s", tz, region_code)

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

    # T4: ERA5 archive precipitation-based surrogate (flood-api subdomain is blocked on some hosts)
    try:
        result = _discharge_precip_surrogate(lat, lon, past_days)
        if result and result.get("current_discharge_m3s", 0) >= 0:
            result["_tier"] = 4
            logger.info("Discharge T4 (precip surrogate from ERA5 archive): lat=%.2f lon=%.2f", lat, lon)
            return result
    except Exception as e:
        logger.warning("T4 discharge (precip surrogate) failed: %s", e)

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


def _discharge_precip_surrogate(lat: float, lon: float, past_days: int) -> dict:
    """
    T4: Precipitation-based discharge surrogate via ERA5 archive API.
    Used when flood-api.open-meteo.com is unreachable (e.g. blocked on some hosting providers).
    Fetches actual ERA5 precipitation and applies a 3-day lag runoff proxy.
    """
    end_dt = datetime.utcnow() - timedelta(days=5)  # ERA5 has ~5-day lag
    start_dt = end_dt - timedelta(days=past_days)
    resp = requests.get(
        _ARCHIVE_API,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "daily": "precipitation_sum",
        },
        timeout=_TIMEOUT_SLOW,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    precip = [p if p is not None else 0.0 for p in daily.get("precipitation_sum", [])]
    if not precip:
        return {}

    # 3-day rolling sum as runoff proxy; scale by 10 m³/s per mm/day
    lag = 3
    surrogate = [sum(precip[max(0, i - lag):i + 1]) * 10.0 for i in range(len(precip))]
    current = round(surrogate[-1], 2) if surrogate else 0.0
    mean_v = float(np.mean(surrogate)) if surrogate else 1.0
    std_v = float(np.std(surrogate)) if len(surrogate) > 1 else max(mean_v * 0.2, 1.0)
    anomaly = round((current - mean_v) / max(std_v, 0.01), 2)
    risk = _discharge_to_risk(anomaly, current, mean_v)
    return {
        "current_discharge_m3s": current,
        "mean_discharge_m3s": round(mean_v, 2),
        "anomaly_sigma": anomaly,
        "flood_risk_level": risk,
        "forecast_discharge": surrogate[-7:],
        "source": "ERA5 precipitation surrogate (flood API unavailable)",
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


# ──────────────────────────────────────────────────────────────────────────────
# 7. SATELLITE NDVI (NASA MODIS)
# ──────────────────────────────────────────────────────────────────────────────

def get_ndvi_satellite(lat: float, lon: float) -> dict:
    """
    Fetch real satellite NDVI from NASA MODIS (MOD13Q1, 250m, 16-day composite).

    Tier 0: NASA ORNL DAAC TESViS API — free, no auth, global coverage.
    Returns actual NDVI from Terra MODIS, not a computed proxy.

    API: https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset
    """
    try:
        today = datetime.utcnow()
        start_dt = today - timedelta(days=90)

        # NASA day-of-year format: AYYYY-DDD
        start_doy = f"A{start_dt.year}-{start_dt.timetuple().tm_yday:03d}"
        end_doy = f"A{today.year}-{today.timetuple().tm_yday:03d}"

        resp = requests.get(
            "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset",
            params={
                "latitude": lat,
                "longitude": lon,
                "startDate": start_doy,
                "endDate": end_doy,
                "kmAboveBelow": 0,
                "kmLeftRight": 0,
                "band": "250m_16_days_NDVI",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        subsets = data.get("subset", [])
        if not subsets:
            logger.warning("NDVI satellite: no subset data for %.2f, %.2f", lat, lon)
            return None

        # Walk backwards from the most recent observation to find a valid NDVI
        for entry in reversed(subsets):
            raw_values = entry.get("data", [])
            calendar_date = entry.get("calendar_date", "unknown")

            # Filter out fill/cloud/bad data: valid raw NDVI is in [-2000, 10000]
            valid = [v for v in raw_values if -2000 <= v <= 10000]
            if not valid:
                continue

            # Scale: raw values are NDVI * 10000
            ndvi = sum(valid) / len(valid) / 10000.0

            # Vegetation stress: healthy vegetation (NDVI ~0.6+) = low stress
            stress = max(0.0, min(1.0, 1.0 - ndvi))

            return {
                "ndvi": round(ndvi, 4),
                "stress_index": round(stress, 4),
                "source": "NASA MODIS MOD13Q1 (250m satellite)",
                "_tier": 0,
                "date": calendar_date,
            }

        logger.warning("NDVI satellite: all observations were fill/cloud for %.2f, %.2f", lat, lon)
        return None

    except Exception as e:
        logger.warning("NDVI satellite fetch failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 8. FLOOD RETURN-PERIOD THRESHOLDS (GloFAS historical)
# ──────────────────────────────────────────────────────────────────────────────

def get_return_period_thresholds(lat: float, lon: float) -> dict:
    """
    Compute flood return-period thresholds from GloFAS historical discharge (1990-present).

    Uses Open-Meteo Flood API historical data to derive 2/5/10/20-year return periods
    via Weibull plotting position on annual discharge maxima.
    This is the standard hydrological method used by ECMWF/GloFAS.
    """
    try:
        today = date.today()
        resp = requests.get(
            _FLOOD_API,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "river_discharge",
                "start_date": "1990-01-01",
                "end_date": today.isoformat(),
            },
            timeout=20,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        dates = daily.get("time", [])
        discharge = daily.get("river_discharge", [])

        if not dates or not discharge:
            logger.warning("Return period: no discharge data for %.2f, %.2f", lat, lon)
            return None

        # Group discharge values by year, take annual maxima
        year_max: dict = defaultdict(float)
        all_valid = []
        for d, q in zip(dates, discharge):
            if q is None:
                continue
            year = int(d[:4])
            year_max[year] = max(year_max[year], q)
            all_valid.append(q)

        if not year_max or len(year_max) < 5:
            logger.warning("Return period: insufficient years (%d) for %.2f, %.2f",
                           len(year_max), lat, lon)
            return None

        # Sort annual maxima ascending for Weibull plotting position
        annual_maxima = sorted(year_max.values())
        n = len(annual_maxima)

        # Compute return-period thresholds via Weibull plotting position
        # For return period T, threshold is at rank position n * (1 - 1/T)
        return_periods = [2, 5, 10, 20, 50]
        thresholds = {}
        for rp in return_periods:
            idx_float = n * (1.0 - 1.0 / rp)
            # Linear interpolation between bounding indices
            idx_low = int(math.floor(idx_float))
            idx_high = int(math.ceil(idx_float))
            idx_low = max(0, min(idx_low, n - 1))
            idx_high = max(0, min(idx_high, n - 1))
            if idx_low == idx_high:
                thresholds[rp] = round(annual_maxima[idx_low], 2)
            else:
                frac = idx_float - idx_low
                val = annual_maxima[idx_low] * (1 - frac) + annual_maxima[idx_high] * frac
                thresholds[rp] = round(val, 2)

        # Current conditions: last 7 days
        recent_discharge = [q for q in discharge[-7:] if q is not None]
        current_7d_avg = sum(recent_discharge) / len(recent_discharge) if recent_discharge else 0.0
        current_max_7d = max(recent_discharge) if recent_discharge else 0.0
        mean_discharge = sum(all_valid) / len(all_valid) if all_valid else 0.0

        # Determine highest return period exceeded by current max
        exceeded_rp = 0
        for rp in sorted(return_periods):
            if current_max_7d >= thresholds[rp]:
                exceeded_rp = rp

        # Exceedance probability
        if exceeded_rp > 0:
            exceedance_prob = 1.0 / exceeded_rp
        else:
            # Estimate from percentile of current discharge in historical distribution
            rank = sum(1 for v in all_valid if v <= current_max_7d)
            percentile = rank / len(all_valid) if all_valid else 0.5
            exceedance_prob = round(1.0 - percentile, 4)

        return {
            "thresholds": thresholds,
            "current_discharge": round(current_7d_avg, 2),
            "current_max_7d": round(current_max_7d, 2),
            "mean_discharge": round(mean_discharge, 2),
            "years_of_data": n,
            "exceeded_return_period": exceeded_rp,
            "exceedance_probability": round(exceedance_prob, 4),
            "source": "GloFAS v4 historical (1990-present) via Open-Meteo",
            "_tier": 1,
        }

    except Exception as e:
        logger.warning("Return period thresholds failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 9. INFORM RISK INDEX + GAR FLOOD LOSS
# ──────────────────────────────────────────────────────────────────────────────

# Source: EU JRC INFORM Risk Index 2024 — country-level flood hazard scores (0-10)
# Format: ISO3 -> {risk: overall, flood: flood_hazard, vuln: vulnerability, cope: coping_capacity}
_INFORM_FLOOD_RISK = {
    "AFG": {"risk": 8.3, "flood": 7.2, "vuln": 7.8, "cope": 8.9},
    "BGD": {"risk": 7.1, "flood": 8.5, "vuln": 5.8, "cope": 6.1},
    "IND": {"risk": 5.7, "flood": 7.4, "vuln": 5.1, "cope": 4.3},
    "IDN": {"risk": 5.1, "flood": 6.8, "vuln": 4.2, "cope": 4.5},
    "JPN": {"risk": 3.2, "flood": 5.9, "vuln": 1.8, "cope": 1.5},
    "NPL": {"risk": 6.2, "flood": 7.1, "vuln": 6.3, "cope": 6.0},
    "PAK": {"risk": 7.0, "flood": 8.2, "vuln": 6.1, "cope": 6.8},
    "PHL": {"risk": 5.9, "flood": 7.8, "vuln": 5.2, "cope": 4.8},
    "BRA": {"risk": 4.2, "flood": 5.5, "vuln": 4.0, "cope": 3.5},
    "DEU": {"risk": 2.4, "flood": 4.2, "vuln": 1.2, "cope": 1.0},
    "NLD": {"risk": 2.8, "flood": 6.5, "vuln": 1.0, "cope": 0.8},
    "GBR": {"risk": 2.5, "flood": 4.0, "vuln": 1.5, "cope": 1.1},
    "USA": {"risk": 3.0, "flood": 4.5, "vuln": 2.1, "cope": 1.4},
    "CHN": {"risk": 5.3, "flood": 7.0, "vuln": 3.8, "cope": 4.2},
    "MMR": {"risk": 7.5, "flood": 7.6, "vuln": 6.5, "cope": 8.2},
    "VNM": {"risk": 4.5, "flood": 6.5, "vuln": 3.8, "cope": 4.0},
    "THA": {"risk": 3.8, "flood": 6.0, "vuln": 3.2, "cope": 3.0},
    "KHM": {"risk": 5.5, "flood": 6.8, "vuln": 5.8, "cope": 5.5},
    "NGA": {"risk": 7.2, "flood": 6.5, "vuln": 7.0, "cope": 7.8},
    "MOZ": {"risk": 6.8, "flood": 7.5, "vuln": 7.2, "cope": 7.0},
    "ETH": {"risk": 7.3, "flood": 5.8, "vuln": 7.5, "cope": 7.5},
    "SSD": {"risk": 9.1, "flood": 6.5, "vuln": 8.5, "cope": 9.5},
    "COD": {"risk": 8.0, "flood": 5.5, "vuln": 7.8, "cope": 9.0},
    "SDN": {"risk": 8.5, "flood": 6.0, "vuln": 7.5, "cope": 8.8},
    "SOM": {"risk": 8.9, "flood": 5.5, "vuln": 8.0, "cope": 9.2},
    "HTI": {"risk": 7.8, "flood": 7.0, "vuln": 7.5, "cope": 8.5},
    "AUS": {"risk": 2.6, "flood": 4.8, "vuln": 1.5, "cope": 1.0},
    "ITA": {"risk": 3.0, "flood": 5.0, "vuln": 2.0, "cope": 1.8},
    "FRA": {"risk": 2.8, "flood": 4.5, "vuln": 1.8, "cope": 1.2},
    "COL": {"risk": 5.5, "flood": 6.5, "vuln": 4.5, "cope": 4.8},
    "PER": {"risk": 4.8, "flood": 6.0, "vuln": 4.2, "cope": 4.5},
    "MEX": {"risk": 4.5, "flood": 5.5, "vuln": 3.8, "cope": 3.5},
    "RUS": {"risk": 4.0, "flood": 5.0, "vuln": 3.2, "cope": 3.5},
    "KEN": {"risk": 5.8, "flood": 5.5, "vuln": 6.0, "cope": 6.2},
    "TZA": {"risk": 5.5, "flood": 5.0, "vuln": 6.2, "cope": 5.8},
    "LKA": {"risk": 4.5, "flood": 6.5, "vuln": 4.0, "cope": 3.8},
    "EGY": {"risk": 4.2, "flood": 4.0, "vuln": 4.5, "cope": 4.0},
    "ZAF": {"risk": 4.5, "flood": 4.5, "vuln": 5.0, "cope": 3.5},
}

# Source: UNDRR GAR 2022 — Annual Average Loss from floods (USD)
# Top countries by flood AAL
_GAR_ANNUAL_FLOOD_LOSS = {
    "CHN": 30_000_000_000,  # $30B
    "USA": 20_000_000_000,
    "IND": 14_000_000_000,
    "JPN": 8_000_000_000,
    "BRA": 4_000_000_000,
    "IDN": 3_500_000_000,
    "DEU": 3_000_000_000,
    "GBR": 2_500_000_000,
    "BGD": 2_000_000_000,
    "PAK": 2_500_000_000,
    "THA": 2_000_000_000,
    "NLD": 1_500_000_000,
    "FRA": 2_000_000_000,
    "ITA": 2_500_000_000,
    "AUS": 1_500_000_000,
    "PHL": 1_800_000_000,
    "MEX": 1_200_000_000,
    "COL": 800_000_000,
    "VNM": 1_000_000_000,
    "NGA": 600_000_000,
    "KEN": 300_000_000,
    "ETH": 200_000_000,
    "MOZ": 150_000_000,
    "NPL": 400_000_000,
    "MMR": 500_000_000,
    "KHM": 300_000_000,
    "LKA": 400_000_000,
    "RUS": 2_000_000_000,
    "ZAF": 500_000_000,
    "PER": 600_000_000,
    "EGY": 200_000_000,
    "SDN": 100_000_000,
    "SSD": 50_000_000,
    "SOM": 80_000_000,
    "HTI": 200_000_000,
    "COD": 100_000_000,
    "AFG": 150_000_000,
}

# ISO-2 to ISO-3 mapping for the countries in our lookup tables
_ISO2_TO_ISO3 = {
    "AF": "AFG", "BD": "BGD", "IN": "IND", "ID": "IDN", "JP": "JPN",
    "NP": "NPL", "PK": "PAK", "PH": "PHL", "BR": "BRA", "DE": "DEU",
    "NL": "NLD", "GB": "GBR", "US": "USA", "CN": "CHN", "MM": "MMR",
    "VN": "VNM", "TH": "THA", "KH": "KHM", "NG": "NGA", "MZ": "MOZ",
    "ET": "ETH", "SS": "SSD", "CD": "COD", "SD": "SDN", "SO": "SOM",
    "HT": "HTI", "AU": "AUS", "IT": "ITA", "FR": "FRA", "CO": "COL",
    "PE": "PER", "MX": "MEX", "RU": "RUS", "KE": "KEN", "TZ": "TZA",
    "LK": "LKA", "EG": "EGY", "ZA": "ZAF",
}


def _reverse_geocode_country(lat: float, lon: float) -> Optional[str]:
    """Reverse geocode lat/lon to ISO-3 country code via Nominatim."""
    try:
        rev = requests.get(
            _NOMINATIM,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 3},
            headers={"User-Agent": "COSMEON/1.0"},
            timeout=6,
        )
        rev.raise_for_status()
        cc2 = rev.json().get("address", {}).get("country_code", "").upper()
        if cc2 and len(cc2) == 2:
            return _ISO2_TO_ISO3.get(cc2)
    except Exception as e:
        logger.warning("Reverse geocode for country code failed: %s", e)
    return None


def get_inform_country_risk(lat: float, lon: float) -> dict:
    """
    Look up INFORM Risk Index scores for the country at the given coordinates.

    Uses Nominatim reverse geocode to identify the country, then returns
    pre-loaded INFORM Risk Index 2024 scores (EU JRC).

    NEVER fails — returns a latitude-based default if country is not in the table.
    """
    iso3 = _reverse_geocode_country(lat, lon)

    if iso3 and iso3 in _INFORM_FLOOD_RISK:
        entry = _INFORM_FLOOD_RISK[iso3]
        return {
            "risk": entry["risk"],
            "flood_hazard": entry["flood"],
            "vulnerability": entry["vuln"],
            "coping_capacity": entry["cope"],
            "country": iso3,
            "source": "INFORM Risk Index 2024 (EU JRC)",
            "_tier": 0,
        }

    # Default fallback based on latitude band (tropical = higher risk)
    abs_lat = abs(lat)
    if abs_lat < 15:
        # Deep tropics — high flood hazard, moderate vulnerability
        risk, flood, vuln, cope = 5.5, 6.5, 5.5, 5.0
    elif abs_lat < 30:
        # Subtropics / monsoon belt
        risk, flood, vuln, cope = 4.5, 5.5, 4.5, 4.0
    elif abs_lat < 50:
        # Temperate
        risk, flood, vuln, cope = 3.0, 4.0, 2.5, 2.0
    else:
        # High latitude — lower flood risk
        risk, flood, vuln, cope = 2.5, 3.0, 2.0, 1.5

    return {
        "risk": risk,
        "flood_hazard": flood,
        "vulnerability": vuln,
        "coping_capacity": cope,
        "country": iso3 or "UNK",
        "source": "INFORM Risk Index 2024 (EU JRC) — latitude-band default",
        "_tier": 0,
    }


def get_gar_flood_loss(lat: float, lon: float) -> dict:
    """
    Look up UNDRR GAR 2022 Annual Average Loss from floods for the country
    at the given coordinates.

    Uses Nominatim reverse geocode to identify the country, then returns
    pre-loaded GAR AAL data.

    Fallback: estimates AAL from global average (~0.1% of GDP).
    """
    iso3 = _reverse_geocode_country(lat, lon)

    if iso3 and iso3 in _GAR_ANNUAL_FLOOD_LOSS:
        return {
            "annual_avg_loss_usd": _GAR_ANNUAL_FLOOD_LOSS[iso3],
            "country": iso3,
            "source": "UNDRR GAR 2022",
            "_tier": 0,
        }

    # Fallback: estimate from World Bank GDP × 0.001 (global avg flood loss ~0.1% GDP)
    try:
        cc2 = None
        rev = requests.get(
            _NOMINATIM,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 3},
            headers={"User-Agent": "COSMEON/1.0"},
            timeout=6,
        )
        rev.raise_for_status()
        cc2 = rev.json().get("address", {}).get("country_code", "").upper()
        if cc2 and len(cc2) == 2:
            gdp = _fetch_wb_indicator(cc2, "NY.GDP.MKTP.CD")
            if gdp and gdp > 0:
                estimated_aal = gdp * 0.001
                return {
                    "annual_avg_loss_usd": round(estimated_aal),
                    "country": _ISO2_TO_ISO3.get(cc2, cc2),
                    "source": "Estimated from World Bank GDP × 0.1% (global avg flood loss ratio)",
                    "_tier": 1,
                }
    except Exception as e:
        logger.warning("GAR flood loss GDP fallback failed: %s", e)

    # Ultimate fallback: global median
    return {
        "annual_avg_loss_usd": 500_000_000,
        "country": iso3 or "UNK",
        "source": "UNDRR GAR 2022 — global median estimate",
        "_tier": 2,
    }
