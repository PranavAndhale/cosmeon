"""
Phase 6: FastAPI REST API for retrieving processed climate risk insights.

Endpoints:
  GET  /api/regions                    → list monitored regions
  GET  /api/regions/{id}/risk          → latest risk assessment
  GET  /api/regions/{id}/history       → risk over time
  GET  /api/risk                       → all regions filtered by risk level
  GET  /api/changes                    → change detection events
  GET  /api/reports/{id}               → structured summary report
  GET  /api/reports/{id}/download      → download report as JSON
  GET  /api/logs                       → processing logs
  GET  /api/health                     → pipeline health status
  GET  /api/predict/{id}               → flood risk prediction
  GET  /api/external/{id}              → external risk factors
  GET  /api/validate/{id}              → cross-validate vs GloFAS
  GET  /api/discharge/{id}             → raw GloFAS discharge data
  POST /api/analyze/{id}               → trigger live automated analysis
  POST /api/analyze/all                → analyze all regions
  GET  /api/alerts                     → automated flood alerts
  GET  /api/detection/{id}             → latest live detection result
  GET  /                               → dashboard (static HTML)
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from database.db import DatabaseManager
from processing.external_data import ExternalDataIntegrator
from processing.tiered_predictor import TieredFloodPredictor

logger = logging.getLogger("cosmeon.api")

app = FastAPI(
    title="COSMEON Climate Risk Intelligence API",
    description="Satellite-based flood detection and climate risk assessment engine",
    version="1.0.0",
)

# CORS for dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DatabaseManager()
external = ExternalDataIntegrator()
predictor = TieredFloodPredictor()

# ─── Static frontend directory ───
# In production (Render), the Next.js static export lives in /app/static
# Locally, you can copy frontend/out/ to static/ to test
static_dir = Path(__file__).parent.parent / "static"


# --- Health ---

@app.get("/api/health")
def health_check():
    """Pipeline health status."""
    return {
        "status": "healthy",
        "service": "COSMEON Climate Risk Intelligence",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# --- Regions ---

@app.get("/api/regions")
def list_regions():
    """List all monitored regions."""
    regions = db.get_all_regions()
    return {
        "count": len(regions),
        "regions": [r.to_dict() for r in regions],
    }


@app.get("/api/regions/{region_id}")
def get_region(region_id: int):
    """Get details of a specific region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
    return region.to_dict()


# --- Risk Assessments ---

@app.get("/api/regions/{region_id}/risk")
def get_latest_risk(region_id: int):
    """Get the latest risk assessment for a region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    risk = db.get_latest_risk(region_id)
    if not risk:
        return {"region_id": region_id, "message": "No risk assessments available"}

    return risk.to_dict()


@app.get("/api/regions/{region_id}/history")
def get_risk_history(region_id: int, limit: int = Query(default=50, le=200)):
    """Get risk assessment history for a region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    history = db.get_risk_history(region_id, limit=limit)
    return {
        "region_id": region_id,
        "region_name": region.name,
        "count": len(history),
        "assessments": [r.to_dict() for r in history],
    }


@app.get("/api/risk")
def get_regions_by_risk(level: str = Query(default="HIGH", description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")):
    """Get all regions matching a risk level."""
    valid_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk level. Must be one of: {valid_levels}",
        )

    results = db.get_regions_by_risk(level.upper())
    return {
        "risk_level": level.upper(),
        "count": len(results),
        "assessments": results,
    }


# --- Situation Overview ---

def _classify_situation(
    risk_level: str,
    prev_risk_level: Optional[str],
    discharge_anomaly: float,
    precip_anomaly: float,
    soil_saturation: float,
) -> str:
    """Classify the operational flood situation into one of 5 states."""
    high_levels = {"HIGH", "CRITICAL"}
    # Active flooding: currently high AND was high before AND river elevated
    if risk_level in high_levels and prev_risk_level in high_levels and discharge_anomaly > 1.5:
        return "FLOODING_NOW"
    # Imminent: just escalated to high, or high with discharge rising
    if risk_level in high_levels and (prev_risk_level not in high_levels or discharge_anomaly > 1.0):
        return "IMMINENT"
    # Receding: was high, now dropped
    if risk_level not in high_levels and prev_risk_level in high_levels:
        return "RECEDING"
    # Watch: medium risk with building conditions
    if risk_level == "MEDIUM" and (precip_anomaly > 0.5 or soil_saturation > 0.5):
        return "WATCH"
    return "NORMAL"


@app.get("/api/situation/all")
def get_situation_all():
    """
    Cross-region situation overview with temporal state classification.
    Returns all regions ranked by severity with situation_status, trend,
    and key contributing factors in a single call.
    """
    regions = db.get_all_regions()
    results = []
    summary = {
        "flooding_now": 0, "imminent": 0, "watch": 0,
        "receding": 0, "normal": 0, "total": len(regions),
    }

    for region in regions:
        risk = db.get_latest_risk(region.id)
        if not risk:
            results.append({
                "id": region.id, "name": region.name, "bbox": region.bbox,
                "risk_level": "UNKNOWN", "situation_status": "NORMAL",
                "flood_area_km2": 0, "flood_percentage": 0,
                "confidence_score": 0, "discharge_anomaly_sigma": 0.0,
                "trend": "stable", "last_assessed": None,
                "contributing_factors": {},
            })
            summary["normal"] += 1
            continue

        # Trend: compare latest vs previous assessment
        history = db.get_risk_history(region.id, limit=2)
        prev_risk_level = history[1].risk_level if len(history) >= 2 else None
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        curr_val = risk_order.get(risk.risk_level, 0)
        prev_val = risk_order.get(prev_risk_level, 0) if prev_risk_level else curr_val
        trend = "escalating" if curr_val > prev_val else "improving" if curr_val < prev_val else "stable"

        # Fetch prediction signals for classification
        discharge_anomaly = 0.0
        precip_anomaly = 0.0
        soil_saturation = 0.0
        contributing_factors: dict = {}
        try:
            bbox = region.bbox
            lat = (bbox[1] + bbox[3]) / 2
            lon = (bbox[0] + bbox[2]) / 2
            expl = predictor.explain_prediction(
                [h.to_dict() for h in history],
                {"_lat": lat, "_lon": lon},
                region.name,
            )
            if expl and "feature_values" in expl:
                fv = expl["feature_values"]
                discharge_anomaly = fv.get("discharge_anomaly_sigma", 0.0)
                precip_anomaly = fv.get("precip_anomaly", 0.0)
                soil_saturation = fv.get("soil_saturation", 0.0)
            if expl:
                contributing_factors = {
                    "risk_level": expl.get("risk_level"),
                    "probability": expl.get("probability"),
                    "confidence": expl.get("confidence"),
                    "discharge_anomaly_sigma": round(discharge_anomaly, 2),
                    "precip_anomaly": round(precip_anomaly, 2),
                    "soil_saturation": round(soil_saturation, 3),
                    "model_inputs_source": expl.get("model_inputs_source", ""),
                }
        except Exception as e:
            logger.warning("Situation explain failed for region %s: %s", region.id, e)

        situation = _classify_situation(
            risk.risk_level, prev_risk_level,
            discharge_anomaly, precip_anomaly, soil_saturation,
        )
        summary[situation.lower()] = summary.get(situation.lower(), 0) + 1

        results.append({
            "id": region.id,
            "name": region.name,
            "bbox": region.bbox,
            "risk_level": risk.risk_level,
            "situation_status": situation,
            "flood_area_km2": risk.flood_area_km2,
            "flood_percentage": risk.flood_percentage,
            "confidence_score": risk.confidence_score,
            "discharge_anomaly_sigma": round(discharge_anomaly, 2),
            "trend": trend,
            "last_assessed": str(risk.timestamp) if risk.timestamp else None,
            "contributing_factors": contributing_factors,
        })

    # Sort: CRITICAL first, then HIGH, then by flood area descending
    risk_sort = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    results.sort(key=lambda r: (risk_sort.get(r["risk_level"], 4), -r["flood_area_km2"]))

    return {"regions": results, "summary": summary}


# --- Change Events ---

@app.get("/api/changes")
def get_change_events(
    region_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    """Get change detection events with optional filters."""
    events = db.get_change_events(
        region_id=region_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
    )
    return {
        "count": len(events),
        "events": [e.to_dict() for e in events],
    }


# --- Reports ---

@app.get("/api/reports/{region_id}")
def get_report(region_id: int):
    """Get a structured summary report for a region."""
    report = db.generate_summary_report(region_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report


@app.get("/api/reports/{region_id}/download")
def download_report(region_id: int):
    """Download a structured report as JSON file."""
    report = db.generate_summary_report(region_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    return JSONResponse(
        content=report,
        headers={
            "Content-Disposition": f'attachment; filename="cosmeon_report_region_{region_id}.json"',
        },
    )


# --- Processing Logs ---

@app.get("/api/logs")
def get_processing_logs(limit: int = Query(default=100, le=500)):
    """Get pipeline processing logs."""
    logs = db.get_processing_logs(limit=limit)
    return {
        "count": len(logs),
        "logs": [l.to_dict() for l in logs],
    }


# --- Predictions ---

@app.get("/api/predict/{region_id}")
def predict_risk(region_id: int):
    """Predict future flood risk for a region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    # Get historical assessments
    history = db.get_risk_history(region_id, limit=10)
    history_dicts = [r.to_dict() for r in history]

    # Get external factors + add lat/lon for model_hub enrichment
    factors = external.get_risk_factors(region.bbox)
    factors_dict = factors.to_dict()
    lat = (region.bbox[1] + region.bbox[3]) / 2
    lon = (region.bbox[0] + region.bbox[2]) / 2
    factors_dict["_lat"] = lat
    factors_dict["_lon"] = lon

    # Run prediction
    prediction = predictor.predict(
        flood_history=history_dicts,
        external_factors=factors_dict,
        region_name=region.name,
    )

    return prediction.to_dict()


# --- Explainability ---

@app.get("/api/explain/location")
def explain_location(
    lat: float = Query(...),
    lon: float = Query(...),
    known_ml_level: Optional[str] = Query(None),
    known_ml_probability: Optional[float] = Query(None),
):
    """
    Full prediction explainability for any arbitrary coordinates.

    GloFAS river discharge is now integrated directly as ML model features,
    so a single unified prediction is returned with full feature-level analysis.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    name = f"{lat:.2f}, {lon:.2f}"

    # 1. Get external factors + add lat/lon for enrichment (triggers GloFAS fetch internally)
    factors = external.get_risk_factors_by_coords(lat, lon)
    factors_dict = factors.to_dict()
    factors_dict["_lat"] = lat
    factors_dict["_lon"] = lon

    # 2. Run ML prediction — _enrich_with_model_hub fetches GloFAS discharge as features
    ml_result = predictor.explain_prediction(
        flood_history=[],
        external_factors=factors_dict,
        region_name=name,
    )

    compare_ml_level = known_ml_level or ml_result["risk_level"]
    compare_ml_prob  = known_ml_probability if known_ml_probability is not None else ml_result["probability"]

    return {
        "region": {"id": -1, "name": name, "bbox": [lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5]},
        "ml_prediction": {
            "risk_level": compare_ml_level,
            "probability": round(compare_ml_prob, 4),
            "confidence": ml_result["confidence"],
            "class_probabilities": ml_result["class_probabilities"],
            "feature_values": ml_result["feature_values"],
            "top_drivers": ml_result["top_drivers"],
            "explanation": ml_result["explanation"],
            "model_inputs_source": ml_result["model_inputs_source"],
        },
        "model_info": {
            "architecture": "tiered_pre_trained",
            "training_required": False,
            "description": (
                "Compound risk assessment using tiered pre-trained model outputs directly. "
                "Primary signal: GloFAS v4 river discharge (T1: operational forecast → T4: ERA5 surrogate). "
                "Compound signals: ERA5 reanalysis precipitation anomaly, ERA5/ECMWF IFS soil moisture. "
                "No custom training — predictions backed by world-class established models."
            ),
            "data_sources": {
                "GloFAS v4 (primary)": [
                    "flood_risk_level", "discharge_anomaly_sigma",
                    "discharge_current_m3s", "discharge_ratio", "forecast_max_7d_m3s",
                ],
                "ERA5 reanalysis (compound)": ["precip_7d_mm", "precip_anomaly", "precip_30d_mm"],
                "ERA5 / ECMWF IFS (compound)": ["soil_saturation"],
                "Historical Database (context)": ["mean_flood_pct"],
            },
        },
    }


@app.get("/api/explain/{region_id}")
def explain_prediction(
    region_id: int,
    known_ml_level: Optional[str] = Query(None),
    known_ml_probability: Optional[float] = Query(None),
):
    """
    Full prediction explainability for a region.

    GloFAS river discharge is now integrated directly as ML model features,
    producing a single unified prediction with full feature-level analysis.
    """
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    # 1. Get external factors + lat/lon for enrichment (triggers GloFAS fetch internally)
    factors = external.get_risk_factors(region.bbox)
    factors_dict = factors.to_dict()
    lat = (region.bbox[1] + region.bbox[3]) / 2
    lon = (region.bbox[0] + region.bbox[2]) / 2
    factors_dict["_lat"] = lat
    factors_dict["_lon"] = lon

    # 2. Get recent risk history from database
    history = db.get_risk_history(region_id, limit=10)
    history_dicts = [r.to_dict() for r in history]

    # 3. Run ML prediction — _enrich_with_model_hub fetches GloFAS discharge as features
    ml_result = predictor.explain_prediction(
        flood_history=history_dicts,
        external_factors=factors_dict,
        region_name=region.name,
    )

    compare_ml_level = known_ml_level or ml_result["risk_level"]
    compare_ml_prob  = known_ml_probability if known_ml_probability is not None else ml_result["probability"]

    return {
        "region": {
            "id": region.id,
            "name": region.name,
            "bbox": region.bbox,
        },
        "ml_prediction": {
            "risk_level": compare_ml_level,
            "probability": round(compare_ml_prob, 4),
            "confidence": ml_result["confidence"],
            "class_probabilities": ml_result["class_probabilities"],
            "feature_values": ml_result["feature_values"],
            "top_drivers": ml_result["top_drivers"],
            "explanation": ml_result["explanation"],
            "model_inputs_source": ml_result["model_inputs_source"],
        },
        "model_info": {
            "architecture": "tiered_pre_trained",
            "training_required": False,
            "description": (
                "Compound risk assessment using tiered pre-trained model outputs directly. "
                "Primary signal: GloFAS v4 river discharge (T1: operational forecast → T4: ERA5 surrogate). "
                "Compound signals: ERA5 reanalysis precipitation anomaly, ERA5/ECMWF IFS soil moisture. "
                "No custom training — predictions backed by world-class established models."
            ),
            "data_sources": {
                "GloFAS v4 (primary)": [
                    "flood_risk_level", "discharge_anomaly_sigma",
                    "discharge_current_m3s", "discharge_ratio", "forecast_max_7d_m3s",
                ],
                "ERA5 reanalysis (compound)": ["precip_7d_mm", "precip_anomaly", "precip_30d_mm"],
                "ERA5 / ECMWF IFS (compound)": ["soil_saturation"],
                "Historical Database (context)": ["mean_flood_pct"],
            },
        },
    }


# --- External Data ---

@app.get("/api/external/{region_id}")
def get_external_factors(region_id: int):
    """Get external risk factors (rainfall, elevation) for a region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    factors = external.get_risk_factors(region.bbox)
    return {
        "region_id": region_id,
        "region_name": region.name,
        "factors": factors.to_dict(),
    }


# --- Live Validation (GloFAS cross-check) ---

@app.get("/api/validate/{region_id}")
def validate_prediction(region_id: int):
    """
    Cross-validate our prediction against GloFAS river discharge data.

    Returns side-by-side comparison: our prediction vs real GloFAS forecast.
    """
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    # Run our prediction first (with lat/lon for model_hub enrichment)
    history = db.get_risk_history(region_id, limit=10)
    history_dicts = [r.to_dict() for r in history]
    factors = external.get_risk_factors(region.bbox)
    factors_dict = factors.to_dict()
    lat = (region.bbox[1] + region.bbox[3]) / 2
    lon = (region.bbox[0] + region.bbox[2]) / 2
    factors_dict["_lat"] = lat
    factors_dict["_lon"] = lon
    our_prediction = predictor.predict(
        flood_history=history_dicts,
        external_factors=factors_dict,
        region_name=region.name,
    )

    # Cross-validate against GloFAS
    from processing.live_flood_data import LiveFloodDataFetcher
    live_fetcher = LiveFloodDataFetcher()

    validation = live_fetcher.validate_prediction(
        lat=lat,
        lon=lon,
        our_risk_level=our_prediction.predicted_risk_level,
        our_probability=our_prediction.flood_probability,
        our_confidence=our_prediction.confidence,
    )

    # Also get discharge chart data
    discharge = live_fetcher.fetch_river_discharge(lat, lon, past_days=30, forecast_days=7)

    return {
        "region_id": region_id,
        "region_name": region.name,
        "our_prediction": our_prediction.to_dict(),
        "validation": validation.to_dict(),
        "discharge_data": discharge.to_dict(),
    }


@app.get("/api/discharge/{region_id}")
def get_river_discharge(region_id: int):
    """Get live GloFAS river discharge data for a region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    from processing.live_flood_data import LiveFloodDataFetcher
    live_fetcher = LiveFloodDataFetcher()
    lat = (region.bbox[1] + region.bbox[3]) / 2
    lon = (region.bbox[0] + region.bbox[2]) / 2

    discharge = live_fetcher.fetch_river_discharge(lat, lon, past_days=30, forecast_days=7)
    weather_forecast = live_fetcher.fetch_weather_forecast(lat, lon, days=7)

    return {
        "region_id": region_id,
        "region_name": region.name,
        "discharge": discharge.to_dict(),
        "weather_forecast": weather_forecast,
    }


# --- Orb-specific Assessments (Infrastructure + Vegetation) ---

def _compute_orb_assessment(lat: float, lon: float, name: str) -> dict:
    """
    Compute infra exposure and vegetation stress from tiered model_hub sources.

    Infrastructure: soil saturation (ERA5/ECMWF IFS) × GloFAS flood level × World Bank pop density
    Vegetation:     FAO-56 Penman-Monteith ET0 water-balance (Open-Meteo best_match → ECMWF → ERA5)
    """
    from processing.model_hub import (
        get_vegetation_stress, get_soil_moisture,
        get_economic_data, get_river_discharge,
    )
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_soil      = pool.submit(get_soil_moisture, lat, lon)
        f_veg       = pool.submit(get_vegetation_stress, lat, lon)
        f_econ      = pool.submit(get_economic_data, lat, lon, name)
        f_discharge = pool.submit(get_river_discharge, lat, lon, 7)
        soil      = f_soil.result()
        veg       = f_veg.result()
        econ      = f_econ.result()
        discharge = f_discharge.result()

    # ── Infrastructure Exposure ──────────────────────────────────────────────
    _GLOFAS_PROB = {
        "LOW": 0.07, "MEDIUM": 0.28, "HIGH": 0.62, "CRITICAL": 0.87, "UNKNOWN": 0.14,
    }
    flood_prob  = _GLOFAS_PROB.get(discharge.get("flood_risk_level", "UNKNOWN"), 0.14)
    soil_sat    = soil.get("saturation_fraction", 0.3)
    pop_density = econ.get("pop_density_km2", 500.0)

    # Pop factor: 1500 ppl/km² = fully exposed urban zone
    pop_factor  = min(1.0, pop_density / 1500.0)
    ground_risk = flood_prob * 0.55 + soil_sat * 0.45
    exposure    = round(min(1.0, max(0.0, ground_risk * (0.75 + pop_factor * 0.25))), 3)

    if   exposure >= 0.60: infra_risk = "CRITICAL"
    elif exposure >= 0.40: infra_risk = "HIGH"
    elif exposure >= 0.18: infra_risk = "MEDIUM"
    else:                  infra_risk = "LOW"

    infra_desc = (
        f"{infra_risk.title()} infrastructure vulnerability: {soil_sat * 100:.0f}% soil saturation, "
        f"{pop_density:.0f} ppl/km² population exposure, "
        f"{discharge.get('flood_risk_level', 'UNKNOWN')} GloFAS river risk."
    )

    # ── Vegetation Assessment ────────────────────────────────────────────────
    stress   = veg.get("stress_index", 0.3)
    et0      = veg.get("et0_mm_day", 4.0)
    precip   = veg.get("precip_mm_day", 3.0)
    deficit  = round(et0 - precip, 2)

    if   stress >= 0.70: veg_risk = "CRITICAL"
    elif stress >= 0.50: veg_risk = "HIGH"
    elif stress >= 0.25: veg_risk = "MEDIUM"
    else:                veg_risk = "LOW"

    if   deficit > 2.0: veg_condition = f"Significant water deficit ({deficit:.1f} mm/day) — vegetation drought stress"
    elif deficit > 0:   veg_condition = f"Mild water deficit ({deficit:.1f} mm/day) — slight vegetation stress"
    else:               veg_condition = f"Water surplus ({abs(deficit):.1f} mm/day) — vegetation well-watered"

    return {
        "name": name,
        "infra": {
            "risk_level":      infra_risk,
            "exposure_score":  exposure,
            "soil_saturation": round(soil_sat, 3),
            "pop_density_km2": round(pop_density, 0),
            "flood_factor":    round(flood_prob, 3),
            "glofas_risk":     discharge.get("flood_risk_level", "UNKNOWN"),
            "description":     infra_desc,
            "source":          (
                f"{soil.get('source', 'ERA5')} (T{soil.get('_tier', '?')}) + "
                f"World Bank pop density (T{econ.get('_tier', '?')}) + "
                f"GloFAS discharge (T{discharge.get('_tier', '?')})"
            ),
        },
        "veg": {
            "risk_level":    veg_risk,
            "stress_index":  round(stress, 3),
            "et0_mm_day":    round(et0, 2),
            "precip_mm_day": round(precip, 2),
            "deficit_mm_day": deficit,
            "condition":     veg_condition,
            "source":        f"{veg.get('source', 'FAO-56 ET0')} (T{veg.get('_tier', '?')})",
        },
    }


@app.get("/api/orb-assessment/{region_id}")
def get_orb_assessment(region_id: int):
    """
    Orb-specific risk assessments for the Infrastructure and Vegetation orbs.

    Infrastructure uses: ERA5/ECMWF soil saturation + GloFAS discharge + World Bank population
    Vegetation uses:     FAO-56 ET0 water-balance (Open-Meteo T1→T3)
    """
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
    bbox = region.bbox
    lat  = (bbox[1] + bbox[3]) / 2
    lon  = (bbox[0] + bbox[2]) / 2
    return _compute_orb_assessment(lat, lon, region.name)


class _OrbLocationRequest(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None


@app.post("/api/orb-assessment/location")
def orb_assessment_location(body: _OrbLocationRequest):
    """Orb-specific assessments for an ad-hoc lat/lon location."""
    return _compute_orb_assessment(
        body.lat, body.lon,
        body.name or f"{body.lat:.2f},{body.lon:.2f}",
    )


# --- Live Automated Analysis ---

from processing.live_analysis import LiveAnalysisEngine
analysis_engine = LiveAnalysisEngine()


# --- Auth + Periodic Scheduler Setup ---
from api.auth import (
    hash_password, verify_password, create_token, decode_token,
    get_current_user, get_optional_user, require_role,
)
from processing.periodic_scheduler import PeriodicScheduler

periodic_scheduler = PeriodicScheduler(interval_hours=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"  # admin, analyst, viewer


class SchedulerConfigRequest(BaseModel):
    interval_hours: Optional[int] = None
    enabled: Optional[bool] = None


@app.on_event("startup")
async def auto_analyze_on_startup():
    """Auto-analyze all regions on startup, pre-train ML model, and start scheduler."""
    import threading
    logger.info("=== AUTO-ANALYSIS: Running live detection on all regions ===")

    def _train_predictor_background():
        """Pre-train the ML predictor in a background thread so it is ready
        before the first prediction request arrives, avoiding in-band blocking."""
        if predictor.is_trained:
            logger.info("Predictor already trained (loaded from disk) — skipping startup training")
            return
        logger.info("Pre-training ML predictor in background thread...")
        try:
            predictor.train_on_real_data()
            logger.info("ML predictor pre-training complete (is_trained=%s)", predictor.is_trained)
        except Exception as e:
            logger.error("Background ML training failed: %s", e)

    def _startup_background():
        """Run all blocking startup work in a background thread so the async
        event loop stays free and Render's health check always responds."""
        import time
        time.sleep(3)  # Wait for DB to be ready

        # Pre-train ML model
        _train_predictor_background()

        # Run initial region analysis (makes external ERA5 / GloFAS calls)
        try:
            all_regions = db.get_all_regions()
            if all_regions:
                regions_data = [
                    {"id": r.id, "name": r.name, "bbox": r.bbox}
                    for r in all_regions
                ]
                results = analysis_engine.analyze_all_regions(regions_data)
                logger.info(
                    "Auto-analysis complete: %d regions analyzed, %d alerts generated",
                    len(results),
                    sum(1 for r in results if r.alert_triggered),
                )
        except Exception as e:
            logger.error("Startup region analysis failed: %s", e)

        # Start periodic scheduler after initial analysis
        periodic_scheduler.start(analysis_engine, db)

    startup_thread = threading.Thread(
        target=_startup_background, name="startup-worker", daemon=True
    )
    startup_thread.start()


# ── Analyze Location (must come BEFORE /api/analyze/{region_id}) ──────────
class LocationRequest(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None


@app.post("/api/analyze/location")
def analyze_location(req: LocationRequest):
    """
    Run live flood risk analysis for any arbitrary lat/lon on Earth.
    Combines live detection + ML prediction + GloFAS validation.
    """
    lat, lon = req.lat, req.lon
    name = req.name or f"{lat:.2f}, {lon:.2f}"

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    # 1. Live detection (discharge, rainfall, etc.)
    detection = analysis_engine.analyze_by_coords(lat, lon, name)

    # 2. ML prediction (GBM + LSTM ensemble)
    ml_prediction = predictor.predict_by_coords(lat, lon, name)

    # 3. GloFAS validation
    from processing.live_flood_data import LiveFloodDataFetcher
    live_fetcher = LiveFloodDataFetcher()
    discharge = live_fetcher.fetch_river_discharge(lat, lon, past_days=30, forecast_days=7)

    validation_result = live_fetcher.validate_prediction(
        lat=lat, lon=lon,
        our_risk_level=ml_prediction.predicted_risk_level,
        our_probability=ml_prediction.flood_probability,
        our_confidence=ml_prediction.confidence,
    )

    return {
        "status": "analysis_complete",
        "location": {"lat": lat, "lon": lon, "name": name},
        "detection": detection.to_dict(),
        "prediction": ml_prediction.to_dict(),
        "validation": validation_result.to_dict(),
        "discharge_data": discharge.to_dict(),
    }


@app.post("/api/analyze/all")
def run_analysis_all_regions():
    """Run live automated analysis on ALL monitored regions."""
    all_regions = db.get_all_regions()
    if not all_regions:
        return {"status": "no_regions", "results": []}

    regions_data = [
        {"id": r.id, "name": r.name, "bbox": r.bbox}
        for r in all_regions
    ]
    results = analysis_engine.analyze_all_regions(regions_data)

    return {
        "status": "analysis_complete",
        "regions_analyzed": len(results),
        "alerts_generated": sum(1 for r in results if r.alert_triggered),
        "results": [r.to_dict() for r in results],
    }


@app.post("/api/analyze/{region_id}")
def run_live_analysis(region_id: int):
    """
    Trigger live automated flood risk analysis for a region.

    Fetches real-time data from GloFAS (river discharge), Open-Meteo (weather),
    and performs multi-factor risk detection. This is AUTOMATED DETECTION,
    not simple visualization.
    """
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    result = analysis_engine.analyze_region(
        region_id=region.id,
        region_name=region.name,
        bbox=region.bbox,
    )

    # Compute real area from bounding box geometry
    import math
    bbox = region.bbox
    lat_diff = abs(bbox[3] - bbox[1])
    lon_diff = abs(bbox[2] - bbox[0])
    lat_mid = (bbox[1] + bbox[3]) / 2
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat_mid))
    total_area_km2 = lat_diff * km_per_deg_lat * lon_diff * km_per_deg_lon
    flood_area_km2 = total_area_km2 * result.flood_probability

    # Also store as a real risk assessment in the database
    from database.models import RiskAssessmentRecord, get_session
    session = get_session()
    try:
        record = RiskAssessmentRecord(
            region_id=region.id,
            timestamp=datetime.fromisoformat(result.timestamp),
            risk_level=result.detected_risk_level,
            flood_area_km2=round(flood_area_km2, 2),
            total_area_km2=round(total_area_km2, 2),
            flood_percentage=result.flood_probability,
            confidence_score=result.confidence_score,
            change_type="LIVE_DETECTION",
            water_change_pct=0,
            source_dataset="live_hydrometeorological",
            source_items=["GloFAS", "Open-Meteo"],
            assessment_details=result.risk_factors,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

    return {
        "status": "analysis_complete",
        "detection": result.to_dict(),
    }


@app.get("/api/alerts")
def get_alerts(limit: int = Query(default=50, le=200)):
    """Get automated flood alerts generated by the detection engine."""
    alerts = analysis_engine.get_alerts(limit=limit)
    return {
        "count": len(alerts),
        "alerts": alerts,
    }


@app.get("/api/detection/status/all")
def get_detection_status():
    """Get overall detection system status."""
    return analysis_engine.get_system_status()


@app.get("/api/detection/{region_id}")
def get_latest_detection(region_id: int):
    """Get the latest live detection result for a region."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    detection = analysis_engine.get_latest_detection(region_id)
    if not detection:
        return {
            "region_id": region_id,
            "message": "No live detection available. Trigger analysis via POST /api/analyze/{region_id}",
        }

    return {
        "region_id": region_id,
        "region_name": region.name,
        "detection": detection,
    }


# --- Model Training ---

@app.post("/api/train")
def train_prediction_model(synthetic: bool = False):
    """
    Trigger retraining of the flood prediction model.

    By default trains on real historical weather data from Open-Meteo APIs.
    Set ?synthetic=true to use synthetic data instead.

    Returns training metrics: accuracy, CV score, classification report, feature importances.
    """
    logger.info("=== MODEL TRAINING TRIGGERED (synthetic=%s) ===", synthetic)

    if synthetic:
        data = predictor._generate_synthetic_data(500)
        metrics = predictor.train(data)
        metrics["data_source"] = "synthetic"
    else:
        metrics = predictor.train_on_real_data()

    return {
        "status": "training_complete",
        "metrics": metrics,
    }


@app.get("/api/train/metrics")
def get_training_metrics():
    """Get the most recent model training metrics."""
    metrics = predictor.get_training_metrics()
    if not metrics:
        return {"status": "no_training_yet", "message": "Model has not been trained. POST /api/train to train."}
    return {"status": "ok", "metrics": metrics}


# --- Geocoding (place name → coordinates) ---

import urllib.request
import urllib.parse
import json as _json


def _geocode_nominatim(q: str, limit: int = 8) -> list:
    """
    Tier-1 geocoder: OpenStreetMap Nominatim.
    Most comprehensive, works worldwide. Rate limit: 1 req/s (handled by frontend debounce).
    """
    encoded = urllib.parse.quote(q)
    url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={encoded}&format=json&limit={limit}&addressdetails=1&accept-language=en"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "COSMEON/1.0 geocoder"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = _json.loads(resp.read().decode())
    results = []
    for item in data:
        results.append({
            "display_name": item.get("display_name", ""),
            "lat": float(item.get("lat", 0)),
            "lon": float(item.get("lon", 0)),
            "type": item.get("type", "place"),
            "class": item.get("class", ""),
            "importance": float(item.get("importance", 0)),
        })
    return results


def _geocode_photon(q: str, limit: int = 8) -> list:
    """
    Tier-2 geocoder: Photon by Komoot (OSM-based, separate infrastructure from Nominatim).
    Free, no API key, returns GeoJSON.
    """
    encoded = urllib.parse.quote(q)
    url = f"https://photon.komoot.io/api/?q={encoded}&limit={limit}&lang=en"
    req = urllib.request.Request(url, headers={"User-Agent": "COSMEON/1.0 geocoder"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = _json.loads(resp.read().decode())

    results = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [0, 0])
        lon, lat = float(coords[0]), float(coords[1])
        # Build display name from address components
        parts = [
            props.get("name", ""),
            props.get("city", props.get("town", props.get("village", ""))),
            props.get("state", ""),
            props.get("country", ""),
        ]
        display = ", ".join(p for p in parts if p)
        results.append({
            "display_name": display or f"{lat:.3f}, {lon:.3f}",
            "lat": lat,
            "lon": lon,
            "type": props.get("osm_value", props.get("type", "place")),
            "class": props.get("osm_key", ""),
            "importance": 0.5,
        })
    return results


def _geocode_geoapify(q: str, limit: int = 8) -> list:
    """
    Tier-3 geocoder: Geoapify free plan (3000 req/day, no credit card required).
    Provides worldwide coverage with high quality results.
    Uses open data from OSM + additional sources.
    """
    encoded = urllib.parse.quote(q)
    # Public demo key valid for testing — works without registration for basic use
    url = (
        f"https://api.geoapify.com/v1/geocode/search"
        f"?text={encoded}&limit={limit}&format=json&apiKey=YOUR_GEOAPIFY_KEY"
    )
    # Skip if no API key configured
    import os
    api_key = os.environ.get("GEOAPIFY_KEY", "")
    if not api_key:
        return []
    url = url.replace("YOUR_GEOAPIFY_KEY", api_key)
    req = urllib.request.Request(url, headers={"User-Agent": "COSMEON/1.0 geocoder"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = _json.loads(resp.read().decode())
    results = []
    for item in data.get("results", []):
        results.append({
            "display_name": item.get("formatted", ""),
            "lat": float(item.get("lat", 0)),
            "lon": float(item.get("lon", 0)),
            "type": item.get("result_type", "place"),
            "class": item.get("category", ""),
            "importance": float(item.get("rank", {}).get("confidence", 0.5)),
        })
    return results


@app.get("/api/geocode")
def geocode_search(q: str = Query(..., min_length=2)):
    """
    Forward geocoding with 3-tier fallback chain:
      Tier 1: Nominatim (OpenStreetMap) — most comprehensive
      Tier 2: Photon by Komoot — independent OSM-based service
      Tier 3: Geoapify — multi-source, requires GEOAPIFY_KEY env var

    Always returns results if any geocoder is reachable.
    """
    errors = []

    # Tier 1: Nominatim
    try:
        results = _geocode_nominatim(q, limit=8)
        if results:
            logger.info("Geocode (Nominatim) '%s': %d results", q, len(results))
            return {"query": q, "count": len(results), "results": results, "source": "nominatim"}
        logger.info("Nominatim returned 0 results for '%s', trying Photon", q)
    except Exception as e:
        errors.append(f"Nominatim: {e}")
        logger.warning("Nominatim geocode failed for '%s': %s", q, e)

    # Tier 2: Photon (Komoot)
    try:
        results = _geocode_photon(q, limit=8)
        if results:
            logger.info("Geocode (Photon) '%s': %d results", q, len(results))
            return {"query": q, "count": len(results), "results": results, "source": "photon"}
        logger.info("Photon returned 0 results for '%s', trying Geoapify", q)
    except Exception as e:
        errors.append(f"Photon: {e}")
        logger.warning("Photon geocode failed for '%s': %s", q, e)

    # Tier 3: Geoapify (if API key configured)
    try:
        results = _geocode_geoapify(q, limit=8)
        if results:
            logger.info("Geocode (Geoapify) '%s': %d results", q, len(results))
            return {"query": q, "count": len(results), "results": results, "source": "geoapify"}
    except Exception as e:
        errors.append(f"Geoapify: {e}")
        logger.warning("Geoapify geocode failed for '%s': %s", q, e)

    logger.error("All geocoders failed for '%s': %s", q, "; ".join(errors))
    return {"query": q, "count": 0, "results": [], "errors": errors}


@app.get("/api/geocode/reverse")
def reverse_geocode(lat: float = Query(...), lon: float = Query(...)):
    """
    Reverse geocoding: convert lat/lon → place name.
    Uses OpenStreetMap Nominatim (free, no key required).
    """
    try:
        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=10"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "COSMEON/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())

        display = data.get("display_name", f"{lat:.2f}, {lon:.2f}")
        # Build a short name from address parts
        addr = data.get("address", {})
        parts = []
        for key in ["city", "town", "village", "hamlet", "county", "state", "country"]:
            if key in addr:
                parts.append(addr[key])
                if len(parts) >= 2:
                    break
        short_name = ", ".join(parts) if parts else display.split(",")[0].strip()

        return {
            "lat": lat, "lon": lon,
            "display_name": display,
            "short_name": short_name,
        }
    except Exception as e:
        logger.warning("Reverse geocode failed for (%s, %s): %s", lat, lon, e)
        return {
            "lat": lat, "lon": lon,
            "display_name": f"{lat:.2f}, {lon:.2f}",
            "short_name": f"{lat:.2f}, {lon:.2f}",
        }


# --- NLG Summaries ---

_nlg_engine = None

def _get_nlg_engine():
    global _nlg_engine
    if _nlg_engine is None:
        from processing.nlg_engine import NLGEngine
        _nlg_engine = NLGEngine()
    return _nlg_engine


@app.get("/api/nlg/summary/{region_id}")
def get_nlg_summary(region_id: int):
    """Generate an AI-powered narrative summary for a region's flood analysis."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}

        # Gather all analysis data
        risk = db.get_latest_risk(region_id)
        risk_data = risk.to_dict() if risk else {"risk_level": "UNKNOWN", "flood_percentage": 0, "confidence_score": 0, "flood_area_km2": 0, "total_area_km2": 0}

        # Prediction — use global TieredFloodPredictor (GloFAS v4 + ERA5, no training delay)
        prediction_data = None
        try:
            bbox = region.bbox
            lat_c, lon_c = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
            history = db.get_risk_history(region_id, limit=10)
            hist_dicts = [h.to_dict() for h in history]
            pred = predictor.predict(hist_dicts, {"_lat": lat_c, "_lon": lon_c}, region.name)
            prediction_data = pred.to_dict() if pred else None
        except Exception:
            pass

        # Detection (use cached detection if available)
        detection_data = None
        try:
            from processing.live_flood_data import LiveFloodDetector
            detector = LiveFloodDetector()
            bbox = region.bbox
            lat, lon = (bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2
            detection_data = detector.analyze_location(lat, lon, region.name)
        except Exception:
            pass

        # Validation
        validation_data = None
        try:
            from processing.live_flood_data import LiveFloodDetector
            detector = LiveFloodDetector()
            bbox = region.bbox
            lat, lon = (bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2
            validation_data = detector.validate_with_glofas(lat, lon)
        except Exception:
            pass

        engine = _get_nlg_engine()
        result = engine.generate_executive_summary(
            region_name=region.name,
            risk_data=risk_data,
            prediction_data=prediction_data,
            detection_data=detection_data,
            validation_data=validation_data,
            region_id=region_id,
        )

        # Also generate trend narrative from history
        history = db.get_risk_history(region_id, limit=20)
        if history:
            result["trend_narrative"] = engine.generate_trend_narrative(
                [h.to_dict() for h in history]
            )

        return result

    except Exception as e:
        logger.exception("NLG summary failed for region %s", region_id)
        return {"error": str(e)}


@app.post("/api/nlg/summary/location")
def get_nlg_summary_location(body: LocationRequest):
    """Generate an AI-powered narrative summary for an arbitrary location."""
    try:
        lat, lon = body.lat, body.lon
        name = body.name or f"{lat:.2f}, {lon:.2f}"

        # Get live detection data
        detection_data = None
        try:
            det = analysis_engine.analyze_by_coords(lat, lon, name)
            detection_data = det.to_dict() if det else None
        except Exception:
            pass

        # Get ML prediction
        prediction_data = None
        try:
            pred = predictor.predict_by_coords(lat, lon, name)
            prediction_data = pred.to_dict() if pred else None
        except Exception:
            pass

        # Get GloFAS validation
        validation_data = None
        try:
            from processing.live_flood_data import LiveFloodDataFetcher
            live_fetcher = LiveFloodDataFetcher()
            val = live_fetcher.validate_prediction(
                lat=lat, lon=lon,
                our_risk_level=prediction_data.get("predicted_risk_level", "UNKNOWN") if prediction_data else "UNKNOWN",
                our_probability=prediction_data.get("flood_probability", 0) if prediction_data else 0,
                our_confidence=prediction_data.get("confidence", 0) if prediction_data else 0,
            )
            validation_data = val.to_dict() if val else None
        except Exception:
            pass

        # Build risk_data from detection
        risk_data = {
            "risk_level": detection_data.get("detected_risk_level", "UNKNOWN") if detection_data else "UNKNOWN",
            "flood_percentage": detection_data.get("flood_probability", 0) if detection_data else 0,
            "confidence_score": detection_data.get("confidence_score", 0) if detection_data else 0,
            "flood_area_km2": detection_data.get("flood_area_km2", 0) if detection_data else 0,
            "total_area_km2": detection_data.get("total_area_km2", 0) if detection_data else 0,
        }

        engine = _get_nlg_engine()
        result = engine.generate_executive_summary(
            region_name=name,
            risk_data=risk_data,
            prediction_data=prediction_data,
            detection_data=detection_data,
            validation_data=validation_data,
        )

        return result

    except Exception as e:
        logger.exception("NLG summary (location) failed")
        return {"error": str(e)}


# --- Predictive Forecasting ---

_forecast_engine = None

def _get_forecast_engine():
    global _forecast_engine
    if _forecast_engine is None:
        from processing.forecast_engine import ForecastEngine
        _forecast_engine = ForecastEngine()
    return _forecast_engine


@app.get("/api/forecast/{region_id}")
def get_forecast(region_id: int, horizon: int = Query(6, ge=1, le=12)):
    """Generate a multi-month probabilistic flood risk forecast for a region."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}

        bbox = region.bbox
        lat = (bbox[1] + bbox[3]) / 2
        lon = (bbox[0] + bbox[2]) / 2

        engine = _get_forecast_engine()
        forecast = engine.generate_forecast(
            lat=lat, lon=lon,
            region_name=region.name,
            horizon_months=horizon,
            region_id=region_id,
        )
        return forecast

    except Exception as e:
        logger.exception("Forecast failed for region %s", region_id)
        return {"error": str(e)}


def _compute_historical_flood_frequency(lat: float, lon: float) -> dict:
    """
    Compute historical heavy-rain day frequency per calendar month from ERA5 archive.
    Returns dict mapping month (1-12) to fraction of days with >20mm precipitation.
    Uses 5 years of ERA5 reanalysis data.
    """
    import requests
    from collections import defaultdict
    from datetime import timedelta

    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=5 * 365)

    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "daily": "precipitation_sum",
            },
            timeout=20,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
    except Exception:
        return {}

    dates = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    if not dates or not precip:
        return {}

    month_total = defaultdict(int)
    month_heavy = defaultdict(int)
    for d, p in zip(dates, precip):
        m = int(d[5:7])
        month_total[m] += 1
        if p is not None and p > 20:
            month_heavy[m] += 1

    return {m: month_heavy[m] / max(month_total[m], 1) for m in month_total}


def _identify_forecast_drivers(precip_mm, heavy_prob, discharge_anomaly, month, hist_freq):
    """Identify dominant forecast drivers for a given month (algorithmic, no custom model)."""
    drivers = []
    if month in (6, 7, 8, 9):
        drivers.append("monsoon_season")
    if heavy_prob > 0.30:
        drivers.append("high_ensemble_precipitation")
    elif heavy_prob > 0.15:
        drivers.append("moderate_ensemble_precipitation")
    if discharge_anomaly > 2.0:
        drivers.append("elevated_river_discharge")
    if hist_freq > 0.15:
        drivers.append("historically_flood_prone_month")
    if precip_mm > 200:
        drivers.append("very_high_precipitation_forecast")
    if not drivers:
        drivers.append("normal_conditions")
    return drivers


@app.post("/api/forecast/location")
def forecast_location(body: LocationRequest):
    """
    Generate a 6-month probabilistic flood risk forecast using established models only.

    All signals come from globally-trusted sources:
      - Precipitation: ECMWF SEAS5 ensemble (model hub T1/T2/T3)
      - River discharge: GloFAS v4 (model hub)
      - Historical baseline: ERA5 archive 5-year heavy-rain frequency
    No custom ForecastEngine or arbitrary blending.
    """
    try:
        from processing.model_hub import get_precipitation_forecast, get_river_discharge
        from datetime import timedelta

        lat = body.lat
        lon = body.lon
        name = body.name or f"{lat:.2f}, {lon:.2f}"

        # ── ECMWF SEAS5 ensemble precipitation forecast ──
        precip_data = get_precipitation_forecast(lat, lon, months=6)
        monthly_precip = precip_data.get("monthly_precip_mm", [])
        monthly_heavy = precip_data.get("monthly_prob_heavy", [])
        precip_tier = precip_data.get("_tier", 3)
        precip_source = precip_data.get("source", "unknown")

        # ── GloFAS river discharge signal ──
        discharge_data = get_river_discharge(lat, lon)
        discharge_anomaly = discharge_data.get("anomaly_sigma", 0.0)

        # ── ERA5 historical flood frequency (5-year baseline) ──
        hist_flood_freq = _compute_historical_flood_frequency(lat, lon)

        # ── Build monthly forecast from established sources only ──
        now = datetime.utcnow()
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        monthly_forecast = []

        for i in range(6):
            target = now + timedelta(days=30 * (i + 1))
            month_name = month_names[target.month - 1]

            precip_mm = monthly_precip[i] if i < len(monthly_precip) else 80.0
            heavy_prob = monthly_heavy[i] if i < len(monthly_heavy) else 0.15
            hist_freq = hist_flood_freq.get(target.month, 0.10)

            # Risk probability: ECMWF SEAS5 ensemble fraction IS the probability
            risk_prob = heavy_prob

            # Augment with GloFAS: if discharge > 2σ, boost near-term months
            if discharge_anomaly > 2.0 and i < 2:
                risk_prob += 0.15 * (1.0 - i * 0.5)

            # Blend with historical floor: don't drop below 50% of historical rate
            risk_prob = max(risk_prob, hist_freq * 0.5)

            risk_prob = round(min(0.98, max(0.02, risk_prob)), 3)

            # Risk level thresholds
            if risk_prob >= 0.70:
                risk_level = "CRITICAL"
            elif risk_prob >= 0.45:
                risk_level = "HIGH"
            elif risk_prob >= 0.20:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Confidence bounds: ECMWF SEAS5 ensemble spread (tighter for T1)
            spread = {1: 0.12, 2: 0.18}.get(precip_tier, 0.25)
            confidence_lower = round(max(0.01, risk_prob - spread), 3)
            confidence_upper = round(min(0.99, risk_prob + spread), 3)

            # Seasonal factor: this month's precip vs 6-month average
            annual_mean = sum(monthly_precip) / max(len(monthly_precip), 1) if monthly_precip else 80.0
            seasonal_factor = round(precip_mm / max(annual_mean, 1.0), 2)

            # ── Orb-specific forecast metrics ──
            # Infrastructure exposure: higher precip → more infrastructure stress
            # Scale: 0–1 based on precip relative to 300mm threshold (severe flooding)
            infra_exposure = round(min(1.0, precip_mm / 300.0) * risk_prob * seasonal_factor, 3)
            infra_exposure = min(0.98, max(0.02, infra_exposure))

            # Vegetation stress: dry months (low precip) → high stress, wet months → low
            # Inverse relationship with precipitation (FAO-56 principle)
            if precip_mm > 150:
                veg_stress_idx = round(0.05 + risk_prob * 0.1, 3)  # wet = low stress
            elif precip_mm > 80:
                veg_stress_idx = round(0.15 + (1.0 - precip_mm / 150.0) * 0.3, 3)
            else:
                veg_stress_idx = round(0.30 + (1.0 - precip_mm / 80.0) * 0.5, 3)
            veg_stress_idx = min(0.98, max(0.02, veg_stress_idx))

            monthly_forecast.append({
                "month": target.strftime("%Y-%m"),
                "month_name": month_name,
                "risk_probability": risk_prob,
                "risk_level": risk_level,
                "infra_exposure": infra_exposure,
                "vegetation_stress_index": veg_stress_idx,
                "precipitation_forecast_mm": round(precip_mm, 1),
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
                "seasonal_factor": seasonal_factor,
                "drivers": _identify_forecast_drivers(
                    precip_mm, heavy_prob, discharge_anomaly, target.month, hist_freq,
                ),
            })

        # Summary
        probs = [m["risk_probability"] for m in monthly_forecast]
        peak = max(monthly_forecast, key=lambda m: m["risk_probability"])

        # Determine overall trend: compare first half avg vs second half avg
        first_half = probs[:3]
        second_half = probs[3:]
        first_avg = sum(first_half) / max(len(first_half), 1)
        second_avg = sum(second_half) / max(len(second_half), 1)
        if second_avg > first_avg * 1.15:
            overall_trend = "escalating"
        elif second_avg < first_avg * 0.85:
            overall_trend = "declining"
        else:
            overall_trend = "stable"

        return {
            "location": {"lat": lat, "lon": lon, "name": name},
            "horizon_months": 6,
            "monthly_forecast": monthly_forecast,
            "summary": {
                "peak_risk_month": peak["month_name"],
                "peak_probability": peak["risk_probability"],
                "overall_trend": overall_trend,
                "avg_risk_probability": round(sum(probs) / len(probs), 3),
                "months_above_moderate": sum(1 for p in probs if p >= 0.20),
            },
            "data_sources": {
                "precipitation": precip_source,
                "precipitation_tier": precip_tier,
                "river_discharge": f"{discharge_data.get('source', 'GloFAS v4')} (tier {discharge_data.get('_tier', 1)})",
                "historical_baseline": "ERA5 archive (5-year heavy-rain frequency)",
            },
        }

    except Exception as e:
        logger.exception("Location forecast failed")
        return {"error": str(e)}

# --- Multi-Sensor Data Fusion (Phase 2A) ---

_fusion_engine = None

def _get_fusion():
    global _fusion_engine
    if _fusion_engine is None:
        from processing.data_fusion import DataFusionEngine
        _fusion_engine = DataFusionEngine()
    return _fusion_engine


@app.get("/api/fusion/{region_id}")
def get_fusion_analysis(region_id: int):
    """Perform multi-sensor data fusion for a region."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}
        bbox = region.bbox
        lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
        risk = db.get_latest_risk(region_id)
        flood_pct = risk.flood_percentage if risk else 0.0
        result = _get_fusion().fuse_sensors(lat, lon, region.name, flood_pct)
        return result.to_dict()
    except Exception as e:
        logger.exception("Fusion failed for region %s", region_id)
        return {"error": str(e)}


@app.post("/api/fusion/location")
def fusion_location(body: LocationRequest):
    """Multi-sensor fusion for arbitrary coordinates."""
    try:
        result = _get_fusion().fuse_sensors(body.lat, body.lon, body.name or "Custom")
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}


# --- Compound Risk (Phase 2B) ---

_compound_engine = None

def _get_compound():
    global _compound_engine
    if _compound_engine is None:
        from processing.compound_risk import CompoundRiskEngine
        _compound_engine = CompoundRiskEngine()
    return _compound_engine


@app.get("/api/compound-risk/{region_id}")
def get_compound_risk(region_id: int):
    """
    Compute compound multi-hazard risk for a region using INFORM Risk methodology.

    All inputs from established tiered sources (same as ad-hoc location):
      - Soil saturation:    ERA5 → ECMWF IFS 0.25° → ERA5 climatology
      - Vegetation stress:  FAO-56 ET0 best_match → ECMWF IFS → ERA5 archive
      - Thermal anomaly:    ERA5 climatology → CMIP6 EC-Earth3P-HR
      - Flood probability:  Live detection (GloFAS v4 + Open-Meteo)
      - Population/GDP:     World Bank country → regional → global
    """
    try:
        from processing.model_hub import (
            get_soil_moisture, get_vegetation_stress,
            get_temperature_anomaly, get_economic_data,
        )
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}
        bbox = region.bbox
        lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2

        # Live flood detection — always use real-time signal, not stale DB
        detection = analysis_engine.analyze_by_coords(lat, lon, region.name)
        flood_prob = detection.flood_probability
        confidence = detection.confidence_score

        # Fetch LIVE rainfall and elevation
        from processing.external_data import ExternalDataIntegrator
        ext = ExternalDataIntegrator()
        rainfall = ext.fetch_rainfall(lat, lon, days=7)
        elevation = ext.fetch_elevation(lat, lon)

        # ── Tiered inputs via model hub (matches ad-hoc location) ──
        sm_data   = get_soil_moisture(lat, lon)
        veg_data  = get_vegetation_stress(lat, lon)
        temp_data = get_temperature_anomaly(lat, lon)
        econ_data = get_economic_data(lat, lon, region.name)

        soil_saturation   = sm_data["saturation_fraction"]
        vegetation_stress = veg_data["stress_index"]
        thermal_anomaly   = temp_data["anomaly_c"]
        pop_density       = econ_data.get("pop_density_km2", 500.0)
        gdp_usd           = econ_data.get("gdp_usd", 50e9)

        result = _get_compound().compute_compound_risk(
            flood_probability=flood_prob,
            flood_confidence=confidence,
            vegetation_stress=vegetation_stress,
            thermal_anomaly=thermal_anomaly,
            soil_saturation=soil_saturation,
            rainfall_7d_mm=rainfall.get("total_rainfall_mm", 0),
            elevation_m=elevation.get("mean_elevation_m", 100),
            region_name=region.name,
            pop_density=pop_density,
            gdp_usd=gdp_usd,
            lat=lat,
            lon=lon,
        )
        d = result.to_dict()
        d["data_sources"] = {
            "soil_saturation":    f"{sm_data['source']} (tier {sm_data['_tier']})",
            "vegetation_stress":  f"{veg_data['source']} (tier {veg_data['_tier']})",
            "thermal_anomaly":    f"{temp_data['source']} (tier {temp_data['_tier']})",
            "economic":           f"{econ_data['source']} (tier {econ_data['_tier']})",
            "flood_probability":  "Live detection (GloFAS v4 + Open-Meteo)",
            "methodology":        "INFORM Risk Index (EU JRC)",
        }
        return d
    except Exception as e:
        logger.exception("Compound risk failed for region %s", region_id)
        return {"error": str(e)}


@app.post("/api/compound-risk/location")
def compound_risk_location(body: LocationRequest):
    """
    Compute compound multi-hazard risk using INFORM Risk Index methodology.

    All inputs from established tiered sources:
      - Soil saturation:    ERA5 → ECMWF IFS 0.25° → ERA5 climatology
      - Vegetation stress:  FAO-56 ET0 best_match → ECMWF IFS → ERA5 archive
      - Thermal anomaly:    ERA5 climatology → CMIP6 EC-Earth3P-HR
      - Flood probability:  GloFAS v4 + live detection
      - Population/GDP:     World Bank country → regional → global
    """
    try:
        from processing.model_hub import (
            get_soil_moisture, get_vegetation_stress,
            get_temperature_anomaly, get_economic_data,
        )

        lat, lon = body.lat, body.lon
        name = body.name or f"{lat:.2f}, {lon:.2f}"

        # Live flood detection (primary flood probability signal)
        detection  = analysis_engine.analyze_by_coords(lat, lon, name)
        flood_prob = detection.flood_probability
        confidence = detection.confidence_score

        from processing.external_data import ExternalDataIntegrator
        ext = ExternalDataIntegrator()
        rainfall  = ext.fetch_rainfall(lat, lon, days=7)
        elevation = ext.fetch_elevation(lat, lon)

        # ── Tiered inputs via model hub ──
        sm_data   = get_soil_moisture(lat, lon)
        veg_data  = get_vegetation_stress(lat, lon)
        temp_data = get_temperature_anomaly(lat, lon)
        econ_data = get_economic_data(lat, lon, name)

        soil_saturation   = sm_data["saturation_fraction"]
        vegetation_stress = veg_data["stress_index"]
        thermal_anomaly   = temp_data["anomaly_c"]
        pop_density       = econ_data.get("pop_density_km2", 500.0)
        gdp_usd           = econ_data.get("gdp_usd", 50e9)

        result = _get_compound().compute_compound_risk(
            flood_probability=flood_prob,
            flood_confidence=confidence,
            vegetation_stress=vegetation_stress,
            thermal_anomaly=thermal_anomaly,
            soil_saturation=soil_saturation,
            rainfall_7d_mm=rainfall.get("total_rainfall_mm", 0),
            elevation_m=elevation.get("mean_elevation_m", 100),
            region_name=name,
            pop_density=pop_density,
            gdp_usd=gdp_usd,
            lat=lat,
            lon=lon,
        )
        d = result.to_dict()
        d["data_sources"] = {
            "soil_saturation":    f"{sm_data['source']} (tier {sm_data['_tier']})",
            "vegetation_stress":  f"{veg_data['source']} (tier {veg_data['_tier']})",
            "thermal_anomaly":    f"{temp_data['source']} (tier {temp_data['_tier']})",
            "economic":           f"{econ_data['source']} (tier {econ_data['_tier']})",
            "flood_probability":  "GloFAS v4 + live detection",
            "methodology":        "INFORM Risk Index (EU JRC)",
        }
        return d
    except Exception as e:
        logger.exception("Compound risk (location) failed")
        return {"error": str(e)}


# --- Asset-Level Scoring (Phase 3A) ---

_asset_scorer = None

def _get_asset_scorer():
    global _asset_scorer
    if _asset_scorer is None:
        from processing.asset_risk_scorer import AssetRiskScorer
        _asset_scorer = AssetRiskScorer()
    return _asset_scorer


@app.get("/api/assets/{region_id}")
def get_asset_scores(region_id: int):
    """Score demo assets for a region's flood risk."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}
        bbox = region.bbox
        lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
        risk = db.get_latest_risk(region_id)
        scorer = _get_asset_scorer()
        demo_assets = scorer.generate_demo_assets(lat, lon)
        result = scorer.score_assets(
            demo_assets, lat, lon,
            flood_risk_level=risk.risk_level if risk else "MEDIUM",
            flood_probability=risk.flood_percentage if risk else 0.1,
        )
        return result
    except Exception as e:
        logger.exception("Asset scoring failed for region %s", region_id)
        return {"error": str(e)}


# --- Financial Impact (Phase 3B) ---

_financial_engine = None

def _get_financial():
    global _financial_engine
    if _financial_engine is None:
        from processing.financial_impact import FinancialImpactEngine
        _financial_engine = FinancialImpactEngine()
    return _financial_engine


@app.get("/api/financial/{region_id}")
def get_financial_impact(region_id: int):
    """Estimate financial impact using JRC depth-damage functions + World Bank data."""
    try:
        import math
        from processing.model_hub import get_economic_data, get_river_discharge
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}

        bbox = region.bbox
        lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2

        # Prefer live detection data (freshest), fall back to DB risk
        latest_det = analysis_engine.get_latest_detection(region_id)
        risk = db.get_latest_risk(region_id)

        if latest_det and latest_det.get("flood_area_km2", 0) > 0:
            flood_area = latest_det["flood_area_km2"]
            total_area = latest_det["total_area_km2"]
            flood_prob = latest_det["flood_probability"]
            risk_level = latest_det["detected_risk_level"]
        elif risk:
            flood_area = risk.flood_area_km2
            total_area = risk.total_area_km2
            flood_prob = risk.flood_percentage if risk.flood_percentage else 0.1
            risk_level = risk.risk_level
            if (flood_area == 0 or total_area == 0):
                lat_mid = (bbox[1] + bbox[3]) / 2
                total_area = abs(bbox[3] - bbox[1]) * 111 * abs(bbox[2] - bbox[0]) * 111 * math.cos(math.radians(lat_mid))
                flood_area = total_area * flood_prob
        else:
            lat_mid = (bbox[1] + bbox[3]) / 2
            total_area = abs(bbox[3] - bbox[1]) * 111 * abs(bbox[2] - bbox[0]) * 111 * math.cos(math.radians(lat_mid))
            flood_prob = 0.1
            flood_area = total_area * flood_prob
            risk_level = "MEDIUM"

        # World Bank economic data + GloFAS discharge for JRC depth estimation
        econ = get_economic_data(lat, lon, region.name)
        discharge_data = get_river_discharge(lat, lon)

        result = _get_financial().estimate_impact(
            risk_level=risk_level,
            flood_area_km2=flood_area,
            total_area_km2=total_area,
            flood_probability=flood_prob,
            population_density=econ.get("pop_density_km2", 500.0),
            region_name=region.name,
            gdp_usd=econ.get("gdp_usd", 0),
            discharge_anomaly=discharge_data.get("anomaly_sigma", 0.0),
            lat=lat,
            lon=lon,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception("Financial impact failed for region %s", region_id)
        return {"error": str(e)}


@app.post("/api/financial/location")
def financial_impact_location(body: LocationRequest):
    """
    Estimate financial impact using JRC depth-damage functions + World Bank data.

    All sources established and globally trusted:
      - Economic: World Bank country → regional → global aggregate
      - Damage model: JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017)
      - Indirect costs: UNDRR Sendai Framework ratios
      - Displacement: UNHCR per-capita cost brackets
      - Flood depth: GloFAS v4 discharge anomaly
    """
    try:
        from processing.model_hub import get_economic_data, get_river_discharge

        lat, lon = body.lat, body.lon
        name = body.name or f"{lat:.2f}, {lon:.2f}"

        # Live detection for flood geometry
        detection  = analysis_engine.analyze_by_coords(lat, lon, name)
        flood_prob = detection.flood_probability
        total_area = detection.total_area_km2
        flood_area = detection.flood_area_km2
        risk_level = detection.detected_risk_level

        # ── Tiered economic data (World Bank) ──
        econ = get_economic_data(lat, lon, name)
        gdp_usd     = econ["gdp_usd"]
        pop_density = econ["pop_density_km2"]

        # ── GloFAS discharge anomaly for JRC depth estimation ──
        discharge_data = get_river_discharge(lat, lon)
        discharge_anomaly = discharge_data.get("anomaly_sigma", 0.0)

        result = _get_financial().estimate_impact(
            risk_level=risk_level,
            flood_area_km2=flood_area,
            total_area_km2=total_area,
            flood_probability=flood_prob,
            population_density=pop_density,
            region_name=name,
            gdp_usd=gdp_usd,
            discharge_anomaly=discharge_anomaly,
            lat=lat,
            lon=lon,
        )
        d = result.to_dict()

        d["data_sources"] = {
            "economic_source":  f"{econ['source']} (tier {econ['_tier']})",
            "damage_model":     "JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017)",
            "indirect_costs":   "UNDRR Sendai Framework ratios",
            "displacement":     "UNHCR per-capita cost brackets",
            "discharge_source": f"{discharge_data.get('source', 'GloFAS v4')} (tier {discharge_data.get('_tier', 1)})",
            "country_code":     econ["country_code"],
            "country_name":     econ["country_name"],
            "gdp_usd_bn":      round(gdp_usd / 1e9, 1),
            "pop_density_km2":  pop_density,
        }
        return d
    except Exception as e:
        logger.exception("Financial impact (location) failed")
        return {"error": str(e)}


# --- Automated Change Detection (Phase 4) ---

_acd_scheduler = None

def _get_acd():
    global _acd_scheduler
    if _acd_scheduler is None:
        from processing.acd_scheduler import ACDScheduler
        _acd_scheduler = ACDScheduler()
    return _acd_scheduler


@app.get("/api/acd/status")
def get_acd_status():
    """Get automated change detection monitoring status."""
    return _get_acd().get_monitoring_status()


@app.post("/api/acd/aoi")
def add_aoi(body: dict):
    """Add an Area of Interest for automated monitoring."""
    try:
        aoi = _get_acd().add_aoi(
            name=body.get("name", "Unnamed AOI"),
            lat=body.get("lat", 0), lon=body.get("lon", 0),
            radius_km=body.get("radius_km", 10),
            threshold_pct=body.get("threshold_pct", 2.0),
            check_interval_days=body.get("check_interval_days", 5),
        )
        return aoi.to_dict()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/acd/alerts")
def get_acd_alerts():
    """Get all change detection alerts."""
    return {"alerts": _get_acd().get_alerts()}


# --- Regulatory Reports (Phase 5A) ---

_report_gen = None

def _get_report_gen():
    global _report_gen
    if _report_gen is None:
        from processing.report_generator import ReportGenerator
        _report_gen = ReportGenerator()
    return _report_gen


@app.get("/api/reports/types")
def get_report_types():
    """List available report types."""
    return {"types": _get_report_gen().list_report_types()}


@app.get("/api/reports/{report_type}/{region_id}")
def generate_report(report_type: str, region_id: int):
    """Generate a regulatory report for a region."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}
        risk = db.get_latest_risk(region_id)
        risk_data = risk.to_dict() if risk else {}
        gen = _get_report_gen()
        if report_type == "tcfd":
            return gen.generate_tcfd_report(region.name, risk_data=risk_data)
        elif report_type == "sendai":
            return gen.generate_sendai_report(region.name, risk_data=risk_data)
        elif report_type == "executive":
            return gen.generate_executive_report(region.name, risk_data=risk_data)
        else:
            return {"error": f"Unknown report type: {report_type}"}
    except Exception as e:
        logger.exception("Report generation failed")
        return {"error": str(e)}


# --- MLOps Feedback (Phase 5B) ---

_feedback_engine = None

def _get_feedback():
    global _feedback_engine
    if _feedback_engine is None:
        from processing.feedback_engine import FeedbackEngine
        _feedback_engine = FeedbackEngine()
    return _feedback_engine


@app.post("/api/feedback")
def submit_feedback(body: dict):
    """Submit user feedback on a model prediction."""
    try:
        return _get_feedback().submit_feedback(
            detection_type=body.get("detection_type", "flood"),
            model_prediction=body.get("model_prediction", ""),
            user_verdict=body.get("user_verdict", "uncertain"),
            user_label=body.get("user_label", ""),
            notes=body.get("notes", ""),
            region_id=body.get("region_id"),
            lat=body.get("lat", 0), lon=body.get("lon", 0),
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/feedback/stats")
def get_feedback_stats():
    """Get feedback statistics and model accuracy tracking."""
    return _get_feedback().get_feedback_stats()


# --- LSTM Training ---

@app.post("/api/train/lstm")
def train_lstm_model():
    """Train the LSTM time-series flood prediction model."""
    logger.info("=== LSTM TRAINING TRIGGERED ===")
    try:
        from processing.lstm_trainer import LSTMDataBuilder
        from ml.lstm_model import LSTMFloodManager

        builder = LSTMDataBuilder()
        sequences, labels = builder.build_all_regions()

        if len(sequences) == 0:
            return {"status": "error", "message": "No training data could be built"}

        manager = LSTMFloodManager()
        metrics = manager.train(sequences, labels, epochs=30, batch_size=32)

        return {"status": "training_complete", "metrics": metrics}
    except Exception as e:
        logger.error("LSTM training failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/train/lstm/metrics")
def get_lstm_metrics():
    """Get LSTM model training metrics."""
    metrics = predictor.get_lstm_metrics()
    if not metrics:
        return {"status": "not_trained", "message": "LSTM model has not been trained yet."}
    return {"status": "ok", "metrics": metrics}


# --- U-Net Training ---

@app.post("/api/unet/train")
def train_unet_model(samples: int = Query(default=200, le=1000), epochs: int = Query(default=20, le=100)):
    """Train the U-Net flood segmentation model on Sen1Floods11 data."""
    logger.info("=== U-NET TRAINING TRIGGERED (samples=%d, epochs=%d) ===", samples, epochs)
    try:
        from ml.sen1floods11_loader import Sen1Floods11Loader
        from ml.unet_model import FloodModelManager

        loader = Sen1Floods11Loader()
        images, masks = loader.load_dataset(max_samples=samples)

        if len(images) == 0:
            return {"status": "error", "message": "No training data available"}

        manager = FloodModelManager(in_channels=2)
        metrics = manager.train(images, masks, epochs=epochs, batch_size=8)

        return {
            "status": "training_complete",
            "dataset_info": loader.get_dataset_info(),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error("U-Net training failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/unet/status")
def get_unet_status():
    """Get U-Net model status and training metrics."""
    from config.settings import UNET_MODEL_PATH
    trained = UNET_MODEL_PATH.exists()
    return {
        "status": "trained" if trained else "not_trained",
        "model_path": str(UNET_MODEL_PATH) if trained else None,
    }


# ─── Authentication Endpoints ────────────────────────────────────────────────

@app.post("/api/auth/login")
def login(req: LoginRequest):
    """Login with username/password, returns a JWT token."""
    user = db.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    db.update_last_login(user.id)
    token = create_token(user.id, user.username, user.role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user.to_dict(),
    }


@app.post("/api/auth/register")
def register(req: RegisterRequest, _admin=Depends(require_role("admin"))):
    """Register a new user (admin only)."""
    if req.role not in ("admin", "analyst", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role. Must be: admin, analyst, viewer")
    hashed = hash_password(req.password)
    user = db.create_user(req.username, hashed, req.role)
    if not user:
        raise HTTPException(status_code=409, detail=f"Username '{req.username}' already exists")
    return {"status": "created", "user": user.to_dict()}


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    user = db.get_user_by_username(current_user["username"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.to_dict()


@app.get("/api/auth/users")
def list_users(_admin=Depends(require_role("admin"))):
    """List all users (admin only)."""
    users = db.get_all_users()
    return {"users": [u.to_dict() for u in users]}


# ─── Historical Trend Endpoints ───────────────────────────────────────────────

def _build_real_monthly_trends(lat: float, lon: float, bbox: list, months: int) -> list:
    """
    Fetch real historical daily data from ERA5 archive and compute monthly flood
    metrics using ERA5 precipitation percentile ranking.

    For each month, total precipitation is compared against the same calendar month
    over the full ERA5 archive window (up to 5 years). avg_flood_pct is the
    percentile rank (0–100) of the month's precipitation within the historical
    distribution — a standard climatological method, not a custom formula.

    Also includes FAO-56 ET0 evapotranspiration for real vegetation stress.
    """
    import requests
    from collections import defaultdict
    from datetime import datetime, timedelta

    # Fetch extended archive: up to 5 years for percentile baseline + requested months
    end = datetime.utcnow()
    archive_years = 5
    start = end - timedelta(days=max(months * 31, archive_years * 365) + 15)

    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   end.strftime("%Y-%m-%d"),
                "daily": "precipitation_sum,temperature_2m_max,et0_fao_evapotranspiration",
            },
            timeout=20,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
    except Exception as e:
        logger.warning("Open-Meteo archive unavailable for trend: %s", e)
        return []

    dates  = daily.get("time", [])
    precip = [p if p is not None else 0.0 for p in daily.get("precipitation_sum", [])]
    et0_vals = [e if e is not None else 0.0 for e in daily.get("et0_fao_evapotranspiration", [])]

    if not dates or not precip:
        return []

    # Group by YYYY-MM
    monthly: dict = defaultdict(lambda: {"precip": [], "et0": []})
    for i, d in enumerate(dates):
        ym = d[:7]
        monthly[ym]["precip"].append(precip[i] if i < len(precip) else 0.0)
        monthly[ym]["et0"].append(et0_vals[i] if i < len(et0_vals) else 0.0)

    # Build climatological distribution: total monthly precip by calendar month
    # This is the ERA5 historical baseline for percentile ranking
    clim_totals: dict = defaultdict(list)  # month_num → [total_precip_month1, total_precip_month2, ...]
    for ym, vals in monthly.items():
        month_num = int(ym[5:7])
        total = sum(vals["precip"])
        clim_totals[month_num].append(total)

    # Only keep the last N months for output
    sorted_months = sorted(monthly.keys())
    output_months = sorted_months[-months:]

    result = []
    for month_key in output_months:
        vals = monthly[month_key]
        days_p = vals["precip"]
        days_et0 = vals["et0"]
        n_days = len(days_p)
        month_num = int(month_key[5:7])

        total_precip = sum(days_p)
        avg_daily = total_precip / max(n_days, 1)
        max_precip_day = max(days_p) if days_p else 0.0
        heavy_days = sum(1 for d in days_p if d > 20)
        extreme_days = sum(1 for d in days_p if d > 50)

        # ── ERA5 Percentile Ranking ──
        # Compare this month's total precipitation against all same-calendar-month
        # totals in the archive. The percentile rank IS the flood risk metric.
        same_month_totals = clim_totals.get(month_num, [total_precip])
        rank = sum(1 for t in same_month_totals if t <= total_precip)
        percentile = (rank / max(len(same_month_totals), 1)) * 100.0
        avg_flood_pct = round(min(100.0, percentile), 2)
        max_flood_pct = round(min(100.0, avg_flood_pct * 1.3), 2)

        # Precipitation anomaly vs same-month climatological mean
        clim_mean = sum(same_month_totals) / max(len(same_month_totals), 1)
        anomaly = (total_precip - clim_mean) / max(clim_mean, 0.1)
        water_change = round(min(30.0, max(-15.0, anomaly * 15.0)), 2)

        # ── Vegetation stress: dryness anomaly relative to climatological mean ──
        # Values centred at 50 (= normal), >50 means drier than usual (stressed),
        # <50 means wetter than usual (healthy).  Uses same clim_totals baseline
        # already built for the ERA5 percentile flood ranking above.
        total_et0 = sum(days_et0)
        clim_precip_vals = clim_totals.get(month_num, [total_precip])
        clim_precip_mean = sum(clim_precip_vals) / max(len(clim_precip_vals), 1)
        dryness = clim_precip_mean - total_precip  # positive = drier than normal
        veg_stress = round(max(0.0, min(100.0, 50.0 + (dryness / max(clim_precip_mean, 1.0)) * 50.0)), 2)

        # Risk level from ERA5 percentile ranking
        if avg_flood_pct >= 90 or extreme_days >= 5:
            risk = "CRITICAL"
        elif avg_flood_pct >= 75 or heavy_days >= 8:
            risk = "HIGH"
        elif avg_flood_pct >= 50 or heavy_days >= 3:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        result.append({
            "month":                month_key,
            "month_label":          datetime.strptime(month_key, "%Y-%m").strftime("%b %Y"),
            "avg_flood_pct":        avg_flood_pct,
            "max_flood_pct":        max_flood_pct,
            "total_precip_mm":      round(total_precip, 1),
            "max_precip_day_mm":    round(max_precip_day, 1),
            "avg_water_change_pct": water_change,
            "max_water_change_pct": round(water_change * 1.5, 2),
            "avg_vegetation_stress": veg_stress,
            "max_vegetation_stress": round(veg_stress * 1.5, 2),
            "heavy_rain_days":      heavy_days,
            "dominant_risk_level":  risk,
            "risk_distribution": {
                "LOW":      1 if risk == "LOW"      else 0,
                "MEDIUM":   1 if risk == "MEDIUM"   else 0,
                "HIGH":     1 if risk == "HIGH"     else 0,
                "CRITICAL": 1 if risk == "CRITICAL" else 0,
            },
            "assessment_count":     n_days,
        })

    return result


@app.get("/api/trends/location")
def get_trends_location(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    months: int = Query(12, ge=1, le=24),
):
    """
    Monthly aggregated risk trend data for any arbitrary lat/lon.

    Calls the real Open-Meteo ERA5 archive to derive flood-risk metrics from
    heavy-rain days and precipitation anomalies — identical methodology to the
    registered-region endpoint but requires no database entry.

    Used by the frontend time-series chart when a location is searched ad-hoc.
    """
    # Build a small bbox (~50 km) around the point for area estimates
    bbox = [lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5]
    trend = _build_real_monthly_trends(lat, lon, bbox, months)
    return {
        "lat":         lat,
        "lon":         lon,
        "months":      months,
        "data_points": len(trend),
        "trend":       trend,
    }


@app.get("/api/trends/{region_id}")
def get_trends(region_id: int, months: int = Query(12, ge=1, le=36)):
    """Monthly aggregated risk trend data — uses real Open-Meteo historical climate."""
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    bbox = region.bbox
    lat  = (bbox[1] + bbox[3]) / 2
    lon  = (bbox[0] + bbox[2]) / 2

    # Try real historical climate data first; fall back to DB aggregation
    trend = _build_real_monthly_trends(lat, lon, bbox, months)
    if not trend:
        logger.warning("Real trend fetch failed for %s — falling back to DB", region.name)
        trend = db.get_monthly_trends(region_id, months=months)

    return {
        "region_id":   region_id,
        "region_name": region.name,
        "months":      months,
        "data_points": len(trend),
        "trend":       trend,
    }


@app.get("/api/trends/global/summary")
def get_global_trends(months: int = Query(6, ge=1, le=24)):
    """Cross-region trend comparison for the global view."""
    all_regions = db.get_all_regions()
    result = []
    for region in all_regions:
        trend = db.get_monthly_trends(region.id, months=months)
        if trend:
            latest = trend[-1]
            result.append({
                "region_id": region.id,
                "region_name": region.name,
                "latest_month": latest["month_label"],
                "latest_risk": latest["dominant_risk_level"],
                "avg_flood_pct": latest["avg_flood_pct"],
                "trend_direction": (
                    "rising" if len(trend) >= 2 and trend[-1]["avg_flood_pct"] > trend[-2]["avg_flood_pct"]
                    else "falling" if len(trend) >= 2 and trend[-1]["avg_flood_pct"] < trend[-2]["avg_flood_pct"]
                    else "stable"
                ),
                "months": trend,
            })
    return {"regions": result, "months": months}


# ─── Periodic Scheduler Endpoints ─────────────────────────────────────────────

@app.get("/api/scheduler/status")
def get_scheduler_status():
    """Get periodic monitoring scheduler status (next run, interval, runs completed)."""
    return periodic_scheduler.get_status()


@app.post("/api/scheduler/configure")
def configure_scheduler(req: SchedulerConfigRequest, _admin=Depends(require_role("admin"))):
    """Configure the scheduler interval or enable/disable (admin only)."""
    periodic_scheduler.configure(
        interval_hours=req.interval_hours,
        enabled=req.enabled,
    )
    return {"status": "configured", **periodic_scheduler.get_status()}


@app.post("/api/scheduler/trigger")
def trigger_scheduler_now(current_user: dict = Depends(get_current_user)):
    """Trigger an immediate analysis run (analyst or admin)."""
    if current_user.get("role") not in ("admin", "analyst"):
        raise HTTPException(status_code=403, detail="Requires analyst or admin role")
    periodic_scheduler.trigger_now()
    return {"status": "triggered", "message": "Immediate analysis run started"}


# ─── Catch-all: Serve Next.js static frontend ───
# MUST be the LAST route so all /api/* routes take precedence
@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    """
    Serve the Next.js static export.
    Falls back to index.html for SPA client-side routing.
    """
    if not static_dir.exists():
        return {"message": "Frontend not built. Visit /docs for API documentation."}

    # Serve exact file if it exists (JS, CSS, images, etc.)
    file_path = static_dir / full_path
    if file_path.is_file():
        return FileResponse(str(file_path))

    # Next.js export generates /engine → engine.html (not engine/index.html)
    html_file = static_dir / (full_path + ".html")
    if html_file.is_file():
        return FileResponse(str(html_file))

    # Serve index.html inside directory (e.g. /about → /about/index.html)
    index_in_dir = static_dir / full_path / "index.html"
    if index_in_dir.is_file():
        return FileResponse(str(index_in_dir))

    # Fallback: serve root index.html (SPA catch-all)
    root_index = static_dir / "index.html"
    if root_index.is_file():
        return FileResponse(str(root_index))

    return {"message": "Frontend not built. Visit /docs for API documentation."}

