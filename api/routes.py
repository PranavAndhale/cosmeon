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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from database.db import DatabaseManager
from processing.external_data import ExternalDataIntegrator
from processing.predictor import FloodPredictor

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
predictor = FloodPredictor()

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

    # Get external factors
    factors = external.get_risk_factors(region.bbox)

    # Run prediction
    prediction = predictor.predict(
        flood_history=history_dicts,
        external_factors=factors.to_dict(),
        region_name=region.name,
    )

    return prediction.to_dict()


# --- Explainability ---

@app.get("/api/explain/location")
def explain_location(lat: float = Query(...), lon: float = Query(...)):
    """
    Full ML vs GloFAS explainability for any arbitrary coordinates.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    name = f"{lat:.2f}, {lon:.2f}"

    # 1. Get external factors for these coords
    factors = external.get_risk_factors_by_coords(lat, lon)

    # 2. Run ML prediction with full explanation (no history for ad-hoc)
    ml_result = predictor.explain_prediction(
        flood_history=[],
        external_factors=factors.to_dict(),
        region_name=name,
    )

    # 3. Fetch LIVE GloFAS discharge
    from processing.live_flood_data import LiveFloodDataFetcher, generate_difference_analysis
    live_fetcher = LiveFloodDataFetcher()
    discharge = live_fetcher.fetch_river_discharge(lat, lon, past_days=30, forecast_days=7)

    # 4. Generate difference analysis
    comparison = generate_difference_analysis(
        our_level=ml_result["risk_level"],
        our_feature_values=ml_result["feature_values"],
        our_explanation=ml_result["explanation"],
        glofas_level=discharge.flood_risk_level,
        discharge_m3s=discharge.current_discharge,
        anomaly_sigma=discharge.discharge_anomaly,
        mean_discharge=discharge.mean_discharge,
    )

    return {
        "region": {"id": -1, "name": name, "bbox": [lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5]},
        "ml_prediction": {
            "risk_level": ml_result["risk_level"],
            "probability": ml_result["probability"],
            "confidence": ml_result["confidence"],
            "class_probabilities": ml_result["class_probabilities"],
            "feature_values": ml_result["feature_values"],
            "top_drivers": ml_result["top_drivers"],
            "explanation": ml_result["explanation"],
            "model_inputs_source": ml_result["model_inputs_source"],
        },
        "glofas_assessment": {
            "risk_level": discharge.flood_risk_level,
            "discharge_m3s": round(discharge.current_discharge, 2),
            "anomaly_sigma": round(discharge.discharge_anomaly, 2),
            "mean_discharge_m3s": round(discharge.mean_discharge, 2),
            "explanation": comparison["glofas_explanation"],
            "historical_discharge": discharge.discharge_m3s[-30:],
            "historical_dates": discharge.dates[-30:],
            "forecast_discharge": discharge.forecast_discharge[-7:],
            "forecast_dates": discharge.forecast_dates[-7:],
        },
        "comparison": {
            "agreement": comparison["agreement"],
            "agreement_score": comparison["agreement_score"],
            "summary": comparison["summary"],
            "difference_reasons": comparison["difference_reasons"],
            "our_methodology": comparison["our_methodology"],
            "glofas_methodology": comparison["glofas_methodology"],
        },
        "independence_proof": {
            "model_uses": (
                "13 weather and terrain features: precipitation, soil moisture, temperature, "
                "elevation, seasonal month, risk multiplier, historical trends."
            ),
            "model_does_not_use": "River discharge data (GloFAS) is NOT an input to the prediction model.",
            "glofas_uses": f"River discharge measurements from the nearest GloFAS grid cell at ({lat:.2f}, {lon:.2f}).",
            "how_training_works": (
                "During training, GloFAS discharge was used ONLY to generate labels. "
                "The model's input features are exclusively weather and terrain data."
            ),
            "verification": "ML prediction is computed FIRST, then GloFAS data is fetched separately for comparison.",
            "feature_data_sources": {
                "Open-Meteo Weather API": ["precip_7d", "precip_30d", "max_daily_rain_7d", "precip_anomaly", "soil_moisture", "temperature", "rainfall_mm"],
                "Open-Meteo Elevation API": ["elevation_m"],
                "Calendar / Seasonal": ["month", "risk_multiplier"],
            },
        },
    }


@app.get("/api/explain/{region_id}")
def explain_prediction(region_id: int):
    """
    Full prediction explainability for a region.

    Returns side-by-side comparison of our ML prediction vs GloFAS,
    with detailed reasons for any differences, feature-level analysis,
    and proof of model independence.

    This endpoint:
      1. Runs the ML prediction (weather-only features)
      2. Fetches live GloFAS discharge (independent ground truth)
      3. Generates human-readable analysis of why they agree or differ
      4. Proves model independence (different input data)
    """
    region = db.get_region(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

    # 1. Get external factors (weather + terrain — NO discharge)
    factors = external.get_risk_factors(region.bbox)

    # 2. Get recent risk history from database
    history = db.get_risk_history(region_id, limit=10)
    history_dicts = [r.to_dict() for r in history]

    # 3. Run ML prediction with full explanation
    ml_result = predictor.explain_prediction(
        flood_history=history_dicts,
        external_factors=factors.to_dict(),
        region_name=region.name,
    )

    # 4. Fetch LIVE GloFAS discharge (completely independent)
    from processing.live_flood_data import LiveFloodDataFetcher, generate_difference_analysis
    live_fetcher = LiveFloodDataFetcher()
    lat = (region.bbox[1] + region.bbox[3]) / 2
    lon = (region.bbox[0] + region.bbox[2]) / 2
    discharge = live_fetcher.fetch_river_discharge(lat, lon, past_days=30, forecast_days=7)

    # 5. Generate difference analysis
    comparison = generate_difference_analysis(
        our_level=ml_result["risk_level"],
        our_feature_values=ml_result["feature_values"],
        our_explanation=ml_result["explanation"],
        glofas_level=discharge.flood_risk_level,
        discharge_m3s=discharge.current_discharge,
        anomaly_sigma=discharge.discharge_anomaly,
        mean_discharge=discharge.mean_discharge,
    )

    return {
        "region": {
            "id": region.id,
            "name": region.name,
            "bbox": region.bbox,
        },
        "ml_prediction": {
            "risk_level": ml_result["risk_level"],
            "probability": ml_result["probability"],
            "confidence": ml_result["confidence"],
            "class_probabilities": ml_result["class_probabilities"],
            "feature_values": ml_result["feature_values"],
            "top_drivers": ml_result["top_drivers"],
            "explanation": ml_result["explanation"],
            "model_inputs_source": ml_result["model_inputs_source"],
        },
        "glofas_assessment": {
            "risk_level": discharge.flood_risk_level,
            "discharge_m3s": round(discharge.current_discharge, 2),
            "anomaly_sigma": round(discharge.discharge_anomaly, 2),
            "mean_discharge_m3s": round(discharge.mean_discharge, 2),
            "explanation": comparison["glofas_explanation"],
            "historical_discharge": discharge.discharge_m3s[-30:],
            "historical_dates": discharge.dates[-30:],
            "forecast_discharge": discharge.forecast_discharge[-7:],
            "forecast_dates": discharge.forecast_dates[-7:],
        },
        "comparison": {
            "agreement": comparison["agreement"],
            "agreement_score": comparison["agreement_score"],
            "summary": comparison["summary"],
            "difference_reasons": comparison["difference_reasons"],
            "our_methodology": comparison["our_methodology"],
            "glofas_methodology": comparison["glofas_methodology"],
        },
        "independence_proof": {
            "model_uses": (
                "13 weather and terrain features: precipitation (7-day, 30-day, max daily, "
                "anomaly z-score), soil moisture, temperature, elevation, seasonal month, "
                "risk multiplier, historical flood percentage trends."
            ),
            "model_does_not_use": (
                "River discharge data (GloFAS) is NOT an input to the prediction model. "
                "The model has zero access to discharge measurements at prediction time."
            ),
            "glofas_uses": (
                "River discharge measurements (m3/s) from the nearest GloFAS hydrological "
                f"grid cell at coordinates ({lat:.2f}, {lon:.2f}). Risk is derived from "
                "statistical deviation of current discharge from the historical mean."
            ),
            "how_training_works": (
                "During training, GloFAS discharge data was used ONLY to generate labels "
                "(ground truth). The model's input features are exclusively weather and "
                "terrain data. This means the model learned: 'given weather conditions X, "
                "the actual flood outcome was Y.' At prediction time, it uses only weather "
                "to predict the outcome - it never sees discharge data."
            ),
            "verification": (
                "The ML prediction is computed FIRST, then GloFAS data is fetched separately "
                "for comparison. There is no data path where discharge feeds into the prediction."
            ),
            "feature_data_sources": {
                "Open-Meteo Weather API": [
                    "precip_7d", "precip_30d", "max_daily_rain_7d",
                    "precip_anomaly", "soil_moisture", "temperature", "rainfall_mm",
                ],
                "Open-Meteo Elevation API": ["elevation_m"],
                "Calendar / Seasonal": ["month", "risk_multiplier"],
                "Historical Database (past assessments)": [
                    "mean_flood_pct", "max_flood_pct", "trend",
                ],
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

    # Run our prediction first
    history = db.get_risk_history(region_id, limit=10)
    history_dicts = [r.to_dict() for r in history]
    factors = external.get_risk_factors(region.bbox)
    our_prediction = predictor.predict(
        flood_history=history_dicts,
        external_factors=factors.to_dict(),
        region_name=region.name,
    )

    # Cross-validate against GloFAS
    from processing.live_flood_data import LiveFloodDataFetcher
    live_fetcher = LiveFloodDataFetcher()
    lat = (region.bbox[1] + region.bbox[3]) / 2
    lon = (region.bbox[0] + region.bbox[2]) / 2

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


# --- Live Automated Analysis ---

from processing.live_analysis import LiveAnalysisEngine
analysis_engine = LiveAnalysisEngine()


@app.on_event("startup")
async def auto_analyze_on_startup():
    """Automatically run live analysis on all regions when server starts."""
    import asyncio
    logger.info("=== AUTO-ANALYSIS: Running live detection on all regions ===")

    async def run_analysis():
        await asyncio.sleep(3)  # Wait for DB to be ready
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

    asyncio.create_task(run_analysis())


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

    # Also store as a real risk assessment in the database
    from database.models import RiskAssessmentRecord, get_session
    session = get_session()
    try:
        record = RiskAssessmentRecord(
            region_id=region.id,
            timestamp=datetime.fromisoformat(result.timestamp),
            risk_level=result.detected_risk_level,
            flood_area_km2=0,  # Not from satellite
            total_area_km2=0,
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


@app.get("/api/geocode")
def geocode_search(q: str = Query(..., min_length=2)):
    """
    Forward geocoding: convert place name → lat/lon coordinates.
    Uses OpenStreetMap Nominatim (free, no key required).
    Returns up to 8 results with display_name, lat, lon, type.
    """
    try:
        encoded = urllib.parse.quote(q)
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={encoded}&format=json&limit=8&addressdetails=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "COSMEON/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())

        results = []
        for item in data:
            results.append({
                "display_name": item.get("display_name", ""),
                "lat": float(item.get("lat", 0)),
                "lon": float(item.get("lon", 0)),
                "type": item.get("type", ""),
                "class": item.get("class", ""),
                "importance": float(item.get("importance", 0)),
            })

        return {"query": q, "count": len(results), "results": results}
    except Exception as e:
        logger.warning("Geocode failed for '%s': %s", q, e)
        return {"query": q, "count": 0, "results": [], "error": str(e)}


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

        # Prediction
        prediction_data = None
        try:
            from processing.predictor import FloodPredictor
            predictor = FloodPredictor()
            history = db.get_risk_history(region_id, limit=10)
            hist_dicts = [h.to_dict() for h in history]
            from processing.external_data import ExternalDataIntegrator
            ext = ExternalDataIntegrator()
            factors = ext.get_risk_factors(region.bbox)
            pred = predictor.predict(hist_dicts, factors.to_dict(), region.name)
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


@app.post("/api/forecast/location")
def forecast_location(body: LocationRequest):
    """Generate a forecast for arbitrary coordinates."""
    try:
        engine = _get_forecast_engine()
        forecast = engine.generate_forecast(
            lat=body.lat, lon=body.lon,
            region_name=body.name or f"{body.lat:.2f}, {body.lon:.2f}",
            horizon_months=6,
        )
        return forecast
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
    """Compute compound multi-hazard risk for a region."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}
        bbox = region.bbox
        lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
        # Get fusion data for compound risk inputs
        fusion = _get_fusion().fuse_sensors(lat, lon, region.name)
        risk = db.get_latest_risk(region_id)
        result = _get_compound().compute_compound_risk(
            flood_probability=risk.flood_percentage if risk else 0.0,
            flood_confidence=risk.confidence_score if risk else 0.0,
            vegetation_stress=fusion.vegetation_stress,
            thermal_anomaly=fusion.thermal_anomaly,
            soil_saturation=fusion.soil_saturation,
            region_name=region.name,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception("Compound risk failed for region %s", region_id)
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
    """Estimate financial impact for a region's flood risk."""
    try:
        region = db.get_region(region_id)
        if not region:
            return {"error": f"Region {region_id} not found"}
        risk = db.get_latest_risk(region_id)
        result = _get_financial().estimate_impact(
            risk_level=risk.risk_level if risk else "MEDIUM",
            flood_area_km2=risk.flood_area_km2 if risk else 0.0,
            total_area_km2=risk.total_area_km2 if risk else 100.0,
            flood_probability=risk.flood_percentage if risk else 0.1,
            region_name=region.name,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception("Financial impact failed for region %s", region_id)
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

    # Serve index.html inside directory (e.g. /about → /about/index.html)
    index_in_dir = static_dir / full_path / "index.html"
    if index_in_dir.is_file():
        return FileResponse(str(index_in_dir))

    # Fallback: serve root index.html (SPA catch-all)
    root_index = static_dir / "index.html"
    if root_index.is_file():
        return FileResponse(str(root_index))

    return {"message": "Frontend not built. Visit /docs for API documentation."}

