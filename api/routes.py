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


@app.get("/api/alerts")
def get_alerts(limit: int = Query(default=50, le=200)):
    """Get automated flood alerts generated by the detection engine."""
    alerts = analysis_engine.get_alerts(limit=limit)
    return {
        "count": len(alerts),
        "alerts": alerts,
    }


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


@app.get("/api/detection/status/all")
def get_detection_status():
    """Get overall detection system status."""
    return analysis_engine.get_system_status()


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


# --- Any-Location Analysis ---

from pydantic import BaseModel

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

