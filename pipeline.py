"""
COSMEON Pipeline Orchestrator.

Ties together all phases: ingestion → detection → change detection → risk assessment → storage.
"""
import logging
import time
from datetime import datetime, timedelta

from config.logging_config import setup_logging
from config.settings import DEFAULT_BBOX, DEFAULT_REGION_NAME
from ingestion.satellite_fetcher import SatelliteFetcher
from processing.flood_detector import FloodDetector
from processing.change_detector import ChangeDetector
from processing.risk_classifier import RiskClassifier
from database.db import DatabaseManager

logger = logging.getLogger("cosmeon.pipeline")


class CosmeonPipeline:
    """Main pipeline orchestrator for COSMEON."""

    def __init__(self):
        setup_logging()
        logger.info("=== Initializing COSMEON Pipeline ===")

        self.fetcher = SatelliteFetcher()
        self.flood_detector = FloodDetector()
        self.change_detector = ChangeDetector()
        self.risk_classifier = RiskClassifier()
        self.db = DatabaseManager()

        logger.info("All components initialized")

    def run(
        self,
        bbox: list[float] = None,
        region_name: str = None,
        current_date_range: str = None,
        baseline_date_range: str = None,
        limit: int = 2,
    ) -> dict:
        """
        Run the full pipeline.

        Args:
            bbox: [west, south, east, north] bounding box
            region_name: Name for the monitored region
            current_date_range: Date range for current imagery (e.g., "2024-07-01/2024-09-30")
            baseline_date_range: Date range for baseline imagery (e.g., "2024-01-01/2024-03-31")
            limit: Max items to fetch per period

        Returns:
            dict with pipeline results
        """
        bbox = bbox or DEFAULT_BBOX
        region_name = region_name or DEFAULT_REGION_NAME
        pipeline_start = time.time()

        logger.info("=== Starting COSMEON Pipeline for %s ===", region_name)

        # Set default date ranges if not provided
        if current_date_range is None:
            end = datetime.utcnow()
            start = end - timedelta(days=60)
            current_date_range = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        if baseline_date_range is None:
            end = datetime.utcnow() - timedelta(days=180)
            start = end - timedelta(days=60)
            baseline_date_range = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        # Create/get region in database
        region = self.db.create_region(region_name, bbox)

        results = {
            "region": region.to_dict(),
            "current_detections": [],
            "baseline_detections": [],
            "change_results": [],
            "risk_assessments": [],
            "pipeline_duration_s": 0,
        }

        # --- Phase 1: Ingest current imagery ---
        step_start = time.time()
        self.db.log_processing_step(
            step="ingestion_current", status="started",
            region_id=region.id, details={"date_range": current_date_range},
        )

        logger.info("--- Phase 1a: Ingesting current imagery ---")
        current_data = self.fetcher.ingest(
            bbox=bbox, date_range=current_date_range, limit=limit,
        )

        self.db.log_processing_step(
            step="ingestion_current", status="completed",
            region_id=region.id,
            duration_ms=int((time.time() - step_start) * 1000),
            details={"items_fetched": len(current_data)},
        )

        if not current_data:
            logger.warning("No current imagery found. Pipeline cannot continue.")
            return results

        # --- Ingest baseline imagery ---
        step_start = time.time()
        self.db.log_processing_step(
            step="ingestion_baseline", status="started",
            region_id=region.id, details={"date_range": baseline_date_range},
        )

        logger.info("--- Phase 1b: Ingesting baseline imagery ---")
        baseline_data = self.fetcher.ingest(
            bbox=bbox, date_range=baseline_date_range, limit=limit,
        )

        self.db.log_processing_step(
            step="ingestion_baseline", status="completed",
            region_id=region.id,
            duration_ms=int((time.time() - step_start) * 1000),
            details={"items_fetched": len(baseline_data)},
        )

        # --- Phase 2: Flood Detection ---
        logger.info("--- Phase 2: Running flood detection ---")

        for band_data in current_data:
            step_start = time.time()
            self.db.log_processing_step(
                step="flood_detection", status="started",
                region_id=region.id, item_id=band_data["item_id"],
            )

            try:
                flood_result = self.flood_detector.detect(band_data)
                results["current_detections"].append(flood_result)

                self.db.log_processing_step(
                    step="flood_detection", status="completed",
                    region_id=region.id, item_id=band_data["item_id"],
                    duration_ms=int((time.time() - step_start) * 1000),
                    details={
                        "risk_level": flood_result.risk_level,
                        "flood_pct": flood_result.flood_percentage,
                    },
                )
            except Exception as e:
                logger.error("Flood detection failed for %s: %s", band_data["item_id"], e)
                self.db.log_processing_step(
                    step="flood_detection", status="failed",
                    region_id=region.id, item_id=band_data["item_id"],
                    details={"error": str(e)},
                )

        for band_data in baseline_data:
            try:
                flood_result = self.flood_detector.detect(band_data)
                results["baseline_detections"].append(flood_result)
            except Exception as e:
                logger.error("Baseline detection failed for %s: %s", band_data["item_id"], e)

        # --- Phase 3: Change Detection ---
        logger.info("--- Phase 3: Running change detection ---")

        if results["current_detections"] and results["baseline_detections"]:
            for current_det in results["current_detections"]:
                baseline_det = results["baseline_detections"][0]  # Use first baseline

                step_start = time.time()
                self.db.log_processing_step(
                    step="change_detection", status="started",
                    region_id=region.id,
                    details={
                        "baseline": baseline_det.item_id,
                        "current": current_det.item_id,
                    },
                )

                try:
                    change_result = self.change_detector.detect(baseline_det, current_det)
                    results["change_results"].append(change_result)

                    # Store change event
                    self.db.store_change_event(change_result, region.id)

                    self.db.log_processing_step(
                        step="change_detection", status="completed",
                        region_id=region.id,
                        duration_ms=int((time.time() - step_start) * 1000),
                        details={
                            "change_type": change_result.change_type,
                            "water_change_pct": change_result.water_change_pct,
                        },
                    )
                except Exception as e:
                    logger.error("Change detection failed: %s", e)
                    self.db.log_processing_step(
                        step="change_detection", status="failed",
                        region_id=region.id, details={"error": str(e)},
                    )

        # --- Risk Assessment & Storage ---
        logger.info("--- Phase 4: Risk assessment and storage ---")

        for i, current_det in enumerate(results["current_detections"]):
            change_result = results["change_results"][i] if i < len(results["change_results"]) else None

            assessment = self.risk_classifier.assess(
                flood_result=current_det,
                change_result=change_result,
                region_name=region_name,
                region_id=region.id,
            )
            results["risk_assessments"].append(assessment)

            # Store in database
            self.db.store_risk_assessment(assessment, region.id)

        # Pipeline complete
        total_duration = time.time() - pipeline_start
        results["pipeline_duration_s"] = round(total_duration, 2)

        self.db.log_processing_step(
            step="pipeline_complete", status="completed",
            region_id=region.id,
            duration_ms=int(total_duration * 1000),
            details={
                "current_items": len(results["current_detections"]),
                "baseline_items": len(results["baseline_detections"]),
                "changes_detected": len(results["change_results"]),
                "assessments_stored": len(results["risk_assessments"]),
            },
        )

        logger.info(
            "=== Pipeline complete in %.1fs | %d detections | %d changes | %d assessments ===",
            total_duration,
            len(results["current_detections"]),
            len(results["change_results"]),
            len(results["risk_assessments"]),
        )

        return results

    def get_report(self, region_id: int) -> dict:
        """Get a structured summary report for a region."""
        return self.db.generate_summary_report(region_id)
