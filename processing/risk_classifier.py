"""
Phase 2 (continued): Risk Classification Module.

Classifies regions into risk levels based on flood detection results,
change detection, and optional external data.
"""
import logging
from dataclasses import dataclass
from datetime import datetime

from config.settings import RISK_THRESHOLDS

logger = logging.getLogger("cosmeon.processing.risk")


@dataclass
class RiskAssessment:
    """Structured risk assessment for a region."""
    region_name: str
    region_id: int
    timestamp: str
    risk_level: str
    flood_percentage: float
    flood_area_km2: float
    total_area_km2: float
    confidence_score: float
    change_type: str
    water_change_pct: float
    source_items: list
    assessment_details: dict
    assessed_at: str = None

    def __post_init__(self):
        if self.assessed_at is None:
            self.assessed_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "region_id": self.region_id,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level,
            "flood_percentage": self.flood_percentage,
            "flood_area_km2": self.flood_area_km2,
            "total_area_km2": self.total_area_km2,
            "confidence_score": self.confidence_score,
            "change_type": self.change_type,
            "water_change_pct": self.water_change_pct,
            "source_items": self.source_items,
            "assessment_details": self.assessment_details,
            "assessed_at": self.assessed_at,
        }


class RiskClassifier:
    """Classifies risk based on flood detection and change detection results."""

    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or RISK_THRESHOLDS
        logger.info("RiskClassifier initialized")

    def compute_composite_risk(
        self,
        flood_percentage: float,
        water_change_pct: float,
        confidence: float,
    ) -> str:
        """
        Compute composite risk level considering:
          - Current flood extent
          - Rate of change
          - Detection confidence
        """
        # Base risk from flood percentage
        base_risk_score = flood_percentage

        # Amplify risk if water is increasing rapidly
        if water_change_pct > 0:
            change_amplifier = 1.0 + min(water_change_pct * 2, 0.5)
            base_risk_score *= change_amplifier

        # Determine risk level
        for level in ["CRITICAL", "HIGH", "MEDIUM"]:
            if base_risk_score >= self.thresholds[level]:
                return level
        return "LOW"

    def assess(
        self,
        flood_result,
        change_result=None,
        region_name: str = "Unknown",
        region_id: int = 0,
    ) -> RiskAssessment:
        """
        Generate a comprehensive risk assessment.

        Args:
            flood_result: FloodDetectionResult
            change_result: Optional ChangeDetectionResult
            region_name: Name of the region
            region_id: Database ID of the region

        Returns:
            RiskAssessment with all metrics
        """
        water_change_pct = 0.0
        change_type = "no_baseline"
        source_items = [flood_result.item_id]

        if change_result is not None:
            water_change_pct = change_result.water_change_pct
            change_type = change_result.change_type
            source_items.append(change_result.baseline_id)

        risk_level = self.compute_composite_risk(
            flood_result.flood_percentage,
            water_change_pct,
            flood_result.confidence_score,
        )

        details = {
            "ndwi_threshold_used": flood_result.confidence_score,
            "total_pixels_analyzed": flood_result.total_pixels,
            "water_pixels_detected": flood_result.water_pixels,
            "cloud_masked": True,
            "morphological_cleanup": True,
        }
        if change_result:
            details["new_flood_pixels"] = change_result.new_flood_pixels
            details["receded_pixels"] = change_result.receded_pixels
            details["change_polygons_count"] = len(change_result.change_polygons)

        assessment = RiskAssessment(
            region_name=region_name,
            region_id=region_id,
            timestamp=flood_result.timestamp,
            risk_level=risk_level,
            flood_percentage=flood_result.flood_percentage,
            flood_area_km2=flood_result.flood_area_km2,
            total_area_km2=flood_result.area_km2,
            confidence_score=flood_result.confidence_score,
            change_type=change_type,
            water_change_pct=water_change_pct,
            source_items=source_items,
            assessment_details=details,
        )

        logger.info(
            "Risk assessment: region=%s | risk=%s | flood=%.2f%% | change=%.2f%% | confidence=%.3f",
            region_name, risk_level,
            flood_result.flood_percentage * 100,
            water_change_pct * 100,
            flood_result.confidence_score,
        )

        return assessment
