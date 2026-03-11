"""
Phase 2B: Compound Risk Modeling Engine.

Combines multiple hazard layers (flood, heat, vegetation loss, soil saturation)
into a single compound risk score. Uses weighted overlay analysis with
configurable thresholds and cascading risk amplification.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cosmeon.processing.compound_risk")


@dataclass
class HazardLayer:
    """Individual hazard contribution."""
    name: str
    severity: float  # 0-1
    weight: float  # contribution weight
    status: str  # "active", "warning", "normal"
    description: str = ""


@dataclass
class CompoundRiskResult:
    """Result of compound multi-hazard risk analysis."""
    compound_score: float = 0.0  # 0-1 overall
    compound_level: str = "LOW"
    hazard_layers: list = field(default_factory=list)
    cascading_amplification: float = 1.0
    dominant_hazard: str = ""
    interaction_effects: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self):
        return {
            "compound_score": round(self.compound_score, 3),
            "compound_level": self.compound_level,
            "hazard_layers": [
                {"name": h.name, "severity": round(h.severity, 3),
                 "weight": h.weight, "status": h.status, "description": h.description}
                for h in self.hazard_layers
            ],
            "cascading_amplification": round(self.cascading_amplification, 2),
            "dominant_hazard": self.dominant_hazard,
            "interaction_effects": self.interaction_effects,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class CompoundRiskEngine:
    """Computes compound multi-hazard risk scores."""

    # Hazard interaction matrix — when two hazards co-occur,
    # risk amplifies by this factor
    INTERACTION_MATRIX = {
        ("flood", "heat_stress"): 1.15,  # Heat + flood = water contamination risk
        ("flood", "soil_saturation"): 1.25,  # Saturated soil + flood = landslide risk
        ("flood", "vegetation_loss"): 1.20,  # No root structure = worse flooding
        ("heat_stress", "vegetation_loss"): 1.10,  # Drought cascade
        ("soil_saturation", "vegetation_loss"): 1.15,  # Erosion risk
    }

    def __init__(self):
        logger.info("CompoundRiskEngine initialized")

    def compute_compound_risk(
        self,
        flood_probability: float = 0.0,
        flood_confidence: float = 0.0,
        vegetation_stress: float = 0.0,
        thermal_anomaly: float = 0.0,
        soil_saturation: float = 0.0,
        rainfall_7d_mm: float = 0.0,
        elevation_m: float = 100.0,
        region_name: str = "Unknown",
    ) -> CompoundRiskResult:
        """
        Compute compound risk from multiple hazard layers.

        Returns a CompoundRiskResult with overall score, per-hazard breakdown,
        interaction effects, and actionable recommendations.
        """
        result = CompoundRiskResult(timestamp=datetime.utcnow().isoformat())

        # Build hazard layers
        layers = []

        # 1. Flood hazard
        flood_sev = max(flood_probability, flood_confidence)
        layers.append(HazardLayer(
            name="flood",
            severity=flood_sev,
            weight=0.35,
            status=self._severity_status(flood_sev),
            description=f"Flood probability: {flood_sev:.0%}" +
                        (f", rainfall: {rainfall_7d_mm:.0f}mm/7d" if rainfall_7d_mm > 0 else ""),
        ))

        # 2. Heat stress hazard
        heat_sev = min(1.0, max(0.0, thermal_anomaly / 15.0))  # 15°C above normal = max severity
        layers.append(HazardLayer(
            name="heat_stress",
            severity=heat_sev,
            weight=0.20,
            status=self._severity_status(heat_sev),
            description=f"Temperature anomaly: {thermal_anomaly:+.1f}°C",
        ))

        # 3. Vegetation loss hazard
        veg_sev = vegetation_stress
        layers.append(HazardLayer(
            name="vegetation_loss",
            severity=veg_sev,
            weight=0.20,
            status=self._severity_status(veg_sev),
            description=f"Vegetation stress index: {veg_sev:.2f}",
        ))

        # 4. Soil saturation hazard
        soil_sev = min(1.0, soil_saturation / 0.5) if soil_saturation > 0.2 else 0.0
        layers.append(HazardLayer(
            name="soil_saturation",
            severity=soil_sev,
            weight=0.15,
            status=self._severity_status(soil_sev),
            description=f"Soil moisture: {soil_saturation:.0%}",
        ))

        # 5. Elevation risk factor
        elev_sev = max(0.0, 1.0 - elevation_m / 200.0) if elevation_m < 200 else 0.0
        layers.append(HazardLayer(
            name="elevation_risk",
            severity=elev_sev,
            weight=0.10,
            status=self._severity_status(elev_sev),
            description=f"Elevation: {elevation_m:.0f}m" + (" (low-lying)" if elevation_m < 50 else ""),
        ))

        result.hazard_layers = layers

        # Compute base weighted score
        base_score = sum(h.severity * h.weight for h in layers)

        # Check interaction effects and compute cascading amplification
        active_hazards = [(h.name, h.severity) for h in layers if h.severity > 0.3]
        amplification = 1.0
        interactions = []

        for i, (name_a, sev_a) in enumerate(active_hazards):
            for name_b, sev_b in active_hazards[i + 1:]:
                key = tuple(sorted([name_a, name_b]))
                if key in self.INTERACTION_MATRIX:
                    factor = self.INTERACTION_MATRIX[key]
                    # Scale by combined severity
                    combined = (sev_a + sev_b) / 2
                    effective_factor = 1.0 + (factor - 1.0) * combined
                    amplification *= effective_factor
                    interactions.append({
                        "hazards": [name_a, name_b],
                        "amplification": round(effective_factor, 3),
                        "effect": self._describe_interaction(name_a, name_b),
                    })

        result.cascading_amplification = round(amplification, 3)
        result.interaction_effects = interactions

        # Final compound score
        result.compound_score = min(1.0, base_score * amplification)
        result.compound_level = self._score_to_level(result.compound_score)

        # Dominant hazard
        if layers:
            dominant = max(layers, key=lambda h: h.severity * h.weight)
            result.dominant_hazard = dominant.name

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        logger.info(
            "Compound risk for %s: score=%.2f (%s), dominant=%s, amplification=%.2f",
            region_name, result.compound_score, result.compound_level,
            result.dominant_hazard, result.cascading_amplification
        )
        return result

    @staticmethod
    def _severity_status(severity: float) -> str:
        if severity >= 0.7:
            return "active"
        elif severity >= 0.3:
            return "warning"
        return "normal"

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 0.7:
            return "CRITICAL"
        elif score >= 0.45:
            return "HIGH"
        elif score >= 0.2:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _describe_interaction(a: str, b: str) -> str:
        descriptions = {
            ("flood", "heat_stress"): "Floodwater + heat increases waterborne disease risk",
            ("flood", "soil_saturation"): "Saturated ground amplifies flood extent and landslide risk",
            ("flood", "vegetation_loss"): "Reduced vegetation increases runoff and flood severity",
            ("heat_stress", "vegetation_loss"): "Heat accelerates vegetation die-off (drought cascade)",
            ("soil_saturation", "vegetation_loss"): "Weak root systems in saturated soil increase erosion",
        }
        key = tuple(sorted([a, b]))
        return descriptions.get(key, f"Combined {a} and {b} effects")

    def _generate_recommendations(self, result: CompoundRiskResult) -> list:
        recs = []
        active = {h.name: h.severity for h in result.hazard_layers if h.severity > 0.3}

        if "flood" in active:
            if active["flood"] > 0.7:
                recs.append("URGENT: Activate flood emergency protocols and evacuation plans")
            else:
                recs.append("Monitor river gauges and drainage infrastructure closely")

        if "heat_stress" in active:
            recs.append("Issue heat advisory; ensure cooling centers are operational")

        if "soil_saturation" in active and "flood" in active:
            recs.append("HIGH: Landslide risk elevated — restrict access to steep terrain")

        if "vegetation_loss" in active:
            recs.append("Assess erosion-prone areas and reinforce embankments")

        if result.compound_score > 0.6:
            recs.append("Activate multi-hazard response coordination")

        if not recs:
            recs.append("Continue routine monitoring — no elevated risks detected")

        return recs[:5]
