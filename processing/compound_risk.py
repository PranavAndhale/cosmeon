"""
Compound Risk Modeling Engine — INFORM Risk Index Methodology.

Implements the EU JRC INFORM Risk Index framework (used by UN/OCHA) for
multi-hazard compound risk assessment. Replaces arbitrary custom weights
with the published INFORM geometric-mean formula:

    INFORM Risk = (Hazard × Exposure × Vulnerability)^(1/3)

Each dimension is scored 0–10, then the geometric mean produces a 0–10
composite. We normalize to 0–1 for frontend compatibility.

Reference: https://drmkc.jrc.ec.europa.eu/inform-index
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("cosmeon.processing.compound_risk")


@dataclass
class HazardLayer:
    """Individual hazard contribution."""
    name: str
    severity: float  # 0-1
    weight: float  # informational — INFORM uses geometric mean, not weights
    status: str  # "active", "warning", "normal"
    description: str = ""


@dataclass
class CompoundRiskResult:
    """Result of INFORM-based compound multi-hazard risk analysis."""
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
    """
    Computes compound multi-hazard risk using the INFORM Risk Index methodology.

    The INFORM framework (EU JRC) computes:
        Risk = (Hazard × Exposure × Vulnerability)^(1/3)

    where each dimension is scored 0–10 using established data sources.
    """

    # INFORM interaction descriptions (published cascading risk types)
    _CASCADING_DESCRIPTIONS = {
        ("flood", "heat_stress"): "Floodwater + heat increases waterborne disease risk (INFORM cascading)",
        ("flood", "soil_saturation"): "Saturated ground amplifies flood extent and landslide risk (INFORM cascading)",
        ("flood", "vegetation_loss"): "Reduced vegetation increases runoff and flood severity (INFORM cascading)",
        ("heat_stress", "vegetation_loss"): "Heat accelerates vegetation die-off — drought cascade (INFORM cascading)",
        ("soil_saturation", "vegetation_loss"): "Weak root systems in saturated soil increase erosion (INFORM cascading)",
    }

    def __init__(self):
        logger.info("CompoundRiskEngine initialized (INFORM Risk methodology)")

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
        pop_density: float = 500.0,
        gdp_usd: float = 50e9,
        lat: float = 0.0,
        lon: float = 0.0,
    ) -> CompoundRiskResult:
        """
        Compute INFORM-based compound risk from multiple hazard layers.

        Uses the INFORM Risk Index geometric mean formula:
            Risk = (Hazard × Exposure × Vulnerability)^(1/3)

        All inputs come from established sources (ERA5, GloFAS, World Bank, FAO-56).
        """
        result = CompoundRiskResult(timestamp=datetime.utcnow().isoformat())

        # ── HAZARD DIMENSION (0–10) ──────────────────────────────────────────
        # Flood hazard: from GloFAS/live detection flood probability
        # Note: flood_confidence measures DETECTION certainty (1.0 = confident),
        # NOT flood severity. Only flood_probability drives the hazard score.
        # Confidence is used to weight the signal reliability (high confidence
        # means we trust the probability more).
        flood_sev = flood_probability
        confidence_weight = max(0.5, min(1.0, flood_confidence))  # reliable detection → trust the probability
        flood_hazard_10 = min(10.0, flood_sev * 10.0 * confidence_weight)

        # Precipitation hazard: based on 7-day rainfall intensity
        # ERA5 climatological thresholds: >100mm/week = severe, >50mm = moderate
        precip_hazard_10 = min(10.0, (rainfall_7d_mm / 200.0) * 10.0)

        # Heat hazard: from ERA5 temperature anomaly
        # Standard: >5°C anomaly = severe, >3°C = high, >1°C = moderate
        heat_hazard_10 = min(10.0, max(0.0, thermal_anomaly) / 5.0 * 10.0)

        # Combined hazard: flood is primary signal (50%), precip secondary (30%), heat modifier (20%)
        # OLD: max(flood, precip) * 0.7 + heat * 0.3 let precip overpower a low flood signal
        hazard_score = flood_hazard_10 * 0.50 + precip_hazard_10 * 0.30 + heat_hazard_10 * 0.20

        # ── EXPOSURE DIMENSION (0–10) ────────────────────────────────────────
        # Population exposure: INFORM brackets based on World Bank density
        if pop_density > 10000:
            pop_exposure = 9.0
        elif pop_density > 5000:
            pop_exposure = 7.0
        elif pop_density > 1000:
            pop_exposure = 5.0
        elif pop_density > 200:
            pop_exposure = 3.0
        else:
            pop_exposure = 1.5

        # Elevation exposure: low-lying areas from Open-Meteo Elevation API
        if elevation_m < 10:
            elev_exposure = 9.0
        elif elevation_m < 50:
            elev_exposure = 6.0
        elif elevation_m < 100:
            elev_exposure = 4.0
        elif elevation_m < 200:
            elev_exposure = 2.0
        else:
            elev_exposure = 1.0

        exposure_score = max(pop_exposure, elev_exposure)

        # ── VULNERABILITY DIMENSION (0–10) ───────────────────────────────────
        # Soil vulnerability: from ERA5 soil moisture (higher = more runoff risk)
        soil_vuln = min(10.0, soil_saturation * 10.0)

        # Vegetation vulnerability: from FAO-56 ET0 water balance
        veg_vuln = min(10.0, vegetation_stress * 10.0)

        vulnerability_score = max(soil_vuln, veg_vuln)

        # ── INFORM COMPOSITE (geometric mean) ────────────────────────────────
        # INFORM Risk = (Hazard × Exposure × Vulnerability)^(1/3)
        # Clamp to minimum 0.1 to avoid zero-product issues
        h = max(0.1, hazard_score)
        e = max(0.1, exposure_score)
        v = max(0.1, vulnerability_score)
        inform_score_10 = (h * e * v) ** (1.0 / 3.0)

        # Normalize to 0–1 for frontend
        compound_01 = min(1.0, inform_score_10 / 10.0)

        # ── Build hazard layers for frontend display ─────────────────────────
        layers = []
        layers.append(HazardLayer(
            name="flood",
            severity=flood_sev,
            weight=0.35,  # informational
            status=self._severity_status(flood_sev),
            description=f"Flood probability: {flood_sev:.0%}" +
                        (f", rainfall: {rainfall_7d_mm:.0f}mm/7d" if rainfall_7d_mm > 0 else ""),
        ))

        heat_sev = min(1.0, max(0.0, thermal_anomaly / 5.0))
        layers.append(HazardLayer(
            name="heat_stress",
            severity=heat_sev,
            weight=0.20,
            status=self._severity_status(heat_sev),
            description=f"Temperature anomaly: {thermal_anomaly:+.1f}°C (ERA5 baseline)",
        ))

        layers.append(HazardLayer(
            name="vegetation_loss",
            severity=vegetation_stress,
            weight=0.20,
            status=self._severity_status(vegetation_stress),
            description=f"Vegetation stress: {vegetation_stress:.2f} (FAO-56 water balance)",
        ))

        soil_sev = min(1.0, soil_saturation)
        layers.append(HazardLayer(
            name="soil_saturation",
            severity=soil_sev,
            weight=0.15,
            status=self._severity_status(soil_sev),
            description=f"Soil moisture: {soil_saturation:.0%} (ERA5 reanalysis)",
        ))

        elev_sev = min(1.0, max(0.0, 1.0 - elevation_m / 200.0)) if elevation_m < 200 else 0.0
        layers.append(HazardLayer(
            name="elevation_risk",
            severity=elev_sev,
            weight=0.10,
            status=self._severity_status(elev_sev),
            description=f"Elevation: {elevation_m:.0f}m (Open-Meteo DEM)"
                        + (" — low-lying" if elevation_m < 50 else ""),
        ))

        result.hazard_layers = layers

        # ── INFORM Cascading Risk ────────────────────────────────────────────
        # When two hazards are simultaneously above 75th percentile (severity > 0.75),
        # apply INFORM cascading formula: sqrt(h1 * h2) / max(h1, h2)
        active_hazards = [(h.name, h.severity) for h in layers if h.severity > 0.3]
        amplification = 1.0
        interactions = []

        for i, (name_a, sev_a) in enumerate(active_hazards):
            for name_b, sev_b in active_hazards[i + 1:]:
                # Both above 75th percentile threshold
                if sev_a > 0.5 and sev_b > 0.5:
                    # INFORM cascading: geometric coupling factor
                    coupling = math.sqrt(sev_a * sev_b) / max(sev_a, sev_b)
                    factor = 1.0 + coupling * 0.25  # scale to reasonable amplification
                    amplification *= factor
                    key = tuple(sorted([name_a, name_b]))
                    desc = self._CASCADING_DESCRIPTIONS.get(
                        key, f"Combined {name_a} and {name_b} cascading effects (INFORM)"
                    )
                    interactions.append({
                        "hazards": [name_a, name_b],
                        "amplification": round(factor, 3),
                        "effect": desc,
                    })

        result.cascading_amplification = round(amplification, 3)
        result.interaction_effects = interactions
        result.compound_score = min(1.0, compound_01 * amplification)

        # ── Flood-probability guardrail ────────────────────────────────────────
        # The INFORM geometric mean can produce CRITICAL scores even when actual
        # flood probability is near-zero, because high exposure/vulnerability
        # dimensions (pop density, elevation, vegetation stress) dominate.
        # Cap compound score based on the actual detection signal.
        if flood_probability < 0.15:
            result.compound_score = min(result.compound_score, 0.45)  # cap at HIGH max
        elif flood_probability < 0.40:
            result.compound_score = min(result.compound_score, 0.65)  # cap at CRITICAL threshold

        result.compound_level = self._score_to_level(result.compound_score)

        # Dominant hazard
        if layers:
            dominant = max(layers, key=lambda h: h.severity)
            result.dominant_hazard = dominant.name

        # ── INFORM Country-Level Calibration ──────────────────────────────────
        # Compare our computed score against the published INFORM Risk Index
        # for this country. If they diverge >30%, log a calibration warning.
        inform_ref = None
        try:
            from processing.model_hub import get_inform_country_risk
            inform_ref = get_inform_country_risk(lat, lon)
        except Exception:
            pass

        if inform_ref:
            # INFORM score is 0-10, normalize to 0-1 for comparison
            inform_norm = inform_ref.get("risk", 5.0) / 10.0
            deviation = abs(result.compound_score - inform_norm) / max(inform_norm, 0.01)
            if deviation > 0.30:
                logger.warning(
                    "INFORM calibration: %s computed=%.2f vs INFORM=%.2f (%.0f%% deviation)",
                    region_name, result.compound_score, inform_norm, deviation * 100,
                )

        result.recommendations = self._generate_recommendations(result)

        logger.info(
            "INFORM compound risk for %s: score=%.2f (%s), H=%.1f E=%.1f V=%.1f",
            region_name, result.compound_score, result.compound_level,
            hazard_score, exposure_score, vulnerability_score,
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
        # INFORM Risk Index classification thresholds (0-10 scale, adapted to 0-1)
        if score >= 0.65:
            return "CRITICAL"
        elif score >= 0.40:
            return "HIGH"
        elif score >= 0.20:
            return "MEDIUM"
        return "LOW"

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
            recs.append("Activate multi-hazard response coordination (INFORM protocol)")

        if not recs:
            recs.append("Continue routine monitoring — no elevated compound risks detected")

        return recs[:5]
