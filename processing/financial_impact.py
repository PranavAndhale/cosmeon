"""
Financial Impact Quantification Engine — JRC Depth-Damage Functions.

Translates flood risk into financial impact using published methodologies:

  - Direct damage: JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017)
    Maps estimated flood depth to damage fractions per sector.
  - Indirect costs: UNDRR Sendai Framework standard ratios by income classification
  - Displacement: UNHCR per-capita displacement cost brackets
  - GDP data: World Bank Open Data (passed from model_hub)
  - Mitigation ROI: World Bank/GFDRR published cost-benefit ranges

References:
  - Huizinga, J., de Moel, H., Szewczyk, W. (2017). Global flood depth-damage
    functions. JRC Technical Report. doi:10.2760/16510
  - UNDRR (2015). Sendai Framework for Disaster Risk Reduction 2015–2030.
  - World Bank/GFDRR (2017). Unbreakable: Building the Resilience of the Poor.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("cosmeon.processing.financial")


@dataclass
class FinancialImpact:
    """Financial impact assessment for a region."""
    direct_damage_usd: float = 0.0
    indirect_costs_usd: float = 0.0
    total_impact_usd: float = 0.0
    insurance_exposure_usd: float = 0.0
    recovery_cost_usd: float = 0.0
    gdp_impact_pct: float = 0.0
    affected_population: int = 0
    displacement_cost_usd: float = 0.0
    breakdown: dict = field(default_factory=dict)
    mitigation_roi: list = field(default_factory=list)
    confidence: str = "medium"
    timestamp: str = ""

    def to_dict(self):
        return {
            "direct_damage_usd": round(self.direct_damage_usd, 0),
            "indirect_costs_usd": round(self.indirect_costs_usd, 0),
            "total_impact_usd": round(self.total_impact_usd, 0),
            "insurance_exposure_usd": round(self.insurance_exposure_usd, 0),
            "recovery_cost_usd": round(self.recovery_cost_usd, 0),
            "gdp_impact_pct": round(self.gdp_impact_pct, 3),
            "affected_population": self.affected_population,
            "displacement_cost_usd": round(self.displacement_cost_usd, 0),
            "breakdown": self.breakdown,
            "mitigation_roi": self.mitigation_roi,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017)
# ──────────────────────────────────────────────────────────────────────────────
# Damage fraction (0–1) at each depth (meters) per sector.
# These are piecewise-linear interpolation tables from the JRC publication.
# Depths: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] meters
# Values: fraction of asset value damaged at that depth

_JRC_DEPTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

_JRC_DAMAGE_FRACTIONS = {
    # Residential buildings (Table 2, Huizinga et al. 2017)
    "residential":    [0.00, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80],
    # Commercial buildings
    "commercial":     [0.00, 0.15, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65],
    # Infrastructure (roads, utilities, public)
    "infrastructure": [0.00, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50],
    # Agriculture (crops)
    "agriculture":    [0.00, 0.30, 0.55, 0.70, 0.80, 0.85, 0.90, 0.95],
}

# UNDRR Sendai Framework indirect cost ratios by World Bank income classification
_SENDAI_INDIRECT_RATIOS = {
    "HIC": 0.5,    # High-income countries
    "UMC": 0.8,    # Upper-middle income
    "LMC": 1.0,    # Lower-middle income
    "LIC": 1.2,    # Low-income countries
}

# UNHCR displacement cost per person per day (USD) by income bracket
_UNHCR_DISPLACEMENT_COST = {
    "HIC": 85.0,    # High-income: higher services cost
    "UMC": 45.0,    # Upper-middle income
    "LMC": 25.0,    # Lower-middle income
    "LIC": 12.0,    # Low-income: basic needs cost
}

# Average displacement duration (days) from UNDRR Sendai data
_DISPLACEMENT_DAYS = {
    "HIC": 10, "UMC": 14, "LMC": 21, "LIC": 30,
}

# GloFAS discharge anomaly → approximate flood inundation depth mapping
# Based on ECMWF GloFAS operational documentation (return period calibration)
_ANOMALY_TO_DEPTH = [
    (0.0, 0.0),   # no anomaly → no inundation
    (1.0, 0.2),   # ~2-year return period
    (1.5, 0.5),   # ~5-year return period
    (2.0, 0.8),   # ~5-10 year
    (2.5, 1.2),   # ~10-year
    (3.0, 1.5),   # ~20-year
    (4.0, 2.5),   # ~50-year
    (5.0, 4.0),   # ~100-year+
]


def _interpolate_depth(anomaly_sigma: float) -> float:
    """Map GloFAS discharge anomaly (sigma) to estimated flood depth (meters)."""
    if anomaly_sigma <= 0:
        return 0.0
    for i in range(len(_ANOMALY_TO_DEPTH) - 1):
        a0, d0 = _ANOMALY_TO_DEPTH[i]
        a1, d1 = _ANOMALY_TO_DEPTH[i + 1]
        if a0 <= anomaly_sigma <= a1:
            frac = (anomaly_sigma - a0) / (a1 - a0)
            return d0 + frac * (d1 - d0)
    return _ANOMALY_TO_DEPTH[-1][1]  # cap at max


def _jrc_damage_fraction(depth_m: float, sector: str) -> float:
    """
    JRC depth-damage function: interpolate damage fraction for given depth and sector.
    Returns fraction 0–1 of asset value damaged.
    """
    fractions = _JRC_DAMAGE_FRACTIONS.get(sector, _JRC_DAMAGE_FRACTIONS["residential"])
    if depth_m <= 0:
        return 0.0
    for i in range(len(_JRC_DEPTHS) - 1):
        d0 = _JRC_DEPTHS[i]
        d1 = _JRC_DEPTHS[i + 1]
        if d0 <= depth_m <= d1:
            f0 = fractions[i]
            f1 = fractions[i + 1]
            frac = (depth_m - d0) / (d1 - d0)
            return f0 + frac * (f1 - f0)
    return fractions[-1]  # cap at max depth


def _classify_income(gdp_per_capita: float) -> str:
    """
    World Bank income classification from GDP per capita.
    Thresholds from World Bank Atlas Method (FY2024):
      HIC: > $14,005
      UMC: $4,516 - $14,005
      LMC: $1,146 - $4,515
      LIC: < $1,146
    """
    if gdp_per_capita > 14005:
        return "HIC"
    elif gdp_per_capita > 4516:
        return "UMC"
    elif gdp_per_capita > 1146:
        return "LMC"
    return "LIC"


class FinancialImpactEngine:
    """
    Quantifies financial impact using JRC depth-damage functions
    and UNDRR Sendai Framework ratios.
    """

    def __init__(self):
        logger.info("FinancialImpactEngine initialized (JRC depth-damage + UNDRR Sendai)")

    def estimate_impact(
        self,
        risk_level: str = "MEDIUM",
        flood_area_km2: float = 0.0,
        total_area_km2: float = 0.0,
        flood_probability: float = 0.3,
        population_density: float = 500.0,
        asset_scores: dict = None,
        region_name: str = "Unknown",
        gdp_usd: float = 0.0,
        discharge_anomaly: float = 0.0,
    ) -> FinancialImpact:
        """
        Estimate financial impact using JRC depth-damage functions.

        Parameters:
            gdp_usd: Country/region GDP from World Bank (model hub T1)
            discharge_anomaly: GloFAS discharge anomaly (sigma) for depth estimation
            population_density: From World Bank (model hub)
            flood_area_km2: Estimated flood area from GloFAS/detection
        """
        result = FinancialImpact(timestamp=datetime.utcnow().isoformat())

        # Estimate flood depth from GloFAS discharge anomaly
        flood_depth_m = _interpolate_depth(discharge_anomaly)

        # If no discharge data, estimate from risk level
        if flood_depth_m <= 0 and risk_level != "LOW":
            depth_by_risk = {"LOW": 0.0, "MEDIUM": 0.3, "HIGH": 0.8, "CRITICAL": 1.5}
            flood_depth_m = depth_by_risk.get(risk_level, 0.3)

        # ── Risk-level dampening ───────────────────────────────────────────────
        # When detection says LOW or MEDIUM risk and GloFAS shows no elevated
        # discharge, dampen the financial impact to prevent unrealistic numbers.
        # Bihar example: large area × high pop × 10% prob was producing $41M
        # despite GloFAS showing normal discharge.
        risk_dampening = 1.0
        if risk_level == "LOW":
            risk_dampening = 0.05 if discharge_anomaly < 1.0 else 0.20
        elif risk_level == "MEDIUM":
            risk_dampening = 0.30 if discharge_anomaly < 1.0 else 0.70
        # HIGH/CRITICAL: no dampening (risk_dampening stays 1.0)

        # ── Cap flood area for realism ─────────────────────────────────────────
        # Bounding box area can be enormous (Bihar = 166,000 km²).
        # Even worst floods rarely inundate >5% of a region.
        # For LOW risk with no discharge anomaly, cap much more aggressively.
        if total_area_km2 > 0:
            flood_area_km2 = min(flood_area_km2, total_area_km2 * 0.05)
        if risk_level == "LOW" and discharge_anomaly < 1.0:
            flood_area_km2 = min(flood_area_km2, 50.0)  # max 50 km² for LOW risk
        elif risk_level == "MEDIUM" and discharge_anomaly < 1.0:
            flood_area_km2 = min(flood_area_km2, 200.0)  # max 200 km² for MEDIUM

        # JRC damage fractions per sector
        residential_frac = _jrc_damage_fraction(flood_depth_m, "residential")
        commercial_frac = _jrc_damage_fraction(flood_depth_m, "commercial")
        infra_frac = _jrc_damage_fraction(flood_depth_m, "infrastructure")
        agri_frac = _jrc_damage_fraction(flood_depth_m, "agriculture")

        # Estimate exposed asset values from GDP and area
        # Fixed capital formation typically 20-25% of GDP, distributed by density
        if gdp_usd > 0:
            # Capital stock per km² ≈ GDP × 0.22 (fixed capital ratio) / total_area
            effective_area = max(total_area_km2, 100.0)
            capital_per_km2 = gdp_usd * 0.22 / effective_area
        else:
            capital_per_km2 = 500_000  # fallback: $500K/km²

        exposed_capital = flood_area_km2 * capital_per_km2 * flood_probability * risk_dampening

        # Sector breakdown using JRC damage fractions
        infra_damage = exposed_capital * 0.35 * infra_frac
        resi_damage = exposed_capital * 0.30 * residential_frac
        agri_damage = exposed_capital * 0.20 * agri_frac
        comm_damage = exposed_capital * 0.15 * commercial_frac

        result.direct_damage_usd = infra_damage + resi_damage + agri_damage + comm_damage
        result.breakdown = {
            "infrastructure": round(infra_damage, 0),
            "residential": round(resi_damage, 0),
            "agriculture": round(agri_damage, 0),
            "commercial": round(comm_damage, 0),
        }

        # Income classification for UNDRR Sendai ratios
        # Estimate GDP per capita from World Bank data
        gdp_per_capita = gdp_usd / max(population_density * total_area_km2, 1e6) if gdp_usd > 0 else 3000
        income_class = _classify_income(gdp_per_capita)

        # Indirect costs: UNDRR Sendai Framework ratios
        indirect_ratio = _SENDAI_INDIRECT_RATIOS.get(income_class, 0.8)
        result.indirect_costs_usd = result.direct_damage_usd * indirect_ratio

        # Population impact (with risk dampening applied)
        result.affected_population = int(
            flood_area_km2 * population_density * flood_probability * risk_dampening
        )

        # Displacement: UNHCR per-capita costs
        daily_cost = _UNHCR_DISPLACEMENT_COST.get(income_class, 25.0)
        duration = _DISPLACEMENT_DAYS.get(income_class, 14)
        result.displacement_cost_usd = result.affected_population * daily_cost * duration

        # Recovery cost: UNDRR typically 1.3–1.8× direct damage
        recovery_multiplier = {"HIC": 1.3, "UMC": 1.5, "LMC": 1.6, "LIC": 1.8}
        result.recovery_cost_usd = result.direct_damage_usd * recovery_multiplier.get(income_class, 1.5)

        # Insurance exposure: varies by income
        insurance_coverage = {"HIC": 0.55, "UMC": 0.30, "LMC": 0.15, "LIC": 0.05}
        result.insurance_exposure_usd = result.direct_damage_usd * insurance_coverage.get(income_class, 0.20)

        # Total impact
        result.total_impact_usd = (
            result.direct_damage_usd
            + result.indirect_costs_usd
            + result.displacement_cost_usd
        )

        # GDP impact
        if gdp_usd > 0:
            result.gdp_impact_pct = (result.total_impact_usd / gdp_usd) * 100
        else:
            result.gdp_impact_pct = 0.0

        # Mitigation ROI: World Bank/GFDRR published ranges
        result.mitigation_roi = self._compute_mitigation_roi(result, risk_level, income_class)

        # Confidence
        if gdp_usd > 0 and discharge_anomaly > 0 and population_density > 0:
            result.confidence = "high"
        elif flood_area_km2 > 0:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        logger.info(
            "JRC financial impact for %s: total=$%.0f, depth=%.1fm, class=%s, affected=%d",
            region_name, result.total_impact_usd, flood_depth_m,
            income_class, result.affected_population,
        )
        return result

    def _compute_mitigation_roi(
        self, impact: FinancialImpact, risk_level: str, income_class: str,
    ) -> list:
        """
        Compute ROI using World Bank/GFDRR published cost-benefit ranges.

        Reference: World Bank (2017) "Unbreakable" and GFDRR cost-benefit analyses.
        """
        measures = []
        base_damage = impact.direct_damage_usd

        if base_damage <= 0:
            return [{"measure": "No mitigation needed", "cost": 0, "savings": 0, "roi_pct": 0}]

        # Early warning systems: World Bank/GFDRR reports 3–16× ROI
        # Cost ~2% of damage, saves ~15-25%
        ews_cost = min(base_damage * 0.02, 500_000)
        ews_savings = base_damage * 0.20
        measures.append({
            "measure": "Early Warning System (GFDRR recommended)",
            "cost": round(ews_cost, 0),
            "savings": round(ews_savings, 0),
            "roi_pct": round((ews_savings - ews_cost) / max(ews_cost, 1) * 100, 0),
        })

        # Flood barriers: JRC estimates 2–10× ROI
        barrier_cost = min(base_damage * 0.10, 2_000_000)
        barrier_savings = base_damage * 0.40
        measures.append({
            "measure": "Flood Barriers & Levees (JRC validated)",
            "cost": round(barrier_cost, 0),
            "savings": round(barrier_savings, 0),
            "roi_pct": round((barrier_savings - barrier_cost) / max(barrier_cost, 1) * 100, 0),
        })

        # Drainage: GFDRR reports 2–8× ROI
        drain_cost = min(base_damage * 0.08, 1_500_000)
        drain_savings = base_damage * 0.25
        measures.append({
            "measure": "Drainage Infrastructure (GFDRR)",
            "cost": round(drain_cost, 0),
            "savings": round(drain_savings, 0),
            "roi_pct": round((drain_savings - drain_cost) / max(drain_cost, 1) * 100, 0),
        })

        # Nature-based solutions: World Bank reports 2–10× ROI
        if risk_level in ("HIGH", "CRITICAL"):
            nbs_cost = min(base_damage * 0.15, 3_000_000)
            nbs_savings = base_damage * 0.35
            measures.append({
                "measure": "Nature-Based Solutions (World Bank NBS)",
                "cost": round(nbs_cost, 0),
                "savings": round(nbs_savings, 0),
                "roi_pct": round((nbs_savings - nbs_cost) / max(nbs_cost, 1) * 100, 0),
            })

        return measures
