"""
Phase 3B: Financial Impact Quantification Engine.

Translates flood risk scores into financial impact estimates including:
  - Direct damage costs (infrastructure, property, agriculture)
  - Indirect costs (business interruption, displacement, recovery)
  - Insurance exposure estimates
  - Cost-benefit analysis for mitigation measures
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


# Average damage costs per km² by risk level (USD)
DAMAGE_PER_KM2 = {
    "LOW": 5_000,
    "MEDIUM": 50_000,
    "HIGH": 250_000,
    "CRITICAL": 1_000_000,
}

# Indirect cost multipliers
INDIRECT_MULTIPLIER = {
    "LOW": 0.3,
    "MEDIUM": 0.5,
    "HIGH": 0.8,
    "CRITICAL": 1.2,
}

# Metro-area GDP estimates (USD) — used for gdp_impact_pct.
# Sources: World Bank, IMF city-level estimates, Wikipedia metro GDPs.
_CITY_GDP_USD = {
    # South Asia
    "navi mumbai":  110_000_000_000,
    "mumbai":       370_000_000_000,
    "kolkata":      150_000_000_000,
    "dhaka":         80_000_000_000,
    "assam":         25_000_000_000,
    "bihar":         20_000_000_000,
    "sylhet":        12_000_000_000,
    # SE Asia
    "jakarta":      180_000_000_000,
    "bangkok":      190_000_000_000,
    "ho chi minh":   85_000_000_000,
    "manila":       120_000_000_000,
    # Europe
    "rotterdam":     80_000_000_000,
    "bremen":        35_000_000_000,
    "budapest":      55_000_000_000,
    "venice":        30_000_000_000,
    # Americas
    "são paulo":    430_000_000_000,
    "sao paulo":    430_000_000_000,
    "manaus":        25_000_000_000,
    "new orleans":   60_000_000_000,
    "houston":      530_000_000_000,
    # East Asia
    "wuhan":        230_000_000_000,
    "chongqing":    350_000_000_000,
    # Africa
    "khartoum":      22_000_000_000,
    "lagos":         90_000_000_000,
}

# Urban population density estimates (people / km²)
_CITY_POP_DENSITY = {
    "navi mumbai":   9_000,
    "mumbai":       20_000,
    "kolkata":      24_000,
    "dhaka":        44_000,
    "assam":           400,
    "bihar":           900,
    "sylhet":        2_500,
    "jakarta":      15_000,
    "bangkok":       5_600,
    "ho chi minh":   4_200,
    "manila":       46_000,
    "rotterdam":     3_000,
    "bremen":        1_500,
    "budapest":      3_300,
    "venice":          700,
    "são paulo":     7_800,
    "sao paulo":     7_800,
    "manaus":          180,
    "new orleans":     600,
    "houston":       1_400,
    "wuhan":         1_500,
    "chongqing":       400,
    "khartoum":      1_800,
    "lagos":         6_800,
}

_DEFAULT_GDP_USD        = 50_000_000_000   # $50B fallback
_DEFAULT_POP_DENSITY    = 500              # people/km² fallback


def _city_gdp(region_name: str) -> float:
    """Return estimated metro GDP in USD for the region."""
    name = region_name.lower()
    for key, gdp in _CITY_GDP_USD.items():
        if key in name:
            return gdp
    return _DEFAULT_GDP_USD


def _city_pop_density(region_name: str) -> float:
    """Return estimated urban population density (people/km²) for the region."""
    name = region_name.lower()
    for key, density in _CITY_POP_DENSITY.items():
        if key in name:
            return density
    return _DEFAULT_POP_DENSITY


class FinancialImpactEngine:
    """Quantifies financial impact of flood events."""

    def __init__(self):
        logger.info("FinancialImpactEngine initialized")

    def estimate_impact(
        self,
        risk_level: str = "MEDIUM",
        flood_area_km2: float = 0.0,
        total_area_km2: float = 0.0,
        flood_probability: float = 0.3,
        population_density: float = 0.0,
        asset_scores: dict = None,
        region_name: str = "Unknown",
    ) -> FinancialImpact:
        """Estimate comprehensive financial impact of a flood event."""
        result = FinancialImpact(timestamp=datetime.utcnow().isoformat())

        # Direct damage from flood area
        per_km2 = DAMAGE_PER_KM2.get(risk_level, 50_000)
        result.direct_damage_usd = flood_area_km2 * per_km2 * flood_probability

        # Breakdown by category
        breakdown = {
            "infrastructure": round(result.direct_damage_usd * 0.35, 0),
            "residential": round(result.direct_damage_usd * 0.25, 0),
            "agriculture": round(result.direct_damage_usd * 0.20, 0),
            "commercial": round(result.direct_damage_usd * 0.15, 0),
            "public_services": round(result.direct_damage_usd * 0.05, 0),
        }
        result.breakdown = breakdown

        # Indirect costs
        multiplier = INDIRECT_MULTIPLIER.get(risk_level, 0.5)
        result.indirect_costs_usd = result.direct_damage_usd * multiplier

        # Asset-based damage (if available)
        if asset_scores and "total_exposure_usd" in asset_scores:
            asset_damage = asset_scores["total_exposure_usd"]
            result.direct_damage_usd = max(result.direct_damage_usd, asset_damage)

        # Population impact — use region-aware density if caller didn't supply one
        effective_density = population_density if population_density > 0 else _city_pop_density(region_name)
        result.affected_population = int(flood_area_km2 * effective_density * flood_probability)

        # Displacement costs ($50/person/day × 14 days average)
        result.displacement_cost_usd = result.affected_population * 50 * 14

        # Recovery cost (1.5x direct damage for rebuilding)
        result.recovery_cost_usd = result.direct_damage_usd * 1.5

        # Insurance exposure (typically 40-60% of direct damage is insured)
        result.insurance_exposure_usd = result.direct_damage_usd * 0.45

        # Total impact
        result.total_impact_usd = (
            result.direct_damage_usd +
            result.indirect_costs_usd +
            result.displacement_cost_usd
        )

        # GDP impact — use region-aware metro GDP (not a global flat $500M)
        regional_gdp = _city_gdp(region_name)
        result.gdp_impact_pct = (result.total_impact_usd / regional_gdp) * 100 if regional_gdp > 0 else 0

        # Mitigation ROI
        result.mitigation_roi = self._compute_mitigation_roi(result, risk_level)

        # Confidence level
        if asset_scores and population_density > 0:
            result.confidence = "high"
        elif flood_area_km2 > 0:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        logger.info(
            "Financial impact for %s: total=$%.0f, affected=%d, level=%s",
            region_name, result.total_impact_usd, result.affected_population, risk_level
        )
        return result

    def _compute_mitigation_roi(self, impact: FinancialImpact, risk_level: str) -> list:
        """Compute ROI for various mitigation strategies."""
        measures = []
        base_damage = impact.direct_damage_usd

        if base_damage <= 0:
            return [{"measure": "No mitigation needed", "cost": 0, "savings": 0, "roi_pct": 0}]

        # Early warning system
        ews_cost = min(base_damage * 0.02, 500_000)
        ews_savings = base_damage * 0.15
        measures.append({
            "measure": "Early Warning System",
            "cost": round(ews_cost, 0),
            "savings": round(ews_savings, 0),
            "roi_pct": round((ews_savings - ews_cost) / max(ews_cost, 1) * 100, 0),
        })

        # Flood barriers
        barrier_cost = min(base_damage * 0.10, 2_000_000)
        barrier_savings = base_damage * 0.40
        measures.append({
            "measure": "Flood Barriers & Levees",
            "cost": round(barrier_cost, 0),
            "savings": round(barrier_savings, 0),
            "roi_pct": round((barrier_savings - barrier_cost) / max(barrier_cost, 1) * 100, 0),
        })

        # Drainage infrastructure
        drain_cost = min(base_damage * 0.08, 1_500_000)
        drain_savings = base_damage * 0.25
        measures.append({
            "measure": "Drainage Infrastructure",
            "cost": round(drain_cost, 0),
            "savings": round(drain_savings, 0),
            "roi_pct": round((drain_savings - drain_cost) / max(drain_cost, 1) * 100, 0),
        })

        # Relocation of critical assets
        if risk_level in ("HIGH", "CRITICAL"):
            reloc_cost = min(base_damage * 0.20, 5_000_000)
            reloc_savings = base_damage * 0.60
            measures.append({
                "measure": "Critical Asset Relocation",
                "cost": round(reloc_cost, 0),
                "savings": round(reloc_savings, 0),
                "roi_pct": round((reloc_savings - reloc_cost) / max(reloc_cost, 1) * 100, 0),
            })

        return measures
