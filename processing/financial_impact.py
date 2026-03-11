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

        # Population impact
        if population_density > 0:
            result.affected_population = int(flood_area_km2 * population_density)
        else:
            # Estimate from area (global avg ~50 people/km²)
            result.affected_population = int(flood_area_km2 * 50 * flood_probability)

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

        # GDP impact (rough estimate — $500M regional GDP baseline)
        regional_gdp = 500_000_000  # Can be made configurable
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
