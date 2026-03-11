"""
Phase 3A: Asset-Level Risk Scoring Engine.

Scores individual assets (buildings, infrastructure, farmland, etc.)
based on their proximity to flood zones, elevation, and asset type
vulnerability. Produces per-asset risk scores and priority rankings.
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cosmeon.processing.asset_scorer")


@dataclass
class Asset:
    """An asset to be risk-scored."""
    name: str
    lat: float
    lon: float
    asset_type: str  # "residential", "commercial", "industrial", "agricultural", "infrastructure", "hospital"
    value_usd: float = 0.0
    elevation_m: float = 0.0
    population: int = 0

    def to_dict(self):
        return {
            "name": self.name, "lat": self.lat, "lon": self.lon,
            "asset_type": self.asset_type, "value_usd": self.value_usd,
            "elevation_m": self.elevation_m, "population": self.population,
        }


@dataclass
class AssetRiskScore:
    """Risk score for an individual asset."""
    asset: dict = field(default_factory=dict)
    exposure_score: float = 0.0  # 0-1, proximity to flood zone
    vulnerability_score: float = 0.0  # 0-1, asset type vulnerability
    adaptive_capacity: float = 0.5  # 0-1, ability to recover
    overall_risk_score: float = 0.0  # 0-1
    risk_level: str = "LOW"
    estimated_damage_usd: float = 0.0
    priority_rank: int = 0
    factors: list = field(default_factory=list)

    def to_dict(self):
        return {
            "asset": self.asset,
            "exposure_score": round(self.exposure_score, 3),
            "vulnerability_score": round(self.vulnerability_score, 3),
            "adaptive_capacity": round(self.adaptive_capacity, 3),
            "overall_risk_score": round(self.overall_risk_score, 3),
            "risk_level": self.risk_level,
            "estimated_damage_usd": round(self.estimated_damage_usd, 0),
            "priority_rank": self.priority_rank,
            "factors": self.factors,
        }


# Type vulnerability multipliers
TYPE_VULNERABILITY = {
    "hospital": 0.95,  # Critical infrastructure, highly vulnerable
    "residential": 0.75,
    "commercial": 0.65,
    "industrial": 0.60,
    "agricultural": 0.80,  # Crops easily destroyed
    "infrastructure": 0.70,  # Roads, bridges
    "school": 0.85,
    "warehouse": 0.50,
}

# Damage ratio by flood depth (simplified)
DAMAGE_RATIO = {
    "LOW": 0.02,
    "MEDIUM": 0.10,
    "HIGH": 0.30,
    "CRITICAL": 0.55,
}


class AssetRiskScorer:
    """Scores individual assets for flood risk exposure."""

    def __init__(self):
        logger.info("AssetRiskScorer initialized")

    def score_assets(
        self,
        assets: list[dict],
        flood_center_lat: float,
        flood_center_lon: float,
        flood_radius_km: float = 10.0,
        flood_risk_level: str = "MEDIUM",
        flood_probability: float = 0.3,
    ) -> dict:
        """
        Score a list of assets against current flood risk.

        Returns ranked list of asset risk scores with damage estimates.
        """
        scored = []

        for asset_data in assets:
            asset = Asset(
                name=asset_data.get("name", "Unknown"),
                lat=asset_data.get("lat", 0),
                lon=asset_data.get("lon", 0),
                asset_type=asset_data.get("asset_type", "residential"),
                value_usd=asset_data.get("value_usd", 100000),
                elevation_m=asset_data.get("elevation_m", 50),
                population=asset_data.get("population", 0),
            )

            score = self._score_single_asset(
                asset, flood_center_lat, flood_center_lon,
                flood_radius_km, flood_risk_level, flood_probability
            )
            scored.append(score)

        # Sort by risk and assign ranks
        scored.sort(key=lambda s: s.overall_risk_score, reverse=True)
        for i, s in enumerate(scored):
            s.priority_rank = i + 1

        total_exposure = sum(s.estimated_damage_usd for s in scored)
        critical_count = sum(1 for s in scored if s.risk_level in ("HIGH", "CRITICAL"))

        return {
            "scored_assets": [s.to_dict() for s in scored],
            "total_assets": len(scored),
            "total_exposure_usd": round(total_exposure, 0),
            "critical_assets": critical_count,
            "assessment_time": datetime.utcnow().isoformat(),
        }

    def _score_single_asset(
        self, asset: Asset,
        flood_lat: float, flood_lon: float,
        flood_radius_km: float, risk_level: str, probability: float,
    ) -> AssetRiskScore:
        """Score a single asset."""
        result = AssetRiskScore(asset=asset.to_dict())
        factors = []

        # Exposure: distance to flood center
        distance_km = self._haversine(asset.lat, asset.lon, flood_lat, flood_lon)
        if flood_radius_km > 0:
            distance_ratio = distance_km / flood_radius_km
            result.exposure_score = max(0.0, 1.0 - distance_ratio)
        else:
            result.exposure_score = 0.5

        if result.exposure_score > 0.7:
            factors.append(f"Within primary flood zone ({distance_km:.1f}km from center)")
        elif result.exposure_score > 0.3:
            factors.append(f"Near flood periphery ({distance_km:.1f}km)")

        # Elevation penalty
        if asset.elevation_m < 10:
            result.exposure_score = min(1.0, result.exposure_score + 0.3)
            factors.append(f"Low elevation ({asset.elevation_m:.0f}m) — highly exposed")
        elif asset.elevation_m < 50:
            result.exposure_score = min(1.0, result.exposure_score + 0.1)
            factors.append(f"Moderate elevation ({asset.elevation_m:.0f}m)")

        # Vulnerability by type
        result.vulnerability_score = TYPE_VULNERABILITY.get(asset.asset_type, 0.5)
        factors.append(f"{asset.asset_type.title()} vulnerability: {result.vulnerability_score:.0%}")

        # Adaptive capacity (simplified)
        if asset.asset_type in ("hospital", "school"):
            result.adaptive_capacity = 0.3  # Low — critical services can't relocate
        elif asset.asset_type == "agricultural":
            result.adaptive_capacity = 0.2  # Crops can't be moved
        elif asset.value_usd > 1_000_000:
            result.adaptive_capacity = 0.6  # Higher value = more resources
        else:
            result.adaptive_capacity = 0.4

        # Overall score: exposure × vulnerability × (1 - adaptive capacity) × probability
        result.overall_risk_score = min(1.0, (
            result.exposure_score * 0.4 +
            result.vulnerability_score * 0.3 +
            (1 - result.adaptive_capacity) * 0.15 +
            probability * 0.15
        ))

        # Risk level
        if result.overall_risk_score >= 0.7:
            result.risk_level = "CRITICAL"
        elif result.overall_risk_score >= 0.45:
            result.risk_level = "HIGH"
        elif result.overall_risk_score >= 0.2:
            result.risk_level = "MEDIUM"
        else:
            result.risk_level = "LOW"

        # Damage estimate
        damage_ratio = DAMAGE_RATIO.get(risk_level, 0.05) * result.exposure_score
        result.estimated_damage_usd = asset.value_usd * damage_ratio
        if result.estimated_damage_usd > 0:
            factors.append(f"Estimated damage: ${result.estimated_damage_usd:,.0f}")

        result.factors = factors
        return result

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km."""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def generate_demo_assets(self, lat: float, lon: float) -> list[dict]:
        """Generate demo assets around a location for testing."""
        import random
        random.seed(int(abs(lat * 100 + lon * 100)) % 2**31)

        types = ["residential", "commercial", "hospital", "school", "agricultural", "infrastructure", "industrial"]
        names_by_type = {
            "residential": ["Riverside Homes", "Lowland Estates", "Valley View", "Riverside Apartments"],
            "commercial": ["Central Market", "Business District", "Shopping Center"],
            "hospital": ["General Hospital", "Medical Center", "Emergency Clinic"],
            "school": ["Primary School", "High School", "University Campus"],
            "agricultural": ["Rice Paddies", "Wheat Fields", "Sugarcane Plantation"],
            "infrastructure": ["Main Bridge", "Highway Overpass", "Water Treatment Plant"],
            "industrial": ["Factory Complex", "Warehouse District", "Power Station"],
        }

        assets = []
        for _ in range(8):
            t = random.choice(types)
            offset_lat = random.uniform(-0.1, 0.1)
            offset_lon = random.uniform(-0.1, 0.1)
            names = names_by_type.get(t, ["Asset"])
            assets.append({
                "name": random.choice(names),
                "lat": lat + offset_lat,
                "lon": lon + offset_lon,
                "asset_type": t,
                "value_usd": random.randint(50000, 5000000),
                "elevation_m": random.randint(3, 150),
                "population": random.randint(0, 5000) if t in ("residential", "hospital", "school") else 0,
            })
        return assets
