"""
Phase 4: Automated Change Detection (ACD) Scheduler.

Implements "set-and-forget" monitoring for Areas of Interest (AOI).
Tracks automatic satellite revisits, compares sequential passes,
and triggers alerts when changes exceed configurable thresholds.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("cosmeon.processing.acd")


@dataclass
class AOI:
    """Area of Interest for monitoring."""
    id: int = 0
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    radius_km: float = 10.0
    threshold_pct: float = 2.0  # Alert if change > this %
    check_interval_days: int = 5  # Satellite revisit interval
    active: bool = True
    created_at: str = ""
    last_checked: str = ""
    alert_count: int = 0

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "lat": self.lat, "lon": self.lon,
            "radius_km": self.radius_km, "threshold_pct": self.threshold_pct,
            "check_interval_days": self.check_interval_days, "active": self.active,
            "created_at": self.created_at, "last_checked": self.last_checked,
            "alert_count": self.alert_count,
        }


@dataclass
class ChangeAlert:
    """Alert triggered by detected change."""
    aoi_id: int = 0
    aoi_name: str = ""
    change_type: str = ""  # "flood_increase", "deforestation", "water_recession"
    change_pct: float = 0.0
    severity: str = "INFO"  # INFO, WARNING, CRITICAL
    description: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    detected_at: str = ""
    acknowledged: bool = False

    def to_dict(self):
        return {
            "aoi_id": self.aoi_id, "aoi_name": self.aoi_name,
            "change_type": self.change_type, "change_pct": round(self.change_pct, 2),
            "severity": self.severity, "description": self.description,
            "baseline_value": round(self.baseline_value, 4),
            "current_value": round(self.current_value, 4),
            "detected_at": self.detected_at, "acknowledged": self.acknowledged,
        }


class ACDScheduler:
    """Manages automated change detection monitoring."""

    def __init__(self):
        self._aois: list[AOI] = []
        self._alerts: list[ChangeAlert] = []
        self._next_id = 1
        self._alert_id = 1
        logger.info("ACDScheduler initialized")

    def add_aoi(
        self, name: str, lat: float, lon: float,
        radius_km: float = 10.0, threshold_pct: float = 2.0,
        check_interval_days: int = 5,
    ) -> AOI:
        """Register a new Area of Interest for monitoring."""
        aoi = AOI(
            id=self._next_id, name=name, lat=lat, lon=lon,
            radius_km=radius_km, threshold_pct=threshold_pct,
            check_interval_days=check_interval_days,
            created_at=datetime.utcnow().isoformat(),
            last_checked="",
        )
        self._aois.append(aoi)
        self._next_id += 1
        logger.info("AOI registered: %s (id=%d, threshold=%.1f%%)", name, aoi.id, threshold_pct)
        return aoi

    def remove_aoi(self, aoi_id: int) -> bool:
        """Deactivate an AOI."""
        for aoi in self._aois:
            if aoi.id == aoi_id:
                aoi.active = False
                return True
        return False

    def list_aois(self) -> list[dict]:
        return [a.to_dict() for a in self._aois if a.active]

    def get_alerts(self, aoi_id: int = None, unacknowledged_only: bool = False) -> list[dict]:
        alerts = self._alerts
        if aoi_id:
            alerts = [a for a in alerts if a.aoi_id == aoi_id]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return [a.to_dict() for a in alerts[-50:]]  # Last 50

    def acknowledge_alert(self, alert_index: int) -> bool:
        if 0 <= alert_index < len(self._alerts):
            self._alerts[alert_index].acknowledged = True
            return True
        return False

    def check_aoi(
        self, aoi_id: int,
        current_flood_pct: float = 0.0,
        previous_flood_pct: float = 0.0,
        current_vegetation: float = 0.7,
        previous_vegetation: float = 0.7,
    ) -> list[dict]:
        """
        Check an AOI for changes against baseline.

        Returns list of new alerts if thresholds are exceeded.
        """
        aoi = None
        for a in self._aois:
            if a.id == aoi_id and a.active:
                aoi = a
                break
        if not aoi:
            return []

        new_alerts = []
        now = datetime.utcnow().isoformat()
        aoi.last_checked = now

        # Check flood change
        flood_change = (current_flood_pct - previous_flood_pct) * 100  # as percentage points
        if abs(flood_change) > aoi.threshold_pct:
            if flood_change > 0:
                change_type = "flood_increase"
                desc = f"Flood coverage increased by {flood_change:.1f}% in {aoi.name}"
            else:
                change_type = "water_recession"
                desc = f"Water recession of {abs(flood_change):.1f}% detected in {aoi.name}"

            severity = "CRITICAL" if abs(flood_change) > 10 else "WARNING" if abs(flood_change) > 5 else "INFO"

            alert = ChangeAlert(
                aoi_id=aoi.id, aoi_name=aoi.name,
                change_type=change_type, change_pct=flood_change,
                severity=severity, description=desc,
                baseline_value=previous_flood_pct, current_value=current_flood_pct,
                detected_at=now,
            )
            self._alerts.append(alert)
            aoi.alert_count += 1
            new_alerts.append(alert.to_dict())

        # Check vegetation change (deforestation)
        veg_change = (previous_vegetation - current_vegetation) * 100
        if veg_change > aoi.threshold_pct:
            alert = ChangeAlert(
                aoi_id=aoi.id, aoi_name=aoi.name,
                change_type="deforestation",
                change_pct=veg_change,
                severity="WARNING" if veg_change > 5 else "INFO",
                description=f"Vegetation loss of {veg_change:.1f}% detected in {aoi.name}",
                baseline_value=previous_vegetation,
                current_value=current_vegetation,
                detected_at=now,
            )
            self._alerts.append(alert)
            aoi.alert_count += 1
            new_alerts.append(alert.to_dict())

        if new_alerts:
            logger.warning("ACD: %d alerts generated for AOI %s", len(new_alerts), aoi.name)

        return new_alerts

    def get_monitoring_status(self) -> dict:
        """Get overall monitoring system status."""
        active = [a for a in self._aois if a.active]
        unack = [a for a in self._alerts if not a.acknowledged]
        return {
            "active_aois": len(active),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len(unack),
            "critical_alerts": len([a for a in unack if a.severity == "CRITICAL"]),
            "aois": [a.to_dict() for a in active],
            "recent_alerts": [a.to_dict() for a in self._alerts[-10:]],
        }
