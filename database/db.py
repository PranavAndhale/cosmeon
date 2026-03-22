"""
Phase 4: Database Operations - CRUD operations for all models.

Provides functions to store and retrieve risk assessments, change events,
and processing logs from the SQLite database.
"""
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import desc

from database.models import (
    Region, RiskAssessmentRecord, ChangeEvent, ProcessingLog, User,
    get_session, init_db,
)

logger = logging.getLogger("cosmeon.database")


class DatabaseManager:
    """Manages all database operations."""

    def __init__(self):
        init_db()
        logger.info("Database initialized")

    # --- Region operations ---

    def create_region(self, name: str, bbox: list[float], geometry_geojson: str = None) -> Region:
        session = get_session()
        try:
            existing = session.query(Region).filter_by(name=name).first()
            if existing:
                logger.info("Region '%s' already exists (id=%d)", name, existing.id)
                return existing

            region = Region(
                name=name,
                bbox_west=bbox[0],
                bbox_south=bbox[1],
                bbox_east=bbox[2],
                bbox_north=bbox[3],
                geometry_geojson=geometry_geojson,
            )
            session.add(region)
            session.commit()
            session.refresh(region)
            logger.info("Created region '%s' (id=%d)", name, region.id)
            return region
        finally:
            session.close()

    def get_region(self, region_id: int) -> Optional[Region]:
        session = get_session()
        try:
            return session.query(Region).get(region_id)
        finally:
            session.close()

    def get_all_regions(self) -> list[Region]:
        session = get_session()
        try:
            return session.query(Region).all()
        finally:
            session.close()

    # --- Risk Assessment operations ---

    def store_risk_assessment(self, assessment, region_id: int) -> RiskAssessmentRecord:
        session = get_session()
        try:
            record = RiskAssessmentRecord(
                region_id=region_id,
                timestamp=datetime.fromisoformat(assessment.timestamp) if isinstance(assessment.timestamp, str) else assessment.timestamp,
                risk_level=assessment.risk_level,
                flood_area_km2=assessment.flood_area_km2,
                total_area_km2=assessment.total_area_km2,
                flood_percentage=assessment.flood_percentage,
                confidence_score=assessment.confidence_score,
                change_type=assessment.change_type,
                water_change_pct=assessment.water_change_pct,
                source_dataset="sentinel-2-l2a",
                source_items=assessment.source_items,
                assessment_details=assessment.assessment_details,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("Stored risk assessment id=%d | risk=%s", record.id, record.risk_level)
            return record
        finally:
            session.close()

    def get_latest_risk(self, region_id: int) -> Optional[RiskAssessmentRecord]:
        session = get_session()
        try:
            return (
                session.query(RiskAssessmentRecord)
                .filter_by(region_id=region_id)
                .order_by(desc(RiskAssessmentRecord.timestamp))
                .first()
            )
        finally:
            session.close()

    def get_risk_history(self, region_id: int, limit: int = 50) -> list[RiskAssessmentRecord]:
        session = get_session()
        try:
            return (
                session.query(RiskAssessmentRecord)
                .filter_by(region_id=region_id)
                .order_by(desc(RiskAssessmentRecord.timestamp))
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def get_regions_by_risk(self, risk_level: str) -> list[dict]:
        session = get_session()
        try:
            # Get latest assessment per region with matching risk level
            from sqlalchemy import func
            subq = (
                session.query(
                    RiskAssessmentRecord.region_id,
                    func.max(RiskAssessmentRecord.timestamp).label("max_ts"),
                )
                .group_by(RiskAssessmentRecord.region_id)
                .subquery()
            )

            results = (
                session.query(RiskAssessmentRecord)
                .join(
                    subq,
                    (RiskAssessmentRecord.region_id == subq.c.region_id)
                    & (RiskAssessmentRecord.timestamp == subq.c.max_ts),
                )
                .filter(RiskAssessmentRecord.risk_level == risk_level.upper())
                .all()
            )

            return [r.to_dict() for r in results]
        finally:
            session.close()

    # --- Change Event operations ---

    def store_change_event(self, change_result, region_id: int) -> ChangeEvent:
        session = get_session()
        try:
            event = ChangeEvent(
                region_id=region_id,
                baseline_item_id=change_result.baseline_id,
                current_item_id=change_result.current_id,
                baseline_date=datetime.fromisoformat(change_result.baseline_date) if isinstance(change_result.baseline_date, str) else change_result.baseline_date,
                current_date=datetime.fromisoformat(change_result.current_date) if isinstance(change_result.current_date, str) else change_result.current_date,
                area_change_km2=change_result.affected_area_km2,
                change_type=change_result.change_type,
                water_change_pct=change_result.water_change_pct,
                new_flood_pixels=change_result.new_flood_pixels,
                receded_pixels=change_result.receded_pixels,
                change_polygons=change_result.change_polygons,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.info("Stored change event id=%d | type=%s", event.id, event.change_type)
            return event
        finally:
            session.close()

    def get_change_events(
        self,
        region_id: int = None,
        from_date: str = None,
        to_date: str = None,
        limit: int = 50,
    ) -> list[ChangeEvent]:
        session = get_session()
        try:
            query = session.query(ChangeEvent)
            if region_id:
                query = query.filter_by(region_id=region_id)
            if from_date:
                query = query.filter(ChangeEvent.current_date >= from_date)
            if to_date:
                query = query.filter(ChangeEvent.current_date <= to_date)
            return query.order_by(desc(ChangeEvent.current_date)).limit(limit).all()
        finally:
            session.close()

    # --- Processing Log operations ---

    def log_processing_step(
        self,
        step: str,
        status: str,
        duration_ms: int = None,
        region_id: int = None,
        item_id: str = None,
        details: dict = None,
    ) -> ProcessingLog:
        session = get_session()
        try:
            log = ProcessingLog(
                step=step,
                status=status,
                duration_ms=duration_ms,
                region_id=region_id,
                item_id=item_id,
                details=details,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log
        finally:
            session.close()

    def get_processing_logs(self, limit: int = 100) -> list[ProcessingLog]:
        session = get_session()
        try:
            return (
                session.query(ProcessingLog)
                .order_by(desc(ProcessingLog.timestamp))
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    # --- Report generation ---

    def generate_summary_report(self, region_id: int) -> dict:
        """Generate a structured summary report for a region."""
        session = get_session()
        try:
            region = session.query(Region).get(region_id)
            if not region:
                return {"error": f"Region {region_id} not found"}

            latest_risk = self.get_latest_risk(region_id)
            history = self.get_risk_history(region_id, limit=10)
            changes = self.get_change_events(region_id=region_id, limit=5)

            report = {
                "region": region.to_dict(),
                "latest_assessment": latest_risk.to_dict() if latest_risk else None,
                "risk_history": [r.to_dict() for r in history],
                "recent_changes": [c.to_dict() for c in changes],
                "summary": {
                    "total_assessments": len(history),
                    "current_risk": latest_risk.risk_level if latest_risk else "UNKNOWN",
                    "latest_flood_pct": latest_risk.flood_percentage if latest_risk else 0,
                    "latest_confidence": latest_risk.confidence_score if latest_risk else 0,
                },
            }

            return report
        finally:
            session.close()

    # --- User / Auth operations ---

    def create_user(self, username: str, hashed_password: str, role: str = "viewer") -> Optional[User]:
        """Create a new user. Returns None if username already exists."""
        session = get_session()
        try:
            existing = session.query(User).filter_by(username=username).first()
            if existing:
                return None
            user = User(username=username, hashed_password=hashed_password, role=role)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info("Created user '%s' with role '%s'", username, role)
            return user
        finally:
            session.close()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Fetch a user by username."""
        session = get_session()
        try:
            return session.query(User).filter_by(username=username).first()
        finally:
            session.close()

    def get_all_users(self) -> list[User]:
        """List all users (admin-only)."""
        session = get_session()
        try:
            return session.query(User).all()
        finally:
            session.close()

    def update_last_login(self, user_id: int) -> None:
        """Record last login timestamp."""
        session = get_session()
        try:
            user = session.query(User).get(user_id)
            if user:
                user.last_login = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    def users_exist(self) -> bool:
        """Return True if any users have been seeded."""
        session = get_session()
        try:
            return session.query(User).count() > 0
        finally:
            session.close()

    def get_monthly_trends(self, region_id: int, months: int = 12) -> list[dict]:
        """Aggregate risk history into monthly buckets for the trend dashboard."""
        from collections import defaultdict
        history = self.get_risk_history(region_id, limit=500)
        monthly: dict = defaultdict(list)
        for r in history:
            ts = r.timestamp
            if ts:
                key = f"{ts.year}-{ts.month:02d}"
                monthly[key].append(r)

        result = []
        for month_key in sorted(monthly.keys())[-months:]:
            records = monthly[month_key]
            risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            for r in records:
                risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
            dominant = max(risk_counts, key=lambda k: risk_counts[k])
            from datetime import datetime as _dt
            label = _dt.strptime(month_key, "%Y-%m").strftime("%b %Y")
            # Helper to safely extract vegetation stress from JSON assessment_details
            def get_veg(record):
                details = record.assessment_details or {}
                if isinstance(details, dict):
                    return float(details.get("vegetation_stress", 0))
                return 0.0

            result.append({
                "month": month_key,
                "month_label": label,
                "avg_flood_pct": round(sum(r.flood_percentage for r in records) / len(records) * 100, 2),
                "max_flood_pct": round(max(r.flood_percentage for r in records) * 100, 2),
                "avg_flood_area_km2": round(sum(r.flood_area_km2 for r in records) / len(records), 1),
                "max_flood_area_km2": round(max(r.flood_area_km2 for r in records), 1),
                "avg_water_change_pct": round(sum(r.water_change_pct for r in records) / len(records) * 100, 2),
                "max_water_change_pct": round(max(r.water_change_pct for r in records) * 100, 2),
                "avg_vegetation_stress": round(sum(get_veg(r) for r in records) / len(records) * 100, 2),
                "max_vegetation_stress": round(max(get_veg(r) for r in records) * 100, 2),
                "avg_confidence": round(sum(r.confidence_score for r in records) / len(records) * 100, 1),
                "dominant_risk_level": dominant,
                "risk_distribution": risk_counts,
                "assessment_count": len(records),
                # ERA5 precipitation fields not available in DB fallback
                "total_precip_mm": 0.0,
                "max_precip_day_mm": 0.0,
                "heavy_rain_days": 0,
            })
        return result

