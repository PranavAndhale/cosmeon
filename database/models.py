"""
Phase 4: Database Models - SQLAlchemy ORM for structured state table.

Stores regions, risk assessments, change events, and processing logs.
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config.settings import DATABASE_URL

Base = declarative_base()


class Region(Base):
    """Geographic region being monitored."""
    __tablename__ = "regions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    bbox_west = Column(Float, nullable=False)
    bbox_south = Column(Float, nullable=False)
    bbox_east = Column(Float, nullable=False)
    bbox_north = Column(Float, nullable=False)
    geometry_geojson = Column(Text)  # GeoJSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    risk_assessments = relationship("RiskAssessmentRecord", back_populates="region")
    change_events = relationship("ChangeEvent", back_populates="region")

    @property
    def bbox(self):
        return [self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north]

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "bbox": self.bbox,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RiskAssessmentRecord(Base):
    """Risk assessment result for a region at a point in time."""
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    risk_level = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    flood_area_km2 = Column(Float, default=0.0)
    total_area_km2 = Column(Float, default=0.0)
    flood_percentage = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    change_type = Column(String(50))
    water_change_pct = Column(Float, default=0.0)
    source_dataset = Column(String(255))
    source_items = Column(JSON)
    assessment_details = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)

    region = relationship("Region", back_populates="risk_assessments")

    def to_dict(self):
        return {
            "id": self.id,
            "region_id": self.region_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "risk_level": self.risk_level,
            "flood_area_km2": self.flood_area_km2,
            "total_area_km2": self.total_area_km2,
            "flood_percentage": self.flood_percentage,
            "confidence_score": self.confidence_score,
            "change_type": self.change_type,
            "water_change_pct": self.water_change_pct,
            "source_items": self.source_items,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class ChangeEvent(Base):
    """Change detection event comparing two time periods."""
    __tablename__ = "change_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=False)
    baseline_item_id = Column(String(255))
    current_item_id = Column(String(255))
    baseline_date = Column(DateTime)
    current_date = Column(DateTime)
    area_change_km2 = Column(Float, default=0.0)
    change_type = Column(String(50))
    water_change_pct = Column(Float, default=0.0)
    new_flood_pixels = Column(Integer, default=0)
    receded_pixels = Column(Integer, default=0)
    change_polygons = Column(JSON)  # GeoJSON features
    processed_at = Column(DateTime, default=datetime.utcnow)

    region = relationship("Region", back_populates="change_events")

    def to_dict(self):
        return {
            "id": self.id,
            "region_id": self.region_id,
            "baseline_date": self.baseline_date.isoformat() if self.baseline_date else None,
            "current_date": self.current_date.isoformat() if self.current_date else None,
            "area_change_km2": self.area_change_km2,
            "change_type": self.change_type,
            "water_change_pct": self.water_change_pct,
            "new_flood_pixels": self.new_flood_pixels,
            "receded_pixels": self.receded_pixels,
            "change_polygons": self.change_polygons,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class ProcessingLog(Base):
    """Log of every processing step for audit trail."""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    step = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # started, completed, failed
    duration_ms = Column(Integer)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=True)
    item_id = Column(String(255))
    details = Column(JSON)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "step": self.step,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "region_id": self.region_id,
            "item_id": self.item_id,
            "details": self.details,
        }


class User(Base):
    """User account for authentication and role-based access."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, unique=True)
    hashed_password = Column(String(512), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")  # admin, analyst, viewer
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


# Database engine and session factory
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_session():
    """Get a new database session."""
    return SessionLocal()
