"""
Seed script: Populate cosmeon.db with realistic demo regions,
risk assessments, change events, processing logs, and default users.
Run: python3 seed_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
import random

from database.models import init_db, get_session, Region, RiskAssessmentRecord, ChangeEvent, ProcessingLog, User

init_db()
session = get_session()

# Clear old data (keep users — preserve accounts across re-seeds)
for tbl in [ProcessingLog, ChangeEvent, RiskAssessmentRecord, Region]:
    session.query(tbl).delete()
session.commit()

# ── 1. Regions ──
regions_data = [
    ("Bihar, India",         [83.5, 25.0, 87.5, 27.5]),
    ("Jakarta, Indonesia",   [106.6, -6.4, 107.0, -6.0]),
    ("Bremen, Germany",      [8.6, 53.0, 9.0, 53.2]),
    ("Navi Mumbai, India",   [72.9, 19.0, 73.2, 19.3]),
    ("São Paulo, Brazil",    [-46.8, -23.7, -46.4, -23.4]),
    ("Dhaka, Bangladesh",    [90.3, 23.6, 90.5, 23.9]),
]

created_regions = []
for name, bbox in regions_data:
    r = Region(name=name, bbox_west=bbox[0], bbox_south=bbox[1], bbox_east=bbox[2], bbox_north=bbox[3])
    session.add(r)
    session.flush()
    created_regions.append(r)
session.commit()
print(f"✓ Created {len(created_regions)} regions")

# ── 2. Risk Assessments (24 months per region — rich trend charts) ──
now = datetime.utcnow()

for region in created_regions:
    for month_offset in range(24):
        ts = now - timedelta(days=30 * month_offset + random.randint(0, 5))
        if region.name in ("Bihar, India", "Jakarta, Indonesia", "Dhaka, Bangladesh"):
            flood_pct = random.uniform(0.10, 0.40)
            risk = random.choice(["HIGH", "CRITICAL", "HIGH", "MEDIUM"])
        else:
            flood_pct = random.uniform(0.01, 0.15)
            risk = random.choice(["LOW", "MEDIUM", "LOW", "LOW"])

        total_area = random.uniform(800, 2000)
        flood_area = total_area * flood_pct
        conf = random.uniform(0.82, 0.98)
        water_chg = random.uniform(-0.10, 0.25)
        chg_type = "SIGNIFICANT_INCREASE" if water_chg > 0.1 else ("INCREASE" if water_chg > 0 else "DECREASE")

        rec = RiskAssessmentRecord(
            region_id=region.id,
            timestamp=ts,
            risk_level=risk,
            flood_area_km2=round(flood_area, 2),
            total_area_km2=round(total_area, 2),
            flood_percentage=round(flood_pct, 4),
            confidence_score=round(conf, 4),
            change_type=chg_type,
            water_change_pct=round(water_chg, 4),
            source_dataset="sentinel-2-l2a",
            source_items=["S2A_MSIL2A_demo"],
            assessment_details={"ndwi_threshold": 0.3, "method": "otsu+ndwi"},
        )
        session.add(rec)

session.commit()
print(f"✓ Created {len(created_regions) * 24} risk assessments (24 months)")

# ── 3. Change Events (8 per region) ──
for region in created_regions:
    for i in range(8):
        base_date = now - timedelta(days=30 * (i + 1))
        curr_date = now - timedelta(days=30 * i)
        water_chg = random.uniform(-0.08, 0.20)
        new_flood_px = random.randint(100, 8000)
        receded_px = random.randint(50, 3000)
        area_chg = random.uniform(0.5, 50.0)
        chg_type = "SIGNIFICANT_INCREASE" if water_chg > 0.1 else ("INCREASE" if water_chg > 0 else "DECREASE")

        ev = ChangeEvent(
            region_id=region.id,
            baseline_item_id=f"S2A_baseline_{i}",
            current_item_id=f"S2A_current_{i}",
            baseline_date=base_date,
            current_date=curr_date,
            area_change_km2=round(area_chg, 2),
            change_type=chg_type,
            water_change_pct=round(water_chg, 4),
            new_flood_pixels=new_flood_px,
            receded_pixels=receded_px,
            change_polygons={"type": "FeatureCollection", "features": []},
        )
        session.add(ev)

session.commit()
print(f"✓ Created {len(created_regions) * 8} change events")

# ── 4. Processing Logs ──
steps = [
    ("STAC_SEARCH",     "completed", "Searched Sentinel-2 catalog for AOI"),
    ("DOWNLOAD_SCENES", "completed", "Downloaded 4 scenes from Planetary Computer"),
    ("NDWI_COMPUTE",    "completed", "Computed NDWI water index for all bands"),
    ("FLOOD_CLASSIFY",  "completed", "Classified flood/non-flood pixels via Otsu threshold"),
    ("CHANGE_DETECT",   "completed", "Compared baseline vs current water masks"),
    ("RISK_ASSESS",     "completed", "Generated risk classifications for all regions"),
    ("EXTERNAL_DATA",   "completed", "Fetched rainfall and elevation data from Open-Meteo"),
    ("PREDICTION",      "completed", "Ran GBM flood predictor model"),
    ("REPORT_GEN",      "completed", "Generated structured summary reports"),
    ("PIPELINE_DONE",   "completed", "Full pipeline completed successfully"),
]

for region in created_regions:
    for idx, (step, status, detail) in enumerate(steps):
        log = ProcessingLog(
            timestamp=now - timedelta(minutes=len(steps) - idx),
            step=step,
            status=status,
            duration_ms=random.randint(200, 5000),
            region_id=region.id,
            item_id=f"S2A_item_{region.id}_{idx}",
            details={"message": detail},
        )
        session.add(log)

session.commit()
print(f"✓ Created {len(created_regions) * len(steps)} processing logs")

# ── 5. Default Users (only if no users exist yet) ──
from api.auth import hash_password

existing_count = session.query(User).count()
if existing_count == 0:
    default_users = [
        ("admin",   "admin123",   "admin"),
        ("analyst", "analyst123", "analyst"),
        ("viewer",  "viewer123",  "viewer"),
    ]
    for uname, pwd, role in default_users:
        u = User(username=uname, hashed_password=hash_password(pwd), role=role)
        session.add(u)
    session.commit()
    print(f"✓ Created {len(default_users)} default users: admin / analyst / viewer")
else:
    print(f"✓ Users already exist ({existing_count}), skipping user seed")

session.close()
print("\n✅ Database seeded! Credentials: admin/admin123  analyst/analyst123  viewer/viewer123")
