"""
COSMEON Configuration Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'cosmeon.db'}")

# Planetary Computer (no API key needed for public data)
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Satellite settings
SENTINEL2_COLLECTION = "sentinel-2-l2a"
LANDSAT_COLLECTION = "landsat-c2-l2"
MAX_CLOUD_COVER = 30  # percent

# Default region of interest (Bihar, India - flood-prone)
DEFAULT_BBOX = [83.5, 25.0, 87.5, 27.5]  # [west, south, east, north]
DEFAULT_REGION_NAME = "Bihar, India"

# Processing settings
NDWI_THRESHOLD = 0.3  # pixels above this are classified as water
FLOOD_CHANGE_THRESHOLD = 0.05  # 5% increase in water area triggers flood alert

# Risk classification thresholds (flood_percentage)
RISK_THRESHOLDS = {
    "CRITICAL": 0.25,  # >25% flooded
    "HIGH": 0.15,      # >15% flooded
    "MEDIUM": 0.05,    # >5% flooded
    "LOW": 0.0,        # <=5% flooded
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "cosmeon.log"
