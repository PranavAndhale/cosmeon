<div align="center">

<br/>

```
 ██████╗ ██████╗ ███████╗███╗   ███╗███████╗ ██████╗ ███╗   ██╗
██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔════╝██╔═══██╗████╗  ██║
██║     ██║   ██║███████╗██╔████╔██║█████╗  ██║   ██║██╔██╗ ██║
██║     ██║   ██║╚════██║██║╚██╔╝██║██╔══╝  ██║   ██║██║╚██╗██║
╚██████╗╚██████╔╝███████║██║ ╚═╝ ██║███████╗╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
```

### **Geospatial Intelligence Engine**
*Real-time flood risk detection, infrastructure exposure analysis, and vegetation health monitoring — powered by satellite imagery and machine learning.*

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-cosmeon.onrender.com-00E5FF?style=for-the-badge&labelColor=0B0E11)](https://cosmeon.onrender.com)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Frontend](https://img.shields.io/badge/Frontend-Next.js%2016-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%2B%20LightGBM%20Ensemble-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE)

</div>

---

## 🌍 What is Cosmeon?

Cosmeon is a **production-grade Geospatial Intelligence Platform** that transforms raw multi-modal satellite imagery into actionable risk assessments in real time. It's built for analysts, planners, and researchers who need to go beyond static maps and into dynamic, AI-driven geospatial insights.

The platform is anchored around three core **Analysis Orbs**:

| Orb | Focus | Data Source | Color |
|-----|-------|-------------|-------|
| 🌊 **Flood Risk** | Inundation extent via SAR backscatter + GloFAS discharge | Sentinel-1 + GloFAS v4 | Cyan |
| 🏗️ **Infrastructure Exposure** | Water-human structure intersection | Sentinel-1 + OSM | Orange |
| 🌿 **Vegetation Health** | NDVI anomaly + ERA5 dryness stress index | Sentinel-2 + ERA5 | Green |

---

## ✨ Key Features

### 🤖 Machine Learning
- **XGBoost + LightGBM 55/45 Soft-Voting Ensemble** trained on 13 weather and terrain features — precipitation (7-day, 30-day), soil moisture, elevation, slope, GloFAS discharge anomaly, seasonal indicators, and more
- **Thread-Safe Concurrent Training** — `threading.Lock` prevents race conditions when multiple requests trigger simultaneous re-training; non-blocking acquire with graceful fallback
- **Deterministic Predictions** — local `np.random.RandomState(42)` per training run; eliminates global seed mutation so predictions are stable across concurrent requests
- **Background Pre-Training at Startup** — ML model warms up in a daemon thread before the first request hits, eliminating Render's 30s cold-start timeout
- **SHAP Explainability** — every prediction comes with a full feature importance breakdown so analysts know *why* a risk level was assigned
- **6-Month Forecasting** via ARIMA-style time-series extrapolation with confidence bands
- **Anomaly Detection** using Isolation Forest for detecting out-of-distribution satellite events

### 📡 Data & Fusion
- **Live GloFAS v4 Integration** — real river discharge data ingested per region; used as the ground-truth label source for ML training and as a live validation signal alongside the model
- **ERA5 Climate Reanalysis** — Open-Meteo API delivers hourly precipitation, ET₀, and soil moisture; aggregated into 7-day and 30-day windows per location
- **Vegetation Stress Index** — computed as a climatological dryness anomaly: `50 + (clim_mean_precip − actual_precip) / clim_mean_precip × 50`, centred at 50 (normal), rising when drier than seasonal average, falling in wet conditions — eliminates the flat-zero bug that afflicted monsoon regions
- **Multi-Sensor Fusion** — dynamically weights SAR vs Optical imagery based on real-time cloud cover conditions (Sentinel-1 dominates when cloud cover > 70%)
- **Automated Change Detection** — calculates temporal deltas between baseline and current state, quantifying km² gained/lost per period
- **Historical Trend Analysis** — 24-month ERA5 lookback with monthly aggregation, rendered as interactive Recharts graphs; registered regions now fetch ERA5 trends directly rather than relying on stale DB records

### 🧠 Advanced Intelligence
- **ML vs GloFAS Comparison Panel** — side-by-side comparison of the ensemble model's prediction against live GloFAS discharge signal; includes agreement score, per-source explanation blocks (cyan for GloFAS, violet for ML), and a data snapshot of the exact inputs used — frontend passes the already-computed prediction so the explain endpoint never re-derives and diverges
- **Always-Present Explanation Reasons** — every explain response includes a quantitative data snapshot (rainfall 7d/30d, precip anomaly, soil moisture, elevation, GloFAS discharge ratio) so the reasons panel is never empty
- **Financial Impact Simulation** — translates flood depth and income class into Value-at-Risk estimates with tiered USD formatting (`$1.2M`, `$450K`, `$320`) and a confidence badge (HIGH / MEDIUM / LOW); no more `$0K` for sub-thousand values
- **AI NLG Summaries** — integrates with `gpt-4o-mini` to generate human-readable narrative insights from raw risk vectors
- **Compound Risk Engine** — models cascading hazard interactions (Flood → Landslide → Infrastructure failure)
- **Human-in-the-Loop Feedback** — analysts can verify or correct predictions, queuing data for the next re-training epoch

### 🗺️ Navigation & UX
- **Satellite Base Map** — Esri World Imagery + Carto Dark Labels via MapLibre GL JS (`react-map-gl`)
- **Global Geocoding Search** — type any city, country, or coordinates (e.g. `26.0 85.5`) and fly there instantly
- **Ad-Hoc Location Analysis** — click anywhere on the map to generate an instant risk assessment
- **Role-Based Auth** — JWT-authenticated users with Admin / Analyst / Viewer access levels
- **Auto-Scheduler** — background monitoring jobs that re-evaluate all AOIs on a configurable interval

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        COSMEON PLATFORM                          │
├──────────────────────────┬───────────────────────────────────────┤
│   FRONTEND (Next.js)     │          BACKEND (FastAPI)            │
│                          │                                       │
│  • react-map-gl          │  • /api/regions                       │
│  • Recharts              │  • /api/trends/{id}                   │
│  • Framer Motion         │  • /api/trends/location               │
│  • Tailwind CSS          │  • /api/explain/{id}                  │
│  • Analysis Orbs         │  • /api/explain/location              │
│  • Webflow Landing Page  │  • /api/forecast/{id}                 │
│                          │  • /api/financial-impact              │
│                          │  • /api/nlg/summary (GPT-4o)          │
│                          │  • /api/ml/train                      │
├──────────────────────────┴───────────────────────────────────────┤
│               ML PIPELINE (XGBoost + LightGBM)                   │
│  55/45 Soft-Voting Ensemble • SHAP • ARIMA • IsolationForest     │
│  threading.Lock • RandomState(42) • Background Pre-Training      │
├──────────────────────────────────────────────────────────────────┤
│          SATELLITE & CLIMATE DATA LAYER                          │
│  GloFAS v4 (discharge) • ERA5 via Open-Meteo (precip, ET₀)       │
│  Sentinel-1 SAR • Sentinel-2 NDVI • MODIS Land Cover             │
├──────────────────────────────────────────────────────────────────┤
│              DATABASE (SQLite via SQLAlchemy)                    │
│  regions • risk_assessments • change_events • users              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- `pip` and `npm`

### 1. Clone the repo
```bash
git clone https://github.com/PranavAndhale/cosmeon.git
cd cosmeon
```

### 2. Backend Setup
```bash
# Create + activate virtualenv
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Seed the database with demo data
python seed_demo.py

# Train the ML model
python train_model.py

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`.

---

## 🐳 Docker (Full Stack)

```bash
docker-compose up --build
```

This spins up the backend + frontend as a single unified service.

---

## 📁 Project Structure

```
cosmeon/
├── api/
│   └── routes.py               # All FastAPI routes, middleware, catch-all static serving
├── database/                   # SQLAlchemy models, DB manager, migrations
├── ml/                         # Feature engineering, SHAP utils
├── processing/
│   ├── predictor.py            # XGBoost + LightGBM ensemble, training lock, explain
│   └── live_flood_data.py      # GloFAS v4 + ERA5 fetcher, ML vs GloFAS diff analysis
├── ingestion/                  # Data ingestion pipelines
├── frontend/
│   ├── public/
│   │   ├── index.html          # Webflow-exported landing page (served at /)
│   │   └── *.png               # Hero, engine UI, solutions imagery
│   ├── src/app/engine/         # The Geospatial Engine UI (served at /engine)
│   └── src/lib/api.ts          # Typed API client (fetchExplanation, explainLocation)
├── data/                       # Processed ML artifacts (models, scalers)
├── train_model.py              # Standalone model training script
├── seed_demo.py                # Seeds DB with demo regions + assessments
├── Dockerfile.render           # Production Dockerfile (used by Render)
├── render.yaml                 # Render deployment configuration
└── requirements.txt            # Python dependencies
```

---

## 🌐 Deployment

This project is deployed on **[Render](https://render.com)** using Docker.

- **Live URL**: [https://cosmeon.onrender.com](https://cosmeon.onrender.com)
- **Deploy Config**: [`render.yaml`](render.yaml)

Every push to `main` triggers an automatic redeploy on Render.

The **landing page** is also deployed as a standalone service at a separate Render URL via the [`cosmeon-landing`](https://github.com/PranavAndhale/cosmeon-landing) repository — an Express static server serving the same Webflow-exported HTML.

---

## 🛠️ Tech Stack

**Frontend**
| Tool | Role |
|------|------|
| Next.js 16 (Turbopack) | React framework + routing |
| Tailwind CSS | Utility-first styling |
| Framer Motion | UI animations |
| Recharts | Data visualization |
| react-map-gl | MapLibre GL JS wrapper |

**Backend**
| Tool | Role |
|------|------|
| FastAPI | REST API framework |
| SQLAlchemy | ORM + query builder |
| SQLite | Lightweight production DB |
| XGBoost | Primary gradient boosting classifier (55% ensemble weight) |
| LightGBM | Secondary gradient boosting classifier (45% ensemble weight) |
| SHAP | Model explainability |
| statsmodels | ARIMA time-series forecasting |
| OpenAI SDK | NLG narrative generation |

**Data Sources**
| Source | What it provides |
|--------|-----------------|
| GloFAS v4 | Live river discharge — ground truth for ML labels + validation signal |
| ERA5 via Open-Meteo | Hourly precipitation, evapotranspiration (ET₀), soil moisture |
| Sentinel-1 | SAR backscatter for flood inundation mapping |
| Sentinel-2 | NDVI for vegetation health |
| MODIS | Land cover classification |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🛰️ by [Pranav Andhale](https://github.com/PranavAndhale)

⭐ **Star this repo if you found it useful!**

</div>
