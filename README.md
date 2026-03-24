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
*Real-time flood risk detection, infrastructure exposure analysis, and vegetation health monitoring — powered by satellite imagery, live hydrology data, and machine learning.*

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-cosmeon.onrender.com-00E5FF?style=for-the-badge&labelColor=0B0E11)](https://cosmeon.onrender.com)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Frontend](https://img.shields.io/badge/Frontend-Next.js%2016-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![Prediction](https://img.shields.io/badge/Prediction-TieredFloodPredictor%20%2B%20SHAP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLG](https://img.shields.io/badge/NLG-Gemini%202.0%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE)

</div>

---

## 🌍 What is Cosmeon?

Cosmeon is a **production-grade Geospatial Intelligence Platform** that transforms raw multi-modal satellite imagery and live hydrology data into actionable flood risk assessments in real time. It is built for analysts, planners, and researchers who need to go beyond static maps and into dynamic, AI-driven geospatial insights.

The platform is anchored around three core **Analysis Orbs**:

| Orb | Focus | Data Source | Color |
|-----|-------|-------------|-------|
| 🌊 **Flood Risk** | ML flood probability via GloFAS v4 river discharge + ERA5 weather | TieredFloodPredictor + GloFAS v4 | Cyan |
| 🏗️ **Infrastructure Exposure** | Flood–terrain intersection + soil saturation scoring | ERA5 soil moisture + Open-Meteo DEM | Orange |
| 🌿 **Vegetation Health** | NDVI anomaly + FAO-56 ET₀ water balance stress index | NASA MODIS NDVI + ERA5 | Green |

---

## ✨ Key Features

### 🤖 Prediction — TieredFloodPredictor

- **Deterministic Compound Scorer** — aggregates outputs from GloFAS v4, ERA5, ECMWF IFS soil moisture, and regional historical records into a calibrated flood probability and risk level. No custom model training required; predictions are available instantly on startup
- **Tiered GloFAS v4 Fallback (T1 → T4)** — attempts four progressively broader GloFAS API queries (point → 0.1° → 0.25° → 0.5° radius) before falling back to ERA5-only mode, ensuring data availability for any global location
- **Calibrated Risk Levels** — GloFAS flood risk levels are mapped to real flood probabilities (e.g. GloFAS HIGH → ~62% probability) using empirically tuned baselines, then compounded with ERA5 precipitation anomaly and soil saturation signals
- **Feature Attribution Explainability** — every prediction ships a ranked breakdown of the top contributing signals with human-readable influence labels (`INCREASES RISK` / `DECREASES RISK`) and importance scores so analysts always know *why* a risk level was assigned
- **Plain Language Verdict** — the NLG engine distils the top 2–3 drivers into a 2–3 sentence natural-language summary (e.g. *"This location is at HIGH flood risk (62% probability). GloFAS v4 river discharge is the primary driver at +1.8σ above seasonal norms. Elevated 7-day rainfall of 89 mm and 72% soil saturation compound the risk."*) — readable by non-technical field staff without any chart interpretation

### 📡 Data Integration

- **GloFAS v4 (Copernicus Emergency Management Service)** — live river discharge ingested per location; discharge current value, 30-year climatological mean, and σ-anomaly are direct ML features, making the model sensitive to actual water-level extremes rather than proxies alone
- **ERA5 Climate Reanalysis via Open-Meteo** — hourly precipitation, evapotranspiration (ET₀), and soil moisture aggregated into 7-day and 30-day windows; used for precipitation anomaly computation and ML feature construction
- **NASA MODIS NDVI (MOD13Q1, 250m)** — real satellite-derived vegetation index fetched from model hub; proxy estimate used as fallback when live data is unavailable
- **FAO-56 Penman-Monteith ET₀** — water-balance-based vegetation stress index that correctly handles monsoon regions (eliminates the flat-zero artifact that afflicts simpler NDVI-only approaches)
- **Multi-Layer Data Fusion** — combines optical (MODIS NDVI), model-estimated radar proxy, ERA5 thermal reanalysis, and Open-Meteo weather layers with adaptive quality-score weighting based on data availability
- **Historical Trend Analysis** — 24-month ERA5 lookback with monthly aggregation rendered as interactive Recharts graphs; heavy rain days computed from daily precipitation thresholds (>20 mm)

### 🧠 Advanced Intelligence

- **Situation Board** — live per-region risk ranking sorted by severity, driven directly by the ML predictor output (not stale database values); classifies each region as FLOODING NOW / IMMINENT / WATCH / RECEDING / NORMAL based on live flood probability and GloFAS discharge anomaly
- **Compound Risk Engine (INFORM Risk Index)** — multi-hazard composite scoring using the EU JRC INFORM methodology: `Risk = (Hazard × Exposure × Vulnerability)^(1/3)`. Hazard combines ML flood probability, 7-day precipitation intensity, and ERA5 thermal anomaly; Exposure from World Bank population density and Open-Meteo DEM elevation; Vulnerability from ERA5 soil moisture and FAO-56 vegetation stress. Cascading interactions (e.g. flood + heat → waterborne disease; soil saturation + vegetation loss → erosion) are modelled using INFORM's geometric coupling factor
- **Financial Impact Engine (JRC Depth-Damage Functions)** — translates flood depth into sector-level damage estimates using published JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017). Flood depth is derived from GloFAS discharge σ-anomaly. Indirect costs follow UNDRR Sendai Framework ratios by World Bank income classification; displacement costs use UNHCR per-capita brackets. All figures are consistent between the live website panel and the exported PDF report
- **6-Month Probabilistic Forecasting** — Open-Meteo historical climate archive + seasonal decomposition and trend projection, producing per-month risk probabilities with upper/lower confidence bands. Optionally uses Amazon Chronos (time-series foundation model) when available
- **PDF Report Generation** — branded A4 reports generated server-side with ReportLab Platypus: risk badge, NLG narrative, ML prediction drivers table, GloFAS discharge, 6-month forecast table, INFORM compound risk hazard layers, and financial impact breakdown. All data gathered in parallel via `ThreadPoolExecutor` to avoid blocking the API
- **NLG Summaries (Gemini 2.0 Flash)** — the Natural Language Generation engine first attempts **Google Gemini 2.0 Flash** (via `google-genai`) to produce structured JSON narratives and highlight bullets from live signal data; falls back to a deterministic template engine when no API key is present, ensuring summaries are always available. The prompt explicitly instructs the model on GloFAS tier semantics, discharge anomaly, and compound signals so outputs are factually grounded, not hallucinated
- **Risk Build-up Waterfall Chart** — a new explainability visualisation inside the Prediction Drivers panel that shows how the final flood probability is constructed step by step: starting from a 14% neutral baseline, each factor (GloFAS river discharge, ERA5 precipitation anomaly, ECMWF IFS soil moisture, historical baseline) adds or subtracts probability. Bars extend right in red for risk-increasing factors and left in cyan for risk-reducing ones, with the running cumulative shown at each step — giving analysts an instant audit trail for any prediction
- **7-Day Discharge Forecast Slider** — an interactive temporal panel positioned above the ML Prediction card in both registered-region and ad-hoc panels. It renders a 60px mini AreaChart of the 7-day GloFAS v4 forecast discharge curve, a scrubbing slider with risk-color-filled track, and a projected-risk badge that updates live as the analyst drags through future days (e.g. *"DAY +3 · 2026-03-27 — HIGH · 62%"*). Falls back to recent historical discharge when forecast data is unavailable
- **Prediction Explainability Panel** — interactive breakdown of the top contributing signals (GloFAS v4 discharge, ERA5 precipitation, soil saturation, historical baseline) with a consensus confidence score showing how many independent sources agree with the model's conclusion. Now also includes the plain language verdict and the Risk Build-up waterfall chart for a complete, multi-layer explanation
- **Automated Change Detection** — calculates temporal deltas between baseline and current state per registered region, quantifying km² gained/lost per period
- **Human-in-the-Loop Feedback** — analysts can submit accuracy verdicts (correct / incorrect) on predictions; feedback is stored and tracked via `/api/feedback/stats`

### 🗺️ Navigation & UX

- **Satellite Base Map** — Esri World Imagery + Carto Dark Labels via MapLibre GL JS (`react-map-gl`)
- **Risk-Differentiated Map Markers** — markers encode two independent data dimensions simultaneously: the *outer glow and border* reflects the current **situation status** (FLOODING_NOW / IMMINENT / WATCH / RECEDING / NORMAL), while an *inner ring* encodes the **ML risk level** (CRITICAL / HIGH / MEDIUM / LOW) in a separate color. A pulse animation varies in speed by urgency — FLOODING_NOW pulses every 1 second, IMMINENT every 2 seconds, WATCH every 3 seconds — so a field coordinator can instantly read the map without opening any panel. Marker diameter also scales with urgency (16 px → 9 px). Ad-hoc location markers dynamically adopt the prediction risk color rather than a static cyan
- **Global Geocoding Search** — type any city, country, or coordinates (e.g. `26.0 85.5`) and fly there instantly
- **Ad-Hoc Location Analysis** — click anywhere on the map to run a full ML prediction + GloFAS discharge assessment for any coordinate on Earth, without needing a pre-registered region
- **Shareable Links** — copy a direct URL to any registered region (`?region=ID`) or ad-hoc location (`?lat=...&lon=...&name=...`); the app restores full state on load
- **Role-Based Auth** — JWT-authenticated users with Admin / Analyst / Viewer access levels
- **Auto-Scheduler** — background monitoring jobs that re-evaluate all registered AOIs every 6 hours (configurable via API)

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
│  • Framer Motion         │  • /api/situation/all                 │
│  • Tailwind CSS          │  • /api/explain/location              │
│  • Analysis Orbs         │  • /api/forecast/{id}                 │
│  • 7-Day Forecast Slider │  • /api/financial/{id}                │
│  • XAI Waterfall Chart   │  • /api/compound/{id}                 │
│  • Risk Map Markers      │  • /api/reports/{id}/pdf              │
│  • Shareable URL state   │  • /api/discharge/{id}                │
│                          │  • /api/nlg/summary                   │
├──────────────────────────┴───────────────────────────────────────┤
│            TIERED FLOOD PREDICTOR                                │
│  GloFAS T1→T4 Tiered Fallback  •  ERA5 Compound Scorer           │
│  ECMWF IFS Soil Moisture  •  Waterfall Decomposer                │
│  Feature Attribution Drivers  •  Plain Language Verdict          │
│  Daily Progression (7-day forecast risk timeline)                │
├──────────────────────────────────────────────────────────────────┤
│          INTELLIGENCE ENGINES                                    │
│  CompoundRiskEngine (INFORM/EU JRC)  •  FinancialImpactEngine    │
│  (JRC Depth-Damage + UNDRR Sendai)  •  ForecastEngine            │
│  NLGEngine (Gemini 2.0 Flash + template fallback)                │
│  ReportGenerator (PDF)                                           │
├──────────────────────────────────────────────────────────────────┤
│          SATELLITE & CLIMATE DATA LAYER                          │
│  GloFAS v4 (Copernicus)  •  ERA5 via Open-Meteo  •  FAO-56 ET₀   │
│  NASA MODIS NDVI (MOD13Q1)  •  Open-Meteo DEM Elevation          │
│  World Bank GDP + Population  •  Open-Meteo Climate Archive      │
├──────────────────────────────────────────────────────────────────┤
│              DATABASE (SQLite via SQLAlchemy)                    │
│  regions  •  risk_assessments  •  change_events  •  users        │
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

# Seed the database with demo regions
python seed_demo.py

# Start the API server
uvicorn api.routes:app --host 0.0.0.0 --port 8000 --reload
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

## 📁 Project Structure

```
cosmeon/
├── api/
│   └── routes.py                   # All FastAPI routes, middleware, PDF endpoints
├── database/                       # SQLAlchemy models, DB manager, migrations
├── processing/
│   ├── tiered_predictor.py         # TieredFloodPredictor — GloFAS T1→T4 + ERA5 ML
│   ├── compound_risk.py            # INFORM Risk Index compound hazard engine (EU JRC)
│   ├── financial_impact.py         # JRC Depth-Damage + UNDRR Sendai financial engine
│   ├── report_generator.py         # ReportLab PDF generation (A4 branded reports)
│   ├── forecast_engine.py          # Seasonal decomposition + trend 6-month forecasting
│   ├── nlg_engine.py               # NLG engine — Gemini 2.0 Flash primary, template fallback
│   ├── live_analysis.py            # Scheduled analysis engine for registered regions
│   ├── data_fusion.py              # Multi-layer sensor fusion with quality weighting
│   ├── change_detector.py          # Temporal change detection (km² delta per period)
│   ├── live_flood_data.py          # GloFAS v4 + ERA5 data fetchers
│   └── model_hub.py                # Tiered World Bank / GloFAS / elevation data hub
├── frontend/
│   ├── public/
│   │   ├── index.html              # Webflow-exported landing page (served at /)
│   │   └── *.png                   # Hero, engine UI, solutions imagery
│   ├── src/app/engine/             # Geospatial Engine UI (served at /engine)
│   └── src/lib/api.ts              # Typed API client
├── data/                           # Cached data artifacts and regional records
├── seed_demo.py                    # Seeds DB with demo regions + risk assessments
├── Dockerfile.render               # Production Dockerfile (used by Render)
├── render.yaml                     # Render deployment configuration
└── requirements-web.txt            # Production Python dependencies (Render uses this)
```

---

## 🌐 Deployment

This project is deployed on **[Render](https://render.com)** using Docker.

- **Live URL**: [https://cosmeon.onrender.com](https://cosmeon.onrender.com) *(allow ~30s on first load for Render's free tier to spin up)*
- **Deploy Config**: [`render.yaml`](render.yaml) → [`Dockerfile.render`](Dockerfile.render)

Every push to `main` triggers an automatic redeploy on Render. The production build uses `requirements-web.txt` to keep the Docker image lean.

---

## 🛠️ Tech Stack

**Frontend**
| Tool | Role |
|------|------|
| Next.js 16 (Turbopack) | React framework + routing |
| Tailwind CSS | Utility-first styling |
| Framer Motion | UI animations + panel transitions |
| Recharts | Interactive data visualisation |
| react-map-gl | MapLibre GL JS wrapper for satellite maps |

**Backend**
| Tool | Role |
|------|------|
| FastAPI | REST API framework |
| SQLAlchemy | ORM + query builder |
| SQLite | Lightweight production database |
| NumPy | Compound scoring, seasonal decomposition, trend projection |
| ReportLab | Server-side PDF report generation |
| Google Gemini 2.0 Flash (`google-genai`) | AI-powered NLG narratives, executive summaries, and highlight bullets (template fallback if `GEMINI_API_KEY` not set) |

**Data Sources**
| Source | What it provides |
|--------|-----------------|
| GloFAS v4 (Copernicus) | Live river discharge — direct ML feature + hazard signal |
| ERA5 via Open-Meteo | Precipitation (7d/30d), ET₀, soil moisture, temperature anomaly |
| NASA MODIS (MOD13Q1) | Real satellite NDVI at 250m resolution |
| World Bank Open Data | GDP, population density — financial impact + exposure scoring |
| Open-Meteo Elevation API | DEM elevation — low-lying terrain exposure classification |
| Open-Meteo Climate Archive | Historical monthly climate data for 6-month forecasting |

**Published Methodologies**
| Framework | Applied in |
|-----------|-----------|
| INFORM Risk Index (EU JRC) | Compound multi-hazard scoring |
| JRC Depth-Damage Functions — Huizinga et al. (2017) | Financial impact by flood depth and sector |
| UNDRR Sendai Framework | Indirect cost ratios by income class |
| UNHCR Displacement Cost Brackets | Population displacement financial estimates |
| FAO-56 Penman-Monteith | Vegetation stress / ET₀ water balance |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🛰️ by [Pranav Andhale](https://github.com/PranavAndhale)

⭐ **Star this repo if you found it useful!**

</div>
