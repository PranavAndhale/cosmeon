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
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%2B%20SHAP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE)

</div>

---

## 🌍 What is Cosmeon?

Cosmeon is a **production-grade Geospatial Intelligence Platform** that transforms raw multi-modal satellite imagery into actionable risk assessments in real time. It's built for analysts, planners, and researchers who need to go beyond static maps and into dynamic, AI-driven geospatial insights.

The platform is anchored around three core **Analysis Orbs**:

| Orb | Focus | Data Source | Color |
|-----|-------|-------------|-------|
| 🌊 **Flood Risk** | Inundation extent via SAR backscatter | Sentinel-1 | Cyan |
| 🏗️ **Infrastructure Exposure** | Water-human structure intersection | Sentinel-1 + OSM | Orange |
| 🌿 **Vegetation Health** | NDVI anomaly detection | Sentinel-2 | Green |

---

## ✨ Key Features

### 🤖 Machine Learning
- **Gradient Boosting Classifier** trained on elevation, SAR backscatter, NDVI, and historical flood frequency features
- **SHAP Explainability** — every prediction comes with a full feature importance breakdown so analysts know *why* a risk level was assigned
- **6-Month Forecasting** via ARIMA-style time-series extrapolation with confidence bands
- **Anomaly Detection** using Isolation Forest for detecting out-of-distribution satellite events

### 📡 Data & Fusion
- **Multi-Sensor Fusion** — dynamically weights SAR vs Optical imagery based on real-time cloud cover conditions (Sentinel-1 dominates when cloud cover > 70%)
- **Automated Change Detection** — calculates temporal deltas between baseline and current state, quantifying km² gained/lost per period
- **Historical Trend Analysis** — 24-month lookback with monthly aggregation, rendered as interactive Recharts graphs

### 🧠 Advanced Intelligence
- **AI NLG Summaries** — integrates with `gpt-4o-mini` to generate human-readable narrative insights from raw risk vectors
- **Compound Risk Engine** — models cascading hazard interactions (Flood → Landslide → Infrastructure failure)
- **Financial Impact Simulation** — translates risk percentages to Value-at-Risk, estimated loss, and insurance premium multipliers
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
┌─────────────────────────────────────────────────────────┐
│                     COSMEON PLATFORM                    │
├──────────────────────┬──────────────────────────────────┤
│   FRONTEND (Next.js) │         BACKEND (FastAPI)        │
│                      │                                  │
│  • react-map-gl      │  • /api/regions                  │
│  • Recharts          │  • /api/trends/{id}              │
│  • Framer Motion     │  • /api/explain/{id}             │
│  • Tailwind CSS      │  • /api/forecast/{id}            │
│  • Analysis Orbs     │  • /api/nlg/summary (GPT-4o)     │
│                      │  • /api/ml/train                 │
├──────────────────────┴──────────────────────────────────┤
│               ML PIPELINE (scikit-learn)                │
│  GradientBoosting • SHAP • ARIMA • IsolationForest      │
├─────────────────────────────────────────────────────────┤
│              DATABASE (SQLite via SQLAlchemy)           │
│  regions • risk_assessments • change_events • users     │
└─────────────────────────────────────────────────────────┘
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
├── api/                    # FastAPI routes, middleware, auth
├── database/               # SQLAlchemy models, DB manager, migrations
├── ml/                     # ML models, feature engineering, SHAP utils
├── processing/             # Satellite data fetchers, change detectors
├── ingestion/              # Data ingestion pipelines
├── frontend/               # Next.js app (src/app, components, lib)
│   ├── src/app/engine/     # The Geospatial Engine UI
│   ├── src/app/page.tsx    # Landing page
│   └── src/lib/api.ts      # Typed API client
├── data/                   # Processed ML artifacts (models, scalers)
├── train_model.py          # Standalone model training script
├── seed_demo.py            # Seeds DB with demo regions + assessments
├── Dockerfile.render       # Production Dockerfile (used by Render)
├── render.yaml             # Render deployment configuration
└── requirements.txt        # Python dependencies
```

---

## 🌐 Deployment

This project is deployed on **[Render](https://render.com)** using Docker.

- **Live URL**: [https://cosmeon.onrender.com](https://cosmeon.onrender.com)
- **Deploy Config**: [`render.yaml`](render.yaml)

Every push to `main` triggers an automatic redeploy on Render.

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
| scikit-learn | Gradient Boosting, RF, IsolationForest |
| SHAP | Model explainability |
| statsmodels | ARIMA time-series forecasting |
| OpenAI SDK | NLG narrative generation |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🛰️ by [Pranav Andhale](https://github.com/PranavAndhale)

⭐ **Star this repo if you found it useful!**

</div>
