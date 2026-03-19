# COSMEON — Replace Custom Code with Pre-Trained Model APIs

## Problem
Every bug found so far has been in our CUSTOM GLUE CODE, not in the external APIs.
The more custom code we write, the more things break. This plan replaces custom
processing layers with direct pre-trained model/API outputs wherever possible.

## Research Findings — What's Actually Available

| API | Free? | Usable Today? | Auth | Replaces |
|---|---|---|---|---|
| **NASA MODIS NDVI** (ORNL DAAC) | Yes | YES | None | Fake NDVI proxy in data_fusion.py + custom veg stress |
| **Open-Meteo Flood** (historical) | Yes | YES | None | Custom weighted scoring → real return-period probability |
| **INFORM Risk Index** (CSV download) | Yes | YES | None | Custom compound risk calibration |
| **GAR Annual Avg Loss** (CSV download) | Yes | YES | None | Custom financial baseline |
| **Google Flood Hub API** | Yes | Waitlist | GCP key | Entire flood detection pipeline |
| **Copernicus CDS/GloFAS** | Yes | Registration | CDS key | Raw ensemble discharge grids |

## Audit of Current Custom Layers — What to Replace

| Layer | Bug Risk | Real Value? | Action |
|---|---|---|---|
| `data_fusion.py` (sensor fusion) | **HIGH** | NO — SAR/Sentinel-2 are SIMULATED, not real | **Rewrite: use real MODIS NDVI + keep Open-Meteo weather only** |
| `predictor.py` (ML predictor) | MEDIUM-HIGH | Partial | **Simplify: replace with GloFAS return-period exceedance probability** |
| `forecast_engine.py` (6-mo forecast) | MEDIUM-HIGH | Partial — NH-only seasonal patterns | **Fix: add hemisphere detection, use Open-Meteo Seasonal API directly** |
| `financial_impact.py` (JRC damage) | MEDIUM | Partial — custom depth mapping unvalidated | **Improve: add GAR baseline as cross-check** |
| `live_analysis.py` (detection) | MEDIUM | YES | **Improve: use return-period thresholds instead of custom scoring** |
| `compound_risk.py` (INFORM) | LOW-MEDIUM | YES — uses published formula | **Improve: add INFORM country-level lookup as calibration** |
| `model_hub.py` (data fetching) | LOW | YES — tiered fallback is solid | **Keep as-is, add NASA MODIS as new Tier 0 for vegetation** |

---

## Implementation Plan — 4 Phases

### Phase 1: Drop-in API Replacements (Highest Impact, Lowest Risk)

**Step 1.1 — NASA MODIS NDVI for real vegetation data**
- Add `get_ndvi_modis(lat, lon)` to `model_hub.py`
- API: `GET https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset`
  - Params: `latitude, longitude, startDate, endDate, band=250m_16_days_NDVI`
  - Returns: Real NDVI values (divide by 10000 for 0-1 range)
- Replace `get_vegetation_stress()` Tier 1 with MODIS NDVI (real satellite)
  - Current Tier 1: FAO-56 ET0 water-balance (computed from weather data)
  - New Tier 0: NASA MODIS real satellite NDVI → `stress = 1 - ndvi`
  - Existing tiers become Tier 1, 2, 3 as fallbacks
- Update `data_fusion.py` to use real MODIS NDVI instead of `max(0.0, 0.7 - flood_pct * 2)`
- Files: `processing/model_hub.py`, `processing/data_fusion.py`

**Step 1.2 — INFORM Risk Index country-level lookup**
- Download INFORM CSV from EU JRC (191 countries, 54 indicators)
- Store as `data/inform_risk_index.csv`
- Add `get_inform_risk(lat, lon)` to `model_hub.py`:
  - Reverse-geocode lat/lon → country (Nominatim, already used)
  - Lookup country → INFORM scores (overall risk, flood hazard, vulnerability, coping capacity)
- Use in `compound_risk.py` as calibration anchor:
  - If our computed compound score deviates >30% from INFORM country score, log warning
  - Add INFORM country score to response for transparency
- Files: `processing/model_hub.py`, `processing/compound_risk.py`, `data/inform_risk_index.csv`

**Step 1.3 — GAR Annual Average Loss baseline**
- Download UNDRR GAR risk data (expected annual loss by country + hazard type)
- Store as `data/gar_annual_loss.csv`
- Add `get_gar_loss(lat, lon)` to `model_hub.py`:
  - Reverse-geocode lat/lon → country
  - Lookup country → annual average flood loss (USD)
- Use in `financial_impact.py` as scenario baseline:
  - For scenario-based estimates (no active flood), use GAR national loss
    scaled by local area ratio instead of arbitrary 0.2m depth assumption
  - Add GAR reference to response
- Files: `processing/model_hub.py`, `processing/financial_impact.py`, `data/gar_annual_loss.csv`

### Phase 2: Real Flood Probability from Historical Discharge

**Step 2.1 — Build return-period thresholds from Open-Meteo historical data**
- For any lat/lon, fetch 40 years of historical discharge from Open-Meteo Flood API:
  `GET https://flood-api.open-meteo.com/v1/flood?latitude=X&longitude=Y&daily=river_discharge&start_date=1984-01-01&end_date=2026-03-01`
- Compute return-period thresholds:
  - 2-year return period = 50th percentile of annual maxima
  - 5-year = 80th percentile
  - 10-year = 90th percentile
  - 20-year = 95th percentile
  - 50-year = 98th percentile
- Cache thresholds per location (they don't change)

**Step 2.2 — Replace custom scoring with exceedance probability**
- Current `live_analysis.py` uses 7-factor weighted scoring (custom thresholds)
- Replace with: compare current + forecast discharge against computed return periods
  - `exceedance_prob = P(forecast > threshold)` for each return period
  - Risk level directly from return period exceedance:
    - 2yr exceeded → MEDIUM (common flood)
    - 5yr exceeded → HIGH (significant flood)
    - 20yr exceeded → CRITICAL (major flood)
- This is exactly how ECMWF/GloFAS classifies floods — not custom
- Keep rainfall/elevation/seasonal as secondary signals, but discharge return period is PRIMARY
- Files: `processing/live_analysis.py`, `processing/model_hub.py`

**Step 2.3 — Simplify ML predictor to use return-period features**
- Current predictor trains on 13 features with synthetic data
- Replace with: use return-period exceedance as the PRIMARY feature
  - Add `return_period_exceeded` (2/5/10/20/50 year) to feature vector
  - Remove synthetic training data entirely
  - Train on GloFAS ground-truth + return-period features
- This makes the ML model a thin calibration layer on top of established hydrology
- Files: `processing/predictor.py`

### Phase 3: Fix Sensor Fusion Honesty

**Step 3.1 — Remove simulated SAR/Sentinel-2 from data_fusion.py**
- Current `_simulate_sar_data()` generates fake SAR backscatter from random noise
- Current optical layer uses hardcoded `ndvi_proxy = max(0.0, 0.7 - flood_pct * 2)`
- Replace both with REAL data sources only:
  - Optical layer → NASA MODIS NDVI (from Step 1.1)
  - SAR layer → REMOVE entirely (we don't have real SAR data)
  - Thermal layer → Open-Meteo temperature (already real)
  - Weather layer → Open-Meteo soil/precip (already real)
- Rename panel from "Sensor Fusion" to "Multi-Source Analysis"
- Update frontend label and description
- Files: `processing/data_fusion.py`, `frontend/src/app/engine/page.tsx`

**Step 3.2 — Fix hemisphere-blind seasonal patterns**
- `forecast_engine.py` has hardcoded NH monsoon pattern (July/Aug peak)
- `live_analysis.py` has hardcoded NH seasonal multipliers
- Fix: detect hemisphere from latitude, flip seasonal pattern for SH
  - `if lat < 0: shift seasonal by 6 months`
- Also add equatorial detection (lat -10 to +10): use bimodal pattern
- Files: `processing/forecast_engine.py`, `processing/live_analysis.py`

### Phase 4: Google Flood Hub (When Access Granted)

**Step 4.1 — Apply to Google Flood Forecasting API waitlist**
- URL: https://developers.google.com/flood-forecasting
- Apply with GCP project ID
- Expected wait: weeks to months

**Step 4.2 — Integrate when approved**
- `GET floodStatus.searchLatestFloodStatusByArea` → severity, trend, inundation maps
- This would replace our entire custom detection pipeline for covered areas (80+ countries)
- Keep existing pipeline as fallback for uncovered regions

---

## What This Eliminates

### Custom code REMOVED:
- Fake SAR simulation (~40 lines)
- Fake Sentinel-2 NDVI proxy (~10 lines)
- Custom 7-factor weighted scoring in live_analysis.py (~80 lines)
- Synthetic training data generation in predictor.py (~120 lines)
- Arbitrary anomaly-to-depth mapping in financial_impact.py (~15 lines, replaced by GAR)

### Custom code KEPT (it's actually good):
- model_hub.py tiered fallback architecture
- compound_risk.py INFORM geometric mean formula
- JRC depth-damage piecewise interpolation tables
- Alert threshold logic

## Expected Results
- Fewer integration bugs (less custom code = less surface area for errors)
- More trustworthy numbers (established models, not custom formulas)
- Transparent data provenance (every number traced to a published source)
- Honest about capabilities (no fake sensor data)
