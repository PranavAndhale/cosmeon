# COSMEON — Situation Board + Temporal Classification (Sonnet Build Plan)

## Scope

This plan covers the **functional features** — new backend endpoints, data logic, and frontend wiring. The visual/UI styling of these features will be handled separately. Focus on making things work correctly with clean, functional UI. Do NOT spend time on visual polish — just use the existing UI patterns already in the codebase.

---

## Context

The end user is a flood risk / climate intelligence professional. They need situational awareness: which regions are flooding right now, which are about to, which are calm. Currently each region must be clicked individually to see its risk — there's no cross-region overview.

### Existing Architecture

- **Frontend**: Next.js, single file `frontend/src/app/engine/page.tsx` (~2800 lines), Tailwind CSS, framer-motion, recharts, lucide-react icons, react-map-gl/maplibre
- **Backend**: FastAPI (`api/routes.py`), global `predictor = TieredFloodPredictor()` instance
- **API client**: `frontend/src/lib/api.ts` — all fetch functions using relative URLs through Next.js proxy
- **State**: plain React `useState` — no state management library
- **The ad-hoc panel mirrors the registered panel** — any new features that apply to registered regions should also be considered for ad-hoc locations where applicable

### Key Existing Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/regions` | `{ count, regions: [{ id, name, bbox }] }` |
| `GET /api/regions/{id}/risk` | `{ risk_level, flood_percentage, confidence_score, flood_area_km2, total_area_km2, water_change_pct, timestamp }` |
| `GET /api/regions/{id}/history?limit=N` | `{ assessments: [risk_assessment...] }` |
| `GET /api/predict/{id}` | `{ predicted_risk_level, flood_probability, confidence, contributing_factors: { glofas_tier, glofas_source, precip_compound, soil_compound, glofas_primary_prob, history_compound, final_probability } }` |
| `GET /api/discharge/{id}` | `{ current_discharge_m3s, mean_discharge_m3s, anomaly_sigma, forecast_discharge[], flood_risk_level }` |
| `GET /api/explain/{id}` | `{ ml_prediction: { risk_level, probability, confidence, feature_values: { glofas_flood_risk, discharge_anomaly_sigma, precip_7d_mm, precip_anomaly, soil_saturation, mean_flood_pct, ... }, top_drivers, explanation } }` |
| `GET /api/alerts?limit=50` | Recent automated flood alerts |
| `GET /api/risk?level=HIGH` | All regions matching a risk level |

### Key Existing Patterns

- Glass card: `bg-[#0B0E11]/70 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl`
- Card bg: `bg-[#151A22]/80` or `bg-[#0A1628]/80`
- Section headers: `text-[13px] uppercase tracking-widest font-mono text-gray-500`
- Risk colors: `riskColor()` function — CRITICAL=#ef4444, HIGH=#f97316, MEDIUM=#eab308, LOW=#22c55e
- `textMono = "font-mono tracking-tight"`

---

## Feature 1: Backend — `GET /api/situation/all`

### Purpose

Single endpoint that returns all monitored regions with their current risk level, temporal situation status, trend direction, and key contributing factors. Eliminates N+1 fetches from the frontend.

### Implementation in `api/routes.py`

#### Step 1: Add the temporal classification helper

Place this above the endpoint definition:

```python
def _classify_situation(
    risk_level: str,
    prev_risk_level: str | None,
    discharge_anomaly: float,
    precip_anomaly: float,
    soil_saturation: float,
) -> str:
    """
    Classify the operational flood situation state.

    Returns one of: FLOODING_NOW, IMMINENT, WATCH, RECEDING, NORMAL
    """
    high_levels = {"HIGH", "CRITICAL"}

    # Active flooding: currently high risk AND was high before AND discharge elevated
    if risk_level in high_levels and prev_risk_level in high_levels and discharge_anomaly > 1.5:
        return "FLOODING_NOW"

    # Imminent: currently high but just escalated, or discharge rising
    if risk_level in high_levels and (prev_risk_level not in high_levels or discharge_anomaly > 1.0):
        return "IMMINENT"

    # Receding: was high, now dropped
    if risk_level not in high_levels and prev_risk_level in high_levels:
        return "RECEDING"

    # Watch: medium with building conditions
    if risk_level == "MEDIUM" and (precip_anomaly > 0.5 or soil_saturation > 0.5):
        return "WATCH"

    return "NORMAL"
```

#### Step 2: Add the endpoint

```python
@app.get("/api/situation/all")
def get_situation_all():
    """Cross-region situation overview with temporal state classification."""
    regions = db.get_all_regions()
    results = []
    summary = {"flooding_now": 0, "imminent": 0, "watch": 0, "receding": 0, "normal": 0, "total": len(regions)}

    for region in regions:
        risk = db.get_latest_risk(region.id)
        if not risk:
            results.append({
                "id": region.id,
                "name": region.name,
                "bbox": region.bbox,
                "risk_level": "UNKNOWN",
                "situation_status": "NORMAL",
                "flood_area_km2": 0,
                "flood_percentage": 0,
                "discharge_anomaly_sigma": 0,
                "trend": "stable",
                "last_assessed": None,
                "contributing_factors": {},
            })
            summary["normal"] += 1
            continue

        # Get previous risk for trend
        history = db.get_risk_history(region.id, limit=2)
        prev_risk_level = history[1].risk_level if len(history) >= 2 else None

        # Determine trend
        if prev_risk_level:
            risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            curr_val = risk_order.get(risk.risk_level, 0)
            prev_val = risk_order.get(prev_risk_level, 0)
            trend = "escalating" if curr_val > prev_val else "improving" if curr_val < prev_val else "stable"
        else:
            trend = "stable"

        # Get prediction for contributing factors (with fallback)
        contributing_factors = {}
        discharge_anomaly = 0.0
        precip_anomaly = 0.0
        soil_saturation = 0.0
        try:
            bbox = region.bbox
            lat = (bbox[1] + bbox[3]) / 2
            lon = (bbox[0] + bbox[2]) / 2
            pred = predictor.predict(
                [h.to_dict() for h in history],
                {"_lat": lat, "_lon": lon},
                region.name,
            )
            if pred:
                pd = pred.to_dict()
                contributing_factors = pd.get("contributing_factors", {})
                # Extract signals for classification
                # These come from the compound scorer
                discharge_anomaly = contributing_factors.get("glofas_primary_prob", 0)
                # Map primary prob to approximate anomaly: >0.6 means anomaly > 1.5
                discharge_anomaly = (contributing_factors.get("glofas_primary_prob", 0.07) - 0.07) / 0.15
        except Exception:
            pass

        # For more accurate discharge_anomaly, try the explain endpoint data
        # but fall back to the approximation above if unavailable
        try:
            bbox = region.bbox
            lat = (bbox[1] + bbox[3]) / 2
            lon = (bbox[0] + bbox[2]) / 2
            expl = predictor.explain_prediction(
                [h.to_dict() for h in history],
                {"_lat": lat, "_lon": lon},
                region.name,
            )
            if expl and "feature_values" in expl:
                fv = expl["feature_values"]
                discharge_anomaly = fv.get("discharge_anomaly_sigma", discharge_anomaly)
                precip_anomaly = fv.get("precip_anomaly", precip_anomaly)
                soil_saturation = fv.get("soil_saturation", soil_saturation)
                contributing_factors["discharge_anomaly_sigma"] = discharge_anomaly
                contributing_factors["precip_anomaly"] = precip_anomaly
                contributing_factors["soil_saturation"] = soil_saturation
        except Exception:
            pass

        situation = _classify_situation(
            risk.risk_level, prev_risk_level,
            discharge_anomaly, precip_anomaly, soil_saturation,
        )

        summary[situation.lower()] = summary.get(situation.lower(), 0) + 1

        results.append({
            "id": region.id,
            "name": region.name,
            "bbox": region.bbox,
            "risk_level": risk.risk_level,
            "situation_status": situation,
            "flood_area_km2": risk.flood_area_km2,
            "flood_percentage": risk.flood_percentage,
            "confidence_score": risk.confidence_score,
            "discharge_anomaly_sigma": round(discharge_anomaly, 2),
            "trend": trend,
            "last_assessed": risk.timestamp.isoformat() if hasattr(risk, 'timestamp') and risk.timestamp else str(risk.timestamp) if risk.timestamp else None,
            "contributing_factors": contributing_factors,
        })

    # Sort: CRITICAL first, then HIGH, then by flood_area descending
    risk_sort = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    results.sort(key=lambda r: (risk_sort.get(r["risk_level"], 4), -r["flood_area_km2"]))

    return {"regions": results, "summary": summary}
```

**Important**: The `explain_prediction()` call makes 3 external API calls (GloFAS + ERA5 + soil moisture) per region. For 10+ regions this can take 30-60 seconds. Consider:
- Adding a `timeout` parameter to each external call in `tiered_predictor.py`
- Caching results (the predictor could cache per lat/lon for 30 minutes)
- For the initial build, it's acceptable to be slow — optimize later with the existing scheduler

### Step 3: Add frontend API function

In `frontend/src/lib/api.ts`, add:

```typescript
export interface SituationRegion {
    id: number;
    name: string;
    bbox: number[];
    risk_level: string;
    situation_status: 'FLOODING_NOW' | 'IMMINENT' | 'WATCH' | 'RECEDING' | 'NORMAL';
    flood_area_km2: number;
    flood_percentage: number;
    confidence_score?: number;
    discharge_anomaly_sigma: number;
    trend: 'escalating' | 'stable' | 'improving';
    last_assessed: string | null;
    contributing_factors: Record<string, number>;
}

export interface SituationData {
    regions: SituationRegion[];
    summary: {
        flooding_now: number;
        imminent: number;
        watch: number;
        receding: number;
        normal: number;
        total: number;
    };
}

export async function fetchSituation(): Promise<SituationData | null> {
    return safeFetch(`${API}/situation/all`);
}
```

---

## Feature 2: Situation Board (Left Sidebar)

### Where It Goes

In `page.tsx`, the left sidebar currently has:
1. Search bar
2. Region list (clickable region names)
3. Orb switcher

The Situation Board **replaces the existing region list** with a richer, ranked version. Instead of plain text region names, each row becomes a mini status card.

### State

Add at the top of the component:

```typescript
const [situationData, setSituationData] = useState<SituationData | null>(null);
const [situationLoading, setSituationLoading] = useState(false);
```

### Data Fetching

Add a `useEffect` that fetches on mount and refreshes every 5 minutes:

```typescript
useEffect(() => {
    const load = async () => {
        setSituationLoading(true);
        const data = await fetchSituation();
        if (data) setSituationData(data);
        setSituationLoading(false);
    };
    load();
    const interval = setInterval(load, 5 * 60 * 1000); // refresh every 5 min
    return () => clearInterval(interval);
}, []);
```

### UI Structure

Find the existing region list in the left sidebar and replace it with:

```
SITUATION BOARD                    [2 active / 8]

[severity bar ============================]
Bihar, India                    CRITICAL  ▲
  324 km2 flooded · +3.2σ      [ACTIVE FLOODING]
  Updated 14m ago

[severity bar ====================--------]
Dhaka Division                  HIGH      ▲
  187 km2 flooded · +1.8σ      [IMMINENT]
  Updated 22m ago

[severity bar ==========------------------]
Mekong Delta                    MEDIUM    →
  42 km2 · +0.6σ               [WATCH]
  Updated 1h ago

[severity bar ----]
Lake Chad Basin                 LOW       ↓
  0 km2 · -0.3σ                [NORMAL]
  Updated 3h ago
```

**Per-row elements**:

1. **Severity bar** (top of each row): 3px height, full width, colored by risk level, fill width proportional to `flood_percentage` (capped at some reasonable max like 20% = full). Use existing `riskColor()`.

2. **Row 1**: Region name (left, `text-[13px] font-mono text-gray-200`) + Risk badge (right, small colored pill) + Trend arrow (▲/→/▼, colored red/gray/green).

3. **Row 2**: `{flood_area_km2} km2 flooded · {discharge_anomaly_sigma > 0 ? '+' : ''}{discharge_anomaly_sigma}σ` (left, `text-[11px] font-mono text-gray-500`) + Situation badge (right).

4. **Row 3**: `Updated {relative time}` — `text-[10px] font-mono text-gray-600`.

5. **Situation badge**: Colored pill showing the temporal state label:
   - `FLOODING_NOW` → `bg-red-500/20 border-red-500/40 text-red-400` label "ACTIVE FLOODING" with `animate-pulse`
   - `IMMINENT` → `bg-orange-500/20 border-orange-500/40 text-orange-400` label "IMMINENT"
   - `WATCH` → `bg-yellow-500/20 border-yellow-500/40 text-yellow-400` label "WATCH"
   - `RECEDING` → `bg-cyan-500/20 border-cyan-500/40 text-cyan-400` label "RECEDING"
   - `NORMAL` → `bg-gray-500/10 border-gray-600/30 text-gray-500` label "NORMAL"

6. **Click handler**: Clicking a row sets `selectedRegion` to that region (same as current region list behavior).

7. **Header**: "SITUATION BOARD" in standard section header style + count badge showing `{flooding_now + imminent} active / {total}`.

8. **Loading state**: While `situationLoading` and no data, show a skeleton (3-4 rows of `bg-gray-700 animate-pulse rounded` divs).

### Relative Time Helper

Add a helper function at the top of the file:

```typescript
function timeAgo(isoDate: string | null): string {
    if (!isoDate) return 'Never';
    const diff = Date.now() - new Date(isoDate).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}
```

---

## Feature 3: Temporal State in Right Panel Header

### Where It Appears

In the right panel sticky header (both registered and ad-hoc views), add the situation status badge below the risk level line.

### For Registered Regions

The temporal state comes from `situationData` — find the matching region:

```typescript
const situationStatus = situationData?.regions.find(r => r.id === selectedRegion.id)?.situation_status ?? null;
```

Add below the risk level line in the header:

```jsx
{situationStatus && situationStatus !== 'NORMAL' && (
    <span className={`text-[10px] font-mono px-2 py-0.5 rounded border mt-1 inline-block ${
        situationStatus === 'FLOODING_NOW' ? 'bg-red-500/20 border-red-500/40 text-red-400 animate-pulse'
        : situationStatus === 'IMMINENT' ? 'bg-orange-500/20 border-orange-500/40 text-orange-400'
        : situationStatus === 'WATCH' ? 'bg-yellow-500/20 border-yellow-500/40 text-yellow-400'
        : situationStatus === 'RECEDING' ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-400'
        : 'bg-gray-500/10 border-gray-600/30 text-gray-500'
    }`}>
        {situationStatus === 'FLOODING_NOW' ? 'ACTIVE FLOODING'
         : situationStatus === 'IMMINENT' ? 'IMMINENT RISK'
         : situationStatus}
    </span>
)}
```

### For Ad-Hoc Locations

Ad-hoc locations won't be in the situation data (they're not registered regions). Skip the temporal badge for ad-hoc — or compute it client-side from the prediction data if available.

---

## Feature 4: Map Marker Enhancements

### Current State

Map markers are rendered for each region as colored dots. Find where markers are rendered in the map section of `page.tsx`.

### Changes

1. **Color markers by situation status** instead of just risk level. Use the `situationData` to look up each region's status.

2. **Pulse animation** for `FLOODING_NOW` markers: wrap the marker in a `<div>` with a CSS keyframe animation that creates a pulsing glow effect.

3. **Size by severity**: CRITICAL/FLOODING_NOW = 14px diameter, HIGH/IMMINENT = 12px, MEDIUM/WATCH = 10px, LOW/NORMAL = 8px.

4. **Mini label**: Add the region name as a small label below each marker using `text-[9px] font-mono text-gray-400 text-center`. Only show labels at zoom levels where they won't overlap (check map zoom state).

### Marker Color Mapping

```typescript
function markerColor(status: string): string {
    switch (status) {
        case 'FLOODING_NOW': return '#ef4444'; // red
        case 'IMMINENT': return '#f97316';     // orange
        case 'WATCH': return '#eab308';        // yellow
        case 'RECEDING': return '#22d3ee';     // cyan
        default: return '#4b5563';             // gray
    }
}
```

---

## Files to Modify

| File | What to do |
|------|------------|
| `api/routes.py` | Add `_classify_situation()` helper and `GET /api/situation/all` endpoint |
| `frontend/src/lib/api.ts` | Add `SituationRegion`, `SituationData` interfaces and `fetchSituation()` function |
| `frontend/src/app/engine/page.tsx` | Add `situationData`/`situationLoading` state, `timeAgo()` helper, situation board in left sidebar, temporal badge in right panel header, map marker enhancements |

## Implementation Order

1. Backend endpoint first — add `_classify_situation()` + `GET /api/situation/all` to `routes.py`
2. API client — add types and fetch function to `api.ts`
3. Situation Board — replace region list in left sidebar
4. Temporal badge — add to right panel header
5. Map markers — color/size/animate by situation status
6. Test: verify the endpoint returns correct data, the board renders, clicking rows opens the right panel

## Important Notes

- The entire frontend is a single `page.tsx` file. Do NOT create separate component files.
- Use existing UI patterns (glass cards, section headers, risk colors) — do not redesign the visual style.
- The `predictor.explain_prediction()` call in the situation endpoint makes 3 external API calls per region. For initial build this is fine. If it's too slow (>30s for all regions), fall back to using just DB data (risk_level + history trend) without discharge_anomaly for the classification.
- Always sort situation board by severity (CRITICAL first).
- Refresh situation data every 5 minutes.
