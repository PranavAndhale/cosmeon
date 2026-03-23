# COSMEON Design Specification — Phase 10B Features

> Reference for Gemini Pro / Antigravity or any frontend developer extending these components.
> All three features are implemented. This spec documents the **design system**, **styling tokens**, and **interaction patterns** so future changes stay visually coherent.

---

## Design Language

COSMEON is a **climate intelligence mission-control** interface. The visual identity is:

- **Dark orbital glass** — deep navy/charcoal backgrounds with frosted-glass panels
- **Surgical precision** — monospace typography, uppercase labels, wide letter-spacing
- **Signal-driven color** — color only appears when data demands attention (risk levels, signal status)
- **Layered depth** — subtle borders, radial glows, and `backdrop-blur` create z-axis hierarchy
- **No decoration for decoration's sake** — every pixel of color or animation encodes information

### Target users

1. **Field coordinators** (NDMA/SDMA, India) — need to glance at the map and instantly know where to deploy resources. Color hierarchy and pulse speed encode urgency.
2. **Data analysts / researchers** — need to understand *why* the model predicted a certain risk. Waterfall chart and signal verification panels serve this.
3. **General public** — plain language verdict and the forecast slider let non-technical users understand temporal risk progression.

---

## Core Tokens

### Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `bg-primary` | `#0B0E11` | Main canvas, deepest layer |
| `bg-panel` | `#151A22` | Card/panel backgrounds |
| `bg-inset` | `#0B1320` | Inset panels (explanation, waterfall) |
| `bg-track` | `#1a1f2e` | Slider track, progress bar backgrounds |
| `border-subtle` | `rgba(255,255,255,0.05)` | Panel borders (`border-white/5`) |
| `border-glass` | `rgba(255,255,255,0.10)` | Glass panel borders (`border-white/10`) |
| `accent-cyan` | `#00E5FF` | Primary brand accent, system healthy |
| `risk-critical` | `#ef4444` | CRITICAL risk — `red-500` |
| `risk-high` | `#f97316` | HIGH risk — `orange-500` |
| `risk-medium` | `#eab308` | MEDIUM risk — `yellow-500` |
| `risk-low` | `#22c55e` | LOW risk — `green-500` |
| `status-flooding` | `#ef4444` | FLOODING_NOW markers |
| `status-imminent` | `#f97316` | IMMINENT markers |
| `status-watch` | `#eab308` | WATCH markers |
| `status-receding` | `#22d3ee` | RECEDING markers — `cyan-400` |
| `status-normal` | `#4b5563` | NORMAL markers — `gray-600` |
| `signal-up` | `#ef4444` | Waterfall positive delta (increases risk) |
| `signal-down` | `#22d3ee` | Waterfall negative delta (decreases risk) |
| `signal-neutral` | `#4b5563` | Waterfall no-change |

### Typography

| Style | Classes | Usage |
|-------|---------|-------|
| Section header | `text-[12px] font-medium font-mono uppercase tracking-widest text-gray-500` | Panel titles |
| Data badge | `text-[10px] font-mono px-1.5 py-0.5 rounded border border-cyan-500/30 text-cyan-400` | Source tags (e.g. "GloFAS v4") |
| Body mono | `text-[11px] font-mono text-gray-400` | Data values, labels |
| Body sans | `text-[13px] font-sans text-gray-300 leading-relaxed` | Narrative text |
| Risk readout | `text-[20px] font-bold font-mono uppercase tracking-widest` | Risk level hero text |

### Shared Panel Class

```
glassClass = "bg-[#0B0E11]/70 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl"
```

Inner panels use:
```
bg-[#151A22]/80 border border-white/5 rounded-xl p-4
```

Inset/explanation panels use:
```
bg-[#0B1320]/90 border border-white/[0.06] rounded-xl p-4
```

---

## Component 1: 7-Day Forecast Progression Slider

### Layout

```
+------------------------------------------------------+
| 7-DAY DISCHARGE FORECAST               [GloFAS v4]   |  <- section header + badge
|                                                        |
|  ~~~~ mini area chart (56px height) ~~~~               |  <- recharts AreaChart
|                                                        |
|  =====[======||=============]========================  |  <- slider track
|  Today  +1d  +2d  +3d  +4d  +5d  +6d                 |  <- tick labels
|                                                        |
|  [ DAY +2 . 2026-03-25          HIGH . 62% ]          |  <- projected risk badge
+------------------------------------------------------+
```

### Styling Details

**Container**: `bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-3`

**Mini AreaChart**:
- Height: `56px` via `ResponsiveContainer`
- Stroke: risk color of currently selected day (dynamically changes)
- Fill: linearGradient from `{riskColor} @ 35% opacity` (top) to `transparent` (bottom)
- Gradient IDs: `progGrad` (registered), `progGradAdhoc` (ad-hoc) — must be unique per chart instance
- Stroke width: `1.5px`
- No dots, no grid lines — clean silhouette
- Tooltip: `background: #0B0E11`, `border: 1px solid rgba(255,255,255,0.1)`, `borderRadius: 6`, `fontSize: 10`

**Slider Track**:
- Native `<input type="range">` with custom styling
- Height: `h-1` (4px), `rounded-full`, `appearance-none`
- `accentColor`: current day's risk color
- Background: CSS `linear-gradient(to right)` split at current position
  - Left of thumb: risk color (filled)
  - Right of thumb: `#1a1f2e` (track bg)
- Thumb inherits `accentColor` from the browser — no custom thumb styling needed for cross-browser simplicity

**Tick Labels**:
- Font: `text-[9px] font-mono`
- Active day: `text-white font-bold`
- Inactive: `text-gray-600`
- First tick: "Today", rest: "+1d", "+2d", etc.

**Projected Risk Badge**:
- Container: `px-3 py-2 rounded-lg border`
- Border color: `{riskColor}40` (25% opacity of risk color)
- Background: `{riskColor}15` (8% opacity)
- Left text: `text-[11px] font-mono text-gray-400` — shows "TODAY" or "DAY +N" with date
- Right text: `text-[12px] font-mono font-bold` in risk color — shows "HIGH . 62%"

### Interaction

- Sliding updates `sliderDay` state (0-6)
- Chart gradient color, badge, and track fill all react to the selected day's risk level
- No debounce needed — state is lightweight

---

## Component 2: Risk Build-up Waterfall Chart

### Layout

```
+------------------------------------------------------+
| RISK BUILD-UP                                          |
|                                                        |
| Baseline (neutral)   |--- 14% baseline marker ---|  14%|
| GloFAS Discharge     [=========>            ]  +48%  62%|
| ERA5 Precipitation   [====>                 ]  +12%  74%|
| Soil Moisture        [==>                   ]   +6%  80%|
| Flood History        [<==                   ]   -3%  77%|
| ─────────────────────────────────────────────────────  |
| Final                                              77%  |
+------------------------------------------------------+
```

### Styling Details

**Container**: `bg-[#0B1320]/90 border border-white/[0.06] rounded-xl p-4 flex flex-col gap-3`

**Section header**: Standard — `text-[12px] font-medium text-gray-500 uppercase font-mono tracking-widest`

**Baseline row**:
- `text-[10px] font-mono text-gray-600`
- Feature label: `w-28 shrink-0 truncate`
- Track area: `flex-1 relative h-5` with horizontal hairline (`h-[1px] bg-white/5`)
- Baseline marker: vertical `h-3 w-[1px] bg-cyan-500/50` positioned at `{baseline * 100}%` from left
- Value: `w-9 text-right`

**Step rows**:
- `text-[10px] font-mono`
- Feature label: `w-28 shrink-0 truncate text-gray-400`
- Bar container: `flex-1 h-4 bg-[#0B0E11] rounded overflow-hidden`
- Bar fill: `h-full rounded transition-all duration-500`
  - Width: `min((|delta| / maxDelta) * 60, 60)%` — normalized so largest bar fills 60%
  - Color: `#ef4444` (up/positive), `#22d3ee` (down/negative), `#4b5563` (neutral)
  - Opacity: `0.85`
- Delta label: `w-12 text-right font-bold`
  - Positive: `text-red-400`, prefixed with `+`
  - Negative: `text-cyan-400`
  - Neutral: `text-gray-600`
- Cumulative: `w-9 text-right text-gray-500`

**Final row**:
- `text-[11px] font-mono font-bold`
- Separated by `mt-1 border-t border-white/5 pt-2`
- Label: `text-gray-300`
- Value: colored by final risk level via `riskColor()`

### Data Source

Backend returns waterfall in `explain_prediction()` response:
```json
{
  "waterfall": {
    "baseline_probability": 0.14,
    "steps": [
      { "feature": "GloFAS River Discharge", "delta": 0.48, "cumulative": 0.62, "direction": "up", "label": "..." },
      { "feature": "ERA5 Precipitation", "delta": 0.12, "cumulative": 0.74, "direction": "up", "label": "..." }
    ],
    "final_probability": 0.77
  }
}
```

---

## Component 3: Risk-Differentiated Map Markers

### Registered Region Markers

**Structure** (outer → inner):
```
[ Pulse ring (conditional) ]
  [ Outer glow circle (status color) ]
    [ Inner risk ring (risk color) ]
```

**Outer circle**:
- Size: `{markerSizeForStatus(status) + 6}px` — creates visual hierarchy by situation urgency
  - FLOODING_NOW: 20px, IMMINENT: 19px, WATCH: 17px, RECEDING: 16px, NORMAL: 15px
- Border: `2px solid {statusColor}` (or brand accent when selected)
- Background: `markerGlow(statusColor, intensity)`:
  ```
  radial-gradient(circle, {color}cc 0%, {color}55 45%, transparent 75%)
  boxShadow: 0 0 {spread}px {color}90, 0 0 {spread*2}px {color}30
  ```
  - `high` (FLOODING_NOW): spread = 20px
  - `medium` (IMMINENT): spread = 12px
  - `low` (WATCH/RECEDING/NORMAL): spread = 6px
- Selected override: solid `backgroundColor: mColor`, `boxShadow: 0 0 18px {mColor}`

**Inner risk ring** (visible when NOT selected):
- Size: `max(markerSize - 4, 4)px`
- Border: `1.5px solid {riskColor}80`
- Background: `{riskColor}20`
- Purpose: shows risk level independently of situation status (risk != status)

**Pulse ring** (conditional):
- Visible when: FLOODING_NOW, IMMINENT, WATCH, or selected
- Class: `absolute -inset-2 rounded-full animate-ping opacity-30`
- Border: `1px solid {statusColor}`
- **Speed varies by urgency** via `animationDuration`:
  - FLOODING_NOW: `1s` (fast, urgent — "get there now")
  - IMMINENT: `2s` (moderate — "prepare")
  - WATCH: `3s` (slow breathing — "keep an eye on it")
  - RECEDING/NORMAL: no pulse

**Label**: `text-[9px] font-mono text-gray-400` with text-shadow for map readability

### Ad-hoc Location Marker

- `w-7 h-7 rounded-full border-2`
- Border: prediction risk color (not hardcoded cyan)
- Background: `markerGlow(riskColor, 'high')` — always high intensity since user explicitly placed it
- Icon: `<MapPin size={14}>` in risk color
- Pulse: always active, `1s` duration
- Color source: `adHocData?.prediction?.predicted_risk_level` when available, falls back to `#00E5FF`

### Color Encoding Summary

| Visual Property | Encodes | Source |
|----------------|---------|--------|
| Outer border / glow | Situation status | `situation_status` field |
| Inner ring | ML risk level | `risk_level` field |
| Pulse speed | Urgency | `situation_status` field |
| Marker size | Urgency | `situation_status` field |
| Glow intensity | Urgency | `situation_status` field |

This dual-encoding lets a coordinator instantly distinguish "HIGH risk but not flooding yet (WATCH)" from "MEDIUM risk but actively flooding (FLOODING_NOW)" — they look different.

---

## Component 4: Plain Language Verdict

### Placement
Immediately after the "Prediction Explanation" header, before the technical explanation text.

### Styling
```
px-3 py-2.5 rounded-lg border border-cyan-500/20 bg-cyan-500/5
```
Text: `text-[13px] text-cyan-100 font-sans leading-relaxed`

### Purpose
A 2-3 sentence natural language summary generated by `_plain_language_verdict()`. Written for non-technical users. Example:

> "This location is at HIGH flood risk (62% probability). GloFAS v4 river discharge is the primary driver, showing a significant +1.8 sigma anomaly above seasonal norms. Elevated 7-day rainfall (89mm) and high soil saturation (72%) compound the risk."

---

## Animation & Transitions

| Element | Animation | Duration | Purpose |
|---------|-----------|----------|---------|
| Marker pulse | `animate-ping` (scale + fade) | 1s / 2s / 3s | Urgency encoding |
| Waterfall bars | `transition-all duration-500` | 500ms | Smooth reveal on load |
| Slider track fill | CSS gradient (instant) | — | Direct feedback |
| Risk cards (HIGH/CRIT) | `animate-[pulse_3s_ease-in-out_infinite]` | 3s | Subtle attention draw |
| Risk card glow | `blur-[40px]` radial blob | — | Ambient danger halo |

---

## Responsive Notes

- Sidebar panels are fixed-width (`w-[440px]`), so components don't need breakpoint variants
- Signal verification grid uses `grid-cols-2` which works at 440px
- On smaller viewports, the sidebar stacks below the map (handled by parent layout)
- Waterfall feature labels truncate at `w-28` (7rem) — sufficient for "ERA5 Precipitation" etc.

---

## Accessibility

- Risk levels are communicated via **text labels** AND **color** — never color alone
- Pulse speed differences are supplemented by status text badges
- Plain language verdict provides narrative alternative to charts
- All interactive elements (slider, buttons) use native HTML controls
- Tooltip text uses sufficient contrast (`text-gray-300` on `#0B0E11`)

---

## Integration Checklist

- [x] Forecast slider — registered regions panel (line ~1361)
- [x] Forecast slider — ad-hoc locations panel (line ~2394)
- [x] Waterfall chart — registered explain panel (line ~1529)
- [x] Waterfall chart — ad-hoc explain panel (line ~2553)
- [x] Plain language verdict — registered (line ~1519)
- [x] Plain language verdict — ad-hoc (line ~2543)
- [x] Marker glow + variable pulse — registered markers (line ~758)
- [x] Marker risk-colored — ad-hoc marker (line ~804)
- [x] Backend `compute_daily_progression()` — tiered_predictor.py
- [x] Backend `_build_waterfall()` — tiered_predictor.py
- [x] Backend `_plain_language_verdict()` — tiered_predictor.py
- [x] API routes return `progression` — routes.py
- [x] TypeScript interfaces — api.ts (`DailyRiskProgression`, `WaterfallData`)
