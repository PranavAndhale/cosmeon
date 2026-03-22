# COSMEON — Right Panel UI/UX Overhaul (Gemini Build Plan)

## Target Audience

The person using this dashboard is a **flood risk professional** — a hydrologist, disaster management officer, insurance climate risk analyst, or government emergency services coordinator. They spend 6-8 hours a day looking at dashboards like the European Flood Awareness System (EFAS), FEWS NET, ReliefWeb, or FloodList. Their expectations:

- **Data density over decoration** — they want to see more information in less space, not pretty illustrations
- **Glanceability** — scanning the panel for 2 seconds should tell them: is this region in trouble, how bad, what's driving it
- **Verifiability** — every number should name its source. They will cross-check against GloFAS, ERA5, or their own station data
- **Professional authority** — the dashboard should feel like a tool built for operators, not a consumer weather app. Think Bloomberg Terminal, EFAS, or a military C2 dashboard: dark, high-contrast, information-dense, every pixel earns its space
- **Subtlety over flashiness** — no loud animations, no gratuitous gradients, no rounded-everything-softness. Sharp, purposeful, quiet confidence

The design language: **mission control for water**.

---

## Current State of the Right Panel

The right panel is a 400px-wide absolute-positioned sidebar (`frontend/src/app/engine/page.tsx`, ~2800 lines) rendered when a region or ad-hoc location is selected. It uses Tailwind CSS inline classes, framer-motion for transitions, recharts for charts, and lucide-react for icons.

### Current Visual Problems

1. **No information hierarchy** — section labels (text-[13px]), data values (text-[14px]), and body text (text-[13px]) are all nearly the same size. Nothing stands out. A professional can't scan it quickly.

2. **Generic styling** — every card uses the same `bg-[#151A22]/80 border border-white/5 rounded-xl p-4`. There's no visual differentiation between "this is critical data" and "this is a footnote."

3. **Text-wall presentation** — Assessment Details is 7 rows of key:value text. Signal Verification is a 2x2 grid of text cards. Contributing Factors is a list of text + thin bars. It reads like a developer debug panel, not an intelligence briefing.

4. **Overloaded cyan** — cyan is used for: GloFAS features, orb accent color, multi-source panel, historical trends, signal verification "low" badge, and the download button hover state. It means everything and therefore nothing.

5. **Missing visual cues** — numbers like "3.4%" and "+2.3σ" are plain white/gray text. A professional dashboard would color-code these contextually (red if dangerous, green if safe) so the user can scan colors without reading numbers.

6. **Collapsible panels are mystery boxes** — each panel shows only an icon and title. The user has to click to find out what's inside. No subtitle, no preview.

### Current Structure (top to bottom, for both registered and ad-hoc views)

1. **Sticky header** — risk level badge + region name + orb title + close button
2. **Metric cards** (2-col grid) — Area Affected (km2) + Data Quality (%)
3. **Assessment Details** — 6-7 key:value data rows (flood coverage, water change, soil saturation, data source, etc.)
4. **ML Prediction block** — predicted risk level, flood probability, model confidence (3 rows)
5. **"Explain Prediction" button** — triggers:
   - Prediction Explanation card (text paragraph + source line)
   - Signal Verification panel (2x2 grid of source cards with Elevated/Normal/Low badges + consensus count)
   - Contributing Factors (list of 9 features with importance bars)
6. **Risk History Mini Chart** — small area chart
7. **Historical Trend Analysis** (collapsible button) — opens a separate modal
8. **6-Month Forecast** (collapsible) — trend badge + area chart + 3-col monthly cards
9. **AI Insights** (collapsible) — NLG narrative + bullet highlights + trend narrative + engine badge
10. **Multi-Source Analysis** (collapsible) — 5 sensor fusion metric bars + weights + cloud note
11. **Compound Risk** (collapsible) — score + hazard layers + interactions + recommendations
12. **Financial Impact** (collapsible) — total exposure + breakdown + population + mitigation ROI
13. **Model Feedback** (collapsible) — thumbs up/down buttons
14. **Sticky footer** — "Download Structured Report" button

### Current Exact Styling Patterns

These are the exact Tailwind patterns currently used. Preserve the overall dark theme but refine within it:

- **Glass card wrapper**: `bg-[#0B0E11]/70 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl`
- **Card backgrounds**: `bg-[#151A22]/80` (dark slate cards) and `bg-[#0A1628]/80` (deep navy, used for collapsible panels)
- **Signal Verification bg**: `bg-[#0B1320]/90`
- **Section header text**: `text-[13px] uppercase tracking-widest font-mono` in `text-gray-500`
- **Data row text**: `text-[14px] font-mono` values with dynamic color, `text-[13px]` labels
- **Large metric text**: `text-xl font-bold font-mono`
- **Micro/source text**: `text-[11px] font-mono text-gray-600`
- **Nano badges**: `text-[10px] font-mono px-1.5 py-0.5 rounded border`
- **Collapsible panel accent borders**: cyan-500/20, violet-500/20, amber-500/20, rose-500/20, emerald-500/20, gray-500/20
- **Progress bars**: `h-2 bg-[#0B0E11] rounded-full` with colored fill
- **`textMono`**: `font-mono tracking-tight` (utility variable used throughout)
- **Risk color function**: CRITICAL=#ef4444, HIGH=#f97316, MEDIUM=#eab308, LOW=#22c55e
- **Orb colors**: Flood=#00E5FF, Infra=#f97316, Veg=#20E251
- **Framer Motion**: `motion.div` with `initial={{ height: 0, opacity: 0 }}` / `animate={{ height: 'auto', opacity: 1 }}` for collapsible panels

---

## Design Changes

All changes below are **purely visual/UX** — no new API calls, no new data fetching, no new endpoints. Everything uses data that is already loaded and available in the component's state.

The ad-hoc location panel mirrors the registered-region panel — apply the same visual changes to both.

---

### 1. Header Redesign

**Goal**: Make the header a high-impact status strip that communicates severity in 1 second.

**Current**:
- Risk level: `text-sm uppercase tracking-widest font-bold` with `style={{ color: riskColor() }}`
- AlertTriangle icon at 16px
- Pill badge showing change_type
- Region name on next line: `text-[15px] font-semibold`

**Changes**:
- Bump risk level text to `text-[16px] font-bold font-mono uppercase tracking-widest`
- Add a subtle `drop-shadow` or `text-shadow` effect on the risk text using the risk color at 40% opacity. In Tailwind, use inline `style={{ textShadow: '0 0 12px ${riskColor(level)}66' }}` — this creates a soft glow that makes the risk level pop against the dark background
- The AlertTriangle icon: for CRITICAL and HIGH, add `animate-pulse` with a slow cycle. For MEDIUM/LOW, keep it static
- Add a relative timestamp to the header — currently buried in Assessment Details: `Last assessed: 14m ago` in `text-[10px] text-gray-600 font-mono` aligned right on the same line as the region name
- If a temporal state badge exists (from the situation board feature, may or may not be present), it should sit between the risk level and the region name as a small colored pill

**The header should feel like a status bar at the top of a military terminal — compact, high-contrast, immediately communicating threat level.**

---

### 2. Metric Cards — 3-Card Row with Mini Bars

**Goal**: Show the 3 most important numbers prominently with visual context.

**Current**: 2-column grid with "Area Affected" and "Data Quality". Values in `text-xl font-bold`.

**Changes**: Replace with a 3-column grid:

| Card | Label | Value source | Bar meaning |
|------|-------|-------------|-------------|
| Flood Area | "FLOOD AREA" | `flood_area_km2` from latestRisk | Bar width = `flood_percentage`, capped display at ~20% = full width. Color = risk color. Below bar: `{flood_percentage:.1%}` in `text-[10px]` |
| Probability | "RISK PROB." | `flood_probability` from prediction | Bar width = flood_probability (0-1). Color = risk color of `predicted_risk_level`. Below bar: risk level label in `text-[10px]` |
| Confidence | "CONFIDENCE" | `confidence` from prediction | Bar width = confidence. Color = #22d3ee (cyan, always). Below bar: `T{glofas_tier}` from `contributing_factors.glofas_tier` in `text-[10px] text-cyan-400` |

**Card anatomy**:
```
+------------------+
| FLOOD AREA       |  <-- text-[10px] uppercase tracking-widest text-gray-500 font-mono
| 324 km2          |  <-- text-[18px] font-bold font-mono text-white
| [========--] 3.4%|  <-- 3px bar + text-[10px] label
+------------------+
```

- Card bg: `bg-[#151A22]/80 rounded-xl p-3`
- If prediction is not yet loaded, cards 2 and 3 show `--` in `text-gray-600` with `opacity-50`
- The bar: `h-[3px] rounded-full bg-[#1a1f2e]` container, fill with `transition-all duration-700` for smooth animation on load

---

### 3. Assessment Details — 2-Column Grid with Color-Coded Values

**Goal**: Make the data grid scannable by color. A professional should be able to look at the grid and immediately see which values are abnormal (red/orange) vs normal (green/gray).

**Current**: Single column of key:value rows separated by `border-b border-white/5`.

**Changes**: 2-column grid (`grid grid-cols-2 gap-4`), each cell is a self-contained metric:

```
+------------------------+  +------------------------+
| FLOOD COVERAGE         |  | WATER CHANGE           |
| 3.4%                   |  | +12.4%                 |
| [====------]           |  | ▲ expanding            |
+------------------------+  +------------------------+
| SOIL SATURATION        |  | DISCHARGE ANOMALY      |
| 78%                    |  | +2.3σ                  |
| [========--]           |  | [==========-]          |
+------------------------+  +------------------------+
| TOTAL AREA             |  | LAST ANALYZED          |
| 18,420 km2             |  | 14 min ago             |
+------------------------+  +------------------------+
```

**Cell anatomy**:
- Label: `text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1`
- Value: `text-[16px] font-bold font-mono` — COLOR RULES:
  - Flood Coverage: `>10%` = text-red-400, `>5%` = text-orange-400, `>2%` = text-yellow-400, else text-emerald-400
  - Water Change: positive = text-red-400 (water expanding is bad), negative = text-emerald-400
  - Soil Saturation: `>80%` = text-red-400, `>60%` = text-orange-400, `>40%` = text-yellow-400, else text-emerald-400
  - Discharge Anomaly: `>2σ` = text-red-400, `>1σ` = text-orange-400, `>0.5σ` = text-yellow-400, else text-emerald-400
  - Total Area: text-white (neutral)
  - Last Analyzed: text-gray-400 (neutral)
- Optional mini bar (for percentage/ratio values): `h-[2px] rounded-full mt-1`, same color as value
- Optional trend indicator (for Water Change): "▲ expanding" / "▼ receding" / "→ stable" in `text-[10px]`

**Cell container**: `bg-[#151A22]/60 rounded-lg p-3` — slightly lighter than current cards to distinguish from the outer panel, with `hover:bg-[#151A22]/80 transition-colors` for subtle interactivity.

**Data source badge**: In the section header, add a small badge: `[ERA5 + GloFAS]` in `text-[10px] font-mono px-2 py-0.5 rounded border border-white/5 text-gray-600`. This tells the user at a glance where the data comes from.

**For infra/veg orbs**: The grid should show orb-specific metrics in the same 2-column format. Infra: Exposure Score, Soil Saturation, Population Density, Infrastructure Risk. Veg: Stress Index, ET0, Precipitation, NDVI Anomaly.

---

### 4. ML Prediction Block Polish

**Goal**: Make the prediction block feel more authoritative.

**Current**: 3 rows of key:value with an `Activity` icon header.

**Changes**:
- Make the predicted risk level the hero element: `text-[20px] font-bold font-mono` colored by `riskColor()`, with the same soft glow effect as the header (`textShadow`)
- Below it, a single line: `{flood_probability:.0%} probability · {confidence:.0%} confidence` in `text-[13px] font-mono text-gray-400`
- Remove the 3-row table format — condense into 2 lines max
- Add `model_version` as a subtle footer: `text-[9px] text-gray-600 font-mono`
- The card border should pulse subtly for HIGH/CRITICAL predictions — `border` with risk color at 30% opacity

---

### 5. Signal Verification Panel Polish

**Goal**: Make this panel visually distinct — it's the "trust layer" of the platform. It should look different from regular data cards because it serves a different purpose (verification, not information).

**Current**: `bg-[#0B1320]/90 border border-white/10 rounded-xl p-4`. 2x2 grid of signal cards with status badges. Consensus bar at bottom.

**Changes**:

**5a. Card left-border accent**: Each signal card gets a 3px left border colored by its status:
- `↑ Elevated`: `border-l-[3px] border-l-red-500`
- `→ Normal`: `border-l-[3px] border-l-gray-600`
- `↓ Low`: `border-l-[3px] border-l-cyan-500`

This creates an instant visual scan pattern — the user can glance at the left edges and see red/red/red/gray = 3 elevated signals.

**5b. Severity intensity bar**: Add a thin bar (2px) inside each card showing the signal's intensity:
- GloFAS: bar width = `glofasIdx / 3 * 100%` (0-3 scale, 3 = CRITICAL)
- ERA5 Precip: bar width = `min(abs(precipAnom) / 3 * 100, 100)%`
- Soil: bar width = `soil * 100%`
- Historical: bar width = `min(hist / 20 * 100, 100)%`
Bar color matches status (red/gray/cyan).

**5c. Elevated badge**: Make the `↑ Elevated` badge more prominent — use `bg-red-500/20 border-red-500/40` instead of `bg-red-500/10 border-red-500/30`. The elevated state should feel urgent.

**5d. Consensus bar**: Make it slightly taller (`py-2.5`), and for the "well-supported" case, add a subtle border glow: `shadow-[0_0_8px_rgba(16,185,129,0.15)]` (emerald glow). For the "treat with caution" case: `shadow-[0_0_8px_rgba(249,115,22,0.15)]` (orange glow).

**5e. Panel border**: Change from `border-white/10` to `border-white/[0.06]` and add a subtle inner glow at the top: a 1px `border-t` in `border-white/[0.08]`. This makes the panel feel like it has depth — a "command panel" look.

---

### 6. Contributing Factors — "Prediction Drivers" Reframe

**Goal**: Make this section feel like an intelligence briefing, not a debug log.

**Current**: "All Contributing Factors" header, feature names in text, importance % + thin bars, influence text below each.

**Changes**:

**6a. Rename**: "PREDICTION DRIVERS" — subtitle: "Ranked by signal strength" in `text-[10px] text-gray-600 font-mono`

**6b. Semantic icons**: Add a small icon before each feature name using lucide-react:
- `glofas_flood_risk`, `discharge_*`, `forecast_*` → `Droplets` icon (water/river)
- `precip_*` → `CloudRain` icon
- `soil_saturation` → `Mountain` icon (or `Layers`)
- `mean_flood_pct` → `BarChart3` icon
Icon size: 12px, same color as the feature name text.

Import these icons at the top of the file alongside existing imports:
```typescript
import { ..., Droplets, CloudRain, Mountain, BarChart3 } from "lucide-react";
```

**6c. Inline raw value**: Add the actual signal value next to the feature name, before the importance bar:
```
[droplet] discharge anomaly sigma    +2.3σ     [========] 22%
          River discharge 2.3σ above seasonal mean — extreme
```
The value (`+2.3σ`) sits right-aligned in `text-[12px] font-mono text-gray-400`, between the feature name and the importance bar.

**6d. Bar upgrade**: Increase height to `h-2.5`. Add `transition-all duration-500` for smooth render. For GloFAS features with importance > 15%, add a subtle glow: `shadow-[0_0_4px_rgba(34,211,238,0.3)]`.

**6e. Influence text**: Reduce to `text-[11px] text-gray-600` (slightly more muted than current `text-[12px] text-gray-500`).

---

### 7. AI Insights Panel — Structured Narrative

**Goal**: Break the wall of text into distinct visual sections so the user can jump to what they need.

**Current**: Single narrative blob + bullet list + trend text + engine badge.

**Changes**:

The NLG endpoint already returns 3 separate fields: `narrative`, `highlights[]`, and `trend_narrative`. Render each with a visual sub-section header:

```
SITUATION
{narrative text with **bold** parsing}

KEY FINDINGS
— {highlight 1}
— {highlight 2}
— {highlight 3}

TREND
{trend narrative with **bold** parsing}

Template · 2:34 PM
```

**Sub-section headers**: `text-[10px] uppercase tracking-widest text-amber-400/50 font-mono mb-1.5 mt-3` (first one has no `mt-3`). The amber at 50% opacity makes them visible but not dominant.

**Highlight bullets**: Change from `▸` (amber-400) to `—` (em-dash) in `text-amber-500/60`. Slightly more subtle and professional.

**Engine badge**: Shrink to `text-[9px] text-gray-700 font-mono`. The engine info is footnote-level — it should be the quietest element in the panel.

**Source badge in header**: Add `[GloFAS v4 + ERA5]` badge next to the "AI Insights" title in the collapsible button, styled as `text-[9px] font-mono px-1.5 py-0.5 rounded border border-amber-500/15 text-amber-400/50 ml-2`.

---

### 8. Collapsible Panel Subtitles

**Goal**: Tell the user what each panel contains before they click.

**Current**: Each collapsible toggle shows: `[icon] PANEL TITLE [chevron]`

**Change**: Add a subtitle line below each title, inside the toggle button:

| Panel | Icon | Title Color | Subtitle |
|-------|------|-------------|----------|
| 6-Month Forecast | TrendingUp | violet-300 | "Projected risk trajectory — ERA5 + GloFAS" |
| AI Insights | Sparkles | amber-300 | "Natural language situation analysis" |
| Multi-Source Analysis | Radio | cyan-300 | "Sensor fusion: optical, thermal, SAR, soil" |
| Compound Risk | Shield | rose-300 | "Multi-hazard cascading risk assessment" |
| Financial Impact | DollarSign | emerald-300 | "Economic exposure and mitigation ROI" |
| Model Feedback | ThumbsUp | gray-400 | "Rate prediction accuracy" |

Subtitle styling: `text-[10px] text-gray-600 font-mono mt-0.5` on a new line inside the button's flex column.

The button layout changes from horizontal to:
```jsx
<button className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors">
    <div className="flex flex-col items-start">
        <span className="text-[13px] uppercase tracking-widest font-mono text-{color} flex items-center gap-2">
            <Icon size={14} /> {title}
        </span>
        <span className="text-[10px] text-gray-600 font-mono mt-0.5">{subtitle}</span>
    </div>
    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${open ? 'rotate-90' : ''}`} />
</button>
```

---

### 9. Forecast Panel — Compact Bar Chart + Peak Alert

**Current**: 3-column grid of monthly cards showing percentage + risk level text.

**Changes**:

**9a. Peak Alert Banner**: If `forecastData.summary.peak_probability > 0.6`, add a banner at the top of the expanded panel:
```jsx
<div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-[11px] font-mono text-red-300">
    Peak risk expected in {peak_risk_month}: {(peak_probability * 100).toFixed(0)}%
</div>
```
This is the single most actionable insight from the forecast — surface it prominently.

**9b. Monthly Cards**: Change from 3-column to 6-column (`grid grid-cols-6 gap-1`). Each card becomes narrower:
```
+------+
| MAR  |  <-- text-[9px] text-gray-500 font-mono
| ████ |  <-- vertical bar, height proportional to risk_probability (max 40px)
| ████ |
|  62% |  <-- text-[11px] font-bold font-mono, colored by risk level
| HIGH |  <-- text-[8px] text-gray-600
+------+
```
Card: `bg-[#151A22] rounded-lg p-1.5 flex flex-col items-center gap-0.5`
Vertical bar: `w-3 rounded-t-sm` with `height: {prob * 40}px`, colored by risk level. This creates a mini bar chart inline.

---

### 10. Typography Scale

Apply this strict type scale across the entire right panel. Where current code uses sizes outside this scale, adjust to the nearest match:

| Role | Tailwind Class | Example |
|------|---------------|---------|
| Hero metric | `text-[20px] font-bold font-mono` | Predicted risk level in ML block |
| Primary value | `text-[18px] font-bold font-mono` | Metric card values (324 km2) |
| Secondary value | `text-[16px] font-bold font-mono` | Assessment detail values (3.4%) |
| Standard value | `text-[14px] font-semibold font-mono` | Table values, prediction rows |
| Body text | `text-[13px] font-sans text-gray-300 leading-relaxed` | AI Insights narrative |
| Section title | `text-[12px] font-medium font-mono uppercase tracking-widest text-gray-500` | "ASSESSMENT DETAILS" |
| Subtitle / label | `text-[10px] font-mono text-gray-500 uppercase tracking-wider` | Card labels ("FLOOD AREA") |
| Caption / source | `text-[10px] font-mono text-gray-600` | "ERA5 + GloFAS", timestamps |
| Footnote | `text-[9px] font-mono text-gray-700` | Engine badge, model version |

**Key principle**: Labels small (10px), values large (16-20px). Currently both hover around 13-14px which kills hierarchy.

---

### 11. Sticky Footer — Share Button

**Current**: Single "Download Structured Report" `<a>` tag.

**Change**: Add a "Share" button alongside it:

```jsx
<div className="p-5 border-t border-white/10 bg-black/20 flex gap-3">
    <a href={reportUrl} target="_blank"
       className="flex-1 py-3 bg-[#151A22] hover:bg-white/10 font-mono text-[12px] uppercase tracking-widest border border-white/10 hover:border-white/20 rounded-lg text-center text-gray-400 hover:text-white transition-all flex items-center justify-center gap-2">
        <Download size={14} /> Report
    </a>
    <button onClick={() => {
        const text = `${regionName} — ${riskLevel} RISK\nFlood Area: ${floodArea} km2\nProbability: ${probability}%\nConfidence: ${confidence}% (GloFAS T${tier})\nTop Driver: ${topDriver}`;
        navigator.clipboard.writeText(text);
        // show brief toast or change button text to "Copied!" for 2s
    }}
    className="flex-1 py-3 bg-[#151A22] hover:bg-white/10 font-mono text-[12px] uppercase tracking-widest border border-white/10 hover:border-white/20 rounded-lg text-center text-gray-400 hover:text-white transition-all flex items-center justify-center gap-2">
        <Share2 size={14} /> Share
    </button>
</div>
```

Import `Share2` from lucide-react. The share button copies a plain-text summary to clipboard — professional users need to send findings to colleagues via Slack/email quickly.

---

### 12. Micro-Interactions (Throughout)

These are subtle polish items that apply across all sections:

1. **Progress bars**: Add `transition-all duration-700 ease-out` to all bar fills so they animate smoothly when data loads or changes.

2. **Hoverable data cells**: Assessment Detail grid cells and Contributing Factor rows should have `hover:bg-white/[0.03] transition-colors duration-200` — an almost imperceptible highlight on mouseover that confirms the UI is responsive.

3. **Risk badge glow**: Any risk-level colored element (header text, prediction hero, metric bars) gets a soft `textShadow` or `boxShadow` at 20-40% opacity of the risk color. Just enough to feel warm/alive without being flashy.

4. **Collapsible spring config**: Ensure all framer-motion `animate` transitions use the same spring: `transition={{ type: "spring", stiffness: 300, damping: 30 }}` for consistency. Currently the default might vary.

5. **Loading skeletons**: Where data is loading, use `bg-gray-800/50 animate-pulse rounded` rectangles that match the dimensions of the content they're replacing. Currently some loading states are spinners and some are pulse divs — standardize on pulse divs that match the layout.

---

## Semantic Color System (Reference)

Use these consistently across all changes:

| Semantic meaning | Color | Tailwind | Hex |
|-----------------|-------|----------|-----|
| Critical danger / active flood | red-400/500 | text-red-400 | #f87171 / #ef4444 |
| High risk / warning | orange-400/500 | text-orange-400 | #fb923c / #f97316 |
| Medium / watch | yellow-400/500 | text-yellow-400 | #facc15 / #eab308 |
| Low / safe / improving | emerald-400/500 | text-emerald-400 | #34d399 / #22c55e |
| Hydrological data (GloFAS, discharge, river) | cyan-400 | text-cyan-400 | #22d3ee |
| Forecasts / projections | violet-400 | text-violet-400 | #a78bfa |
| AI-generated content | amber-400 | text-amber-400 | #fbbf24 |
| Compound / cascading risk | rose-400 | text-rose-400 | #fb7185 |
| Financial / economic | emerald-400 | text-emerald-400 | #34d399 |
| Neutral data / labels | gray-400-600 | text-gray-500 | #6b7280 |
| Backgrounds | slate/navy | bg-[#151A22] | — |

---

## Files to Modify

Only one file for the UI changes:

**`frontend/src/app/engine/page.tsx`** — all visual changes are inline Tailwind class modifications, some new JSX structure, and icon imports.

New icon imports needed: `Droplets`, `CloudRain`, `Mountain`, `BarChart3`, `Share2` from lucide-react.

---

## Important Notes

1. **Single file architecture** — do NOT create separate component files. Everything stays in `page.tsx`.
2. **Both panels** — the registered-region view and the ad-hoc location view share the same visual structure. Apply all visual changes to BOTH. Search for the ad-hoc versions of each section (they're later in the file, typically after line ~1750).
3. **Preserve functionality** — these are purely visual changes. Do not modify API calls, data fetching, state management, or business logic. If a section currently shows data from `latestRisk.flood_percentage`, keep using that same data source.
4. **Preserve the orb system** — the panel changes content based on the active orb (flood/infra/veg). Keep all orb-conditional rendering intact. The visual changes should work with all three orbs.
5. **No new dependencies** — everything uses existing packages (Tailwind, framer-motion, recharts, lucide-react).
6. **Test with dark backgrounds** — the entire app is dark mode. Ensure all text and borders are visible against `bg-[#0B0E11]` and `bg-[#151A22]` backgrounds.
