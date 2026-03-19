// Use relative URL — Next.js rewrites proxy /api/* to the FastAPI backend
// This lets the site work via localhost, ngrok, or any deployment
const API = process.env.NEXT_PUBLIC_API_URL || "/api";

async function safeFetch(url: string) {
    try {
        const res = await fetch(url, { cache: 'no-store' });
        if (!res.ok) throw new Error(`${res.status}`);
        return res.json();
    } catch {
        return null;
    }
}

export async function fetchHealth() {
    return safeFetch(`${API}/health`);
}

export async function fetchRegions() {
    return safeFetch(`${API}/regions`);
}

export async function fetchChanges(limit = 50) {
    return safeFetch(`${API}/changes?limit=${limit}`);
}

export async function fetchRegionRisk(regionId: number) {
    return safeFetch(`${API}/regions/${regionId}/risk`);
}

export async function fetchRegionHistory(regionId: number, limit = 50) {
    return safeFetch(`${API}/regions/${regionId}/history?limit=${limit}`);
}

export async function fetchReport(regionId: number) {
    return safeFetch(`${API}/reports/${regionId}`);
}

export async function fetchLogs(limit = 50) {
    return safeFetch(`${API}/logs?limit=${limit}`);
}

export async function fetchPrediction(regionId: number) {
    return safeFetch(`${API}/predict/${regionId}`);
}

export async function fetchExternalFactors(regionId: number) {
    return safeFetch(`${API}/external/${regionId}`);
}

export async function fetchValidation(regionId: number) {
    return safeFetch(`${API}/validate/${regionId}`);
}

export async function fetchDischarge(regionId: number) {
    return safeFetch(`${API}/discharge/${regionId}`);
}

export async function fetchDetection(regionId: number) {
    return safeFetch(`${API}/detection/${regionId}`);
}

export async function triggerAnalysis(regionId: number) {
    try {
        const res = await fetch(`${API}/analyze/${regionId}`, { method: 'POST', cache: 'no-store' });
        if (!res.ok) throw new Error(`${res.status}`);
        return res.json();
    } catch {
        return null;
    }
}

export async function fetchAlerts(limit = 50) {
    return safeFetch(`${API}/alerts?limit=${limit}`);
}

export async function fetchExplanation(regionId: number) {
    return safeFetch(`${API}/explain/${regionId}`);
}

export function getReportDownloadUrl(regionId: number) {
    return `${API}/reports/${regionId}/download`;
}

export async function analyzeLocation(lat: number, lon: number, name?: string) {
    try {
        const res = await fetch(`${API}/analyze/location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, name }),
            cache: 'no-store',
        });
        if (!res.ok) throw new Error(`${res.status}`);
        return res.json();
    } catch {
        return null;
    }
}

export async function explainLocation(lat: number, lon: number) {
    return safeFetch(`${API}/explain/location?lat=${lat}&lon=${lon}`);
}

// --- Predictive Forecasting ---

export interface MonthlyForecast {
    month: string;
    month_name: string;
    risk_probability: number;
    risk_level: string;
    infra_exposure: number;
    vegetation_stress_index: number;
    confidence_lower: number;
    confidence_upper: number;
    seasonal_factor: number;
    precipitation_forecast_mm: number;
    assessment_details?: Record<string, any>;
    drivers: string[];
}

export interface ForecastData {
    region: string;
    generated_at: string;
    horizon_months: number;
    monthly_forecast: MonthlyForecast[];
    summary: {
        peak_risk_month: string;
        peak_probability: number;
        overall_trend: string;
        avg_risk_probability: number;
    };
}

export async function fetchForecast(regionId: number, horizon: number = 6): Promise<ForecastData | null> {
    return safeFetch(`${API}/forecast/${regionId}?horizon=${horizon}`);
}

export async function forecastLocation(lat: number, lon: number, name?: string): Promise<ForecastData | null> {
    try {
        const res = await fetch(`${API}/forecast/location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, name }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

// --- NLG Summaries ---

export interface NLGSummary {
    narrative: string;
    highlights: string[];
    risk_trend: string;
    generated_at: string;
    engine: string;
    trend_narrative?: string;
}

export async function fetchNLGSummary(regionId: number): Promise<NLGSummary | null> {
    return safeFetch(`${API}/nlg/summary/${regionId}`);
}

export async function nlgSummaryLocation(lat: number, lon: number, name?: string): Promise<NLGSummary | null> {
    try {
        const res = await fetch(`${API}/nlg/summary/location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, name }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// --- Multi-Sensor Fusion (Phase 2A) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchFusionAnalysis(regionId: number): Promise<any> {
    return safeFetch(`${API}/fusion/${regionId}`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fusionLocation(lat: number, lon: number, name?: string): Promise<any> {
    try {
        const res = await fetch(`${API}/fusion/location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, name }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// --- Compound Risk (Phase 2B) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchCompoundRisk(regionId: number): Promise<any> {
    return safeFetch(`${API}/compound-risk/${regionId}`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function compoundRiskLocation(lat: number, lon: number, name?: string): Promise<any> {
    try {
        const res = await fetch(`${API}/compound-risk/location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, name }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// --- Asset Scoring (Phase 3A) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchAssetScores(regionId: number): Promise<any> {
    return safeFetch(`${API}/assets/${regionId}`);
}

// --- Financial Impact (Phase 3B) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchFinancialImpact(regionId: number): Promise<any> {
    return safeFetch(`${API}/financial/${regionId}`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function financialImpactLocation(lat: number, lon: number, name?: string): Promise<any> {
    try {
        const res = await fetch(`${API}/financial/location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, name }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// --- ACD Monitoring (Phase 4) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchACDStatus(): Promise<any> {
    return safeFetch(`${API}/acd/status`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchACDAlerts(): Promise<any> {
    return safeFetch(`${API}/acd/alerts`);
}

// --- Reports (Phase 5A) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function generateReport(reportType: string, regionId: number): Promise<any> {
    return safeFetch(`${API}/reports/${reportType}/${regionId}`);
}

// --- Feedback (Phase 5B) ---
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function submitFeedback(data: any): Promise<any> {
    try {
        const res = await fetch(`${API}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchFeedbackStats(): Promise<any> {
    return safeFetch(`${API}/feedback/stats`);
}

export interface GeoResult {
    display_name: string;
    lat: number;
    lon: number;
    type: string;
    class: string;
    importance: number;
}

export async function geocodeSearch(query: string): Promise<GeoResult[]> {
    try {
        const res = await fetch(`${API}/geocode?q=${encodeURIComponent(query)}`, { cache: 'no-store' });
        if (!res.ok) return [];
        const data = await res.json();
        return data.results || [];
    } catch {
        return [];
    }
}

export async function reverseGeocode(lat: number, lon: number): Promise<{ display_name: string; short_name: string }> {
    try {
        const res = await fetch(`${API}/geocode/reverse?lat=${lat}&lon=${lon}`, { cache: 'no-store' });
        if (!res.ok) return { display_name: `${lat.toFixed(2)}, ${lon.toFixed(2)}`, short_name: `${lat.toFixed(2)}, ${lon.toFixed(2)}` };
        return res.json();
    } catch {
        return { display_name: `${lat.toFixed(2)}, ${lon.toFixed(2)}`, short_name: `${lat.toFixed(2)}, ${lon.toFixed(2)}` };
    }
}

// === Authentication ===

export interface AuthUser {
    id: number;
    username: string;
    role: 'admin' | 'analyst' | 'viewer';
    created_at: string;
    last_login: string | null;
}

export async function authLogin(username: string, password: string): Promise<{ access_token: string; user: AuthUser } | null> {
    try {
        const res = await fetch(`${API}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

export async function fetchMe(token: string): Promise<AuthUser | null> {
    try {
        const res = await fetch(`${API}/auth/me`, {
            headers: { Authorization: `Bearer ${token}` },
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// === Historical Trends ===

export interface MonthlyTrend {
    month: string;
    month_label: string;
    avg_flood_pct: number;
    max_flood_pct: number;
    total_precip_mm: number;
    max_precip_day_mm: number;
    avg_water_change_pct: number;
    max_water_change_pct: number;
    avg_vegetation_stress: number;
    max_vegetation_stress: number;
    heavy_rain_days: number;
    dominant_risk_level: string;
    risk_distribution: { LOW: number; MEDIUM: number; HIGH: number; CRITICAL: number };
    assessment_count: number;
}

export interface TrendData {
    region_id?: number;
    region_name?: string;
    lat?: number;
    lon?: number;
    months: number;
    data_points: number;
    trend: MonthlyTrend[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchTrends(regionId: number, months = 12): Promise<TrendData | null> {
    return safeFetch(`${API}/trends/${regionId}?months=${months}`);
}

/** Fetch real Open-Meteo ERA5 monthly trend data for any lat/lon (ad-hoc locations). */
export async function fetchTrendsLocation(lat: number, lon: number, months = 12): Promise<TrendData | null> {
    return safeFetch(`${API}/trends/location?lat=${lat}&lon=${lon}&months=${months}`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function fetchGlobalTrends(months = 6): Promise<any> {
    return safeFetch(`${API}/trends/global/summary?months=${months}`);
}

// === Periodic Scheduler ===

export interface SchedulerStatus {
    enabled: boolean;
    interval_hours: number;
    last_run: string | null;
    next_run: string | null;
    runs_completed: number;
    task_active: boolean;
}

export async function fetchSchedulerStatus(): Promise<SchedulerStatus | null> {
    return safeFetch(`${API}/scheduler/status`);
}

export async function triggerSchedulerNow(token: string): Promise<{ status: string } | null> {
    try {
        const res = await fetch(`${API}/scheduler/trigger`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

export async function configureScheduler(token: string, interval_hours?: number, enabled?: boolean): Promise<SchedulerStatus | null> {
    try {
        const res = await fetch(`${API}/scheduler/configure`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({ interval_hours, enabled }),
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

