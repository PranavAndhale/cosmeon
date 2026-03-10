// Use relative URL — Next.js rewrites proxy /api/* to the FastAPI backend
// This lets the site work via localhost, ngrok, or any deployment
const API = process.env.NEXT_PUBLIC_API_URL || "/api";

async function safeFetch(url: string) {
    try {
        const res = await fetch(url);
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
        const res = await fetch(`${API}/analyze/${regionId}`, { method: 'POST' });
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
