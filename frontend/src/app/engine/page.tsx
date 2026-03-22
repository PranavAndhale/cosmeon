"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Map, { Marker, Popup, MapRef } from "react-map-gl/maplibre";
import 'maplibre-gl/dist/maplibre-gl.css';
import { motion, AnimatePresence } from "framer-motion";
import { Search, Satellite, Database, Activity, Layers, Download, ChevronDown, Terminal, Play, Pause, MapPin, X, AlertTriangle, Leaf, Building2, Sparkles, TrendingUp, ChevronRight, Shield, DollarSign, Radio, ThumbsUp, ThumbsDown, Droplets, CloudRain, Mountain, BarChart3, Share2, Link } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip } from "recharts";

import { fetchRegions, fetchRegionRisk, fetchRegionHistory, fetchChanges, fetchLogs, getReportDownloadUrl, fetchPrediction, fetchValidation, fetchExplanation, analyzeLocation, explainLocation, geocodeSearch, reverseGeocode, GeoResult, fetchForecast, fetchNLGSummary, ForecastData, NLGSummary, fetchFusionAnalysis, fetchCompoundRisk, fetchFinancialImpact, submitFeedback, fusionLocation, compoundRiskLocation, financialImpactLocation, forecastLocation, nlgSummaryLocation, authLogin, AuthUser, fetchTrends, fetchTrendsLocation, TrendData, fetchSchedulerStatus, SchedulerStatus, fetchOrbAssessment, orbAssessmentLocation, fetchSituation, SituationData, SituationStatus } from "@/lib/api";
import { BarChart, Bar } from "recharts";

// ─── Types ───
interface Region {
  id: number;
  name: string;
  bbox: number[]; // [west, south, east, north]
}

interface RiskAssessment {
  id: number;
  region_id: number;
  timestamp: string;
  risk_level: string;
  flood_area_km2: number;
  total_area_km2: number;
  flood_percentage: number;
  confidence_score: number;
  change_type?: string;
  water_change_pct: number;
  assessment_details?: Record<string, unknown>;
}

interface ChangeEvent {
  id: number;
  region_id: number;
  current_date: string;
  water_change_pct: number;
  area_change_km2: number;
  change_type: string;
}

interface LogEntry {
  id: number;
  timestamp: string;
  step: string;
  status: string;
  duration_ms: number;
  details: { message?: string } | null;
}

interface Prediction {
  predicted_risk_level: string;
  flood_probability: number;
  confidence: number;
  contributing_factors: Record<string, number>;
}

interface ValidationData {
  our_prediction: { predicted_risk_level: string; flood_probability: number; confidence: number };
  validation: {
    our_prediction: string;
    glofas_risk_level: string;
    glofas_discharge_m3s: number;
    glofas_discharge_anomaly: number;
    agreement: boolean;
    agreement_score: number;
    data_source: string;
  };
  discharge_data: {
    current_discharge_m3s: number;
    mean_discharge_m3s: number;
    discharge_anomaly: number;
    flood_risk_level: string;
    forecast_dates: string[];
    forecast_discharge: number[];
  };
}

interface ExplainData {
  region: { id: number; name: string; bbox: number[] };
  ml_prediction: {
    risk_level: string;
    probability: number;
    confidence: number;
    class_probabilities: Record<string, number>;
    feature_values: Record<string, number>;
    top_drivers: { feature: string; value: number; importance: number; influence: string }[];
    explanation: string;
    model_inputs_source: string;
  };
  model_info: {
    architecture: string;
    training_required: boolean;
    description: string;
    data_sources: Record<string, string[]>;
  };
}

// ─── Orb Definitions ───
const ORB_DEFS = {
  flood: {
    id: "flood",
    label: "Flood Risk Detection",
    icon: Activity,
    color: "#00E5FF",
    panelTitle: "Flood Detection",
    metricLabel: "Flood Coverage",
    areaLabel: "Area Affected",
    chartLabel: "Flood % Over Time",
  },
  infra: {
    id: "infra",
    label: "Infrastructure Exposure",
    icon: Building2,
    color: "#f97316",
    panelTitle: "Infrastructure Exposure",
    metricLabel: "Exposed Infrastructure",
    areaLabel: "Exposure Zone",
    chartLabel: "Infrastructure Exposure Index",
  },
  veg: {
    id: "veg",
    label: "Vegetation Health",
    icon: Leaf,
    color: "#20E251",
    panelTitle: "Vegetation Health (NDVI)",
    metricLabel: "Vegetation Stress",
    areaLabel: "Monitored Coverage",
    chartLabel: "Vegetation Anomaly Index",
  },
} as const;

type OrbKey = keyof typeof ORB_DEFS;

// ─── Reusable Styles ───
const glassClass = "bg-[#0B0E11]/70 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl";
const textMono = "font-mono tracking-tight";

// ─── Helpers ───
function fmtUSD(v: number) {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
  return `$${Math.round(v).toLocaleString()}`;
}

function riskColor(level: string) {
  switch (level) {
    case "CRITICAL": return "#ef4444";
    case "HIGH": return "#f97316";
    case "MEDIUM": return "#eab308";
    default: return "#22c55e";
  }
}

function centerOf(bbox: number[]) {
  return { lon: (bbox[0] + bbox[2]) / 2, lat: (bbox[1] + bbox[3]) / 2 };
}

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

function situationMeta(status: SituationStatus) {
  switch (status) {
    case 'FLOODING_NOW': return { label: 'ACTIVE FLOODING', cls: 'bg-red-500/20 border-red-500/40 text-red-400', pulse: true };
    case 'IMMINENT':     return { label: 'IMMINENT RISK',   cls: 'bg-orange-500/20 border-orange-500/40 text-orange-400', pulse: false };
    case 'WATCH':        return { label: 'UNDER WATCH',     cls: 'bg-yellow-500/20 border-yellow-500/40 text-yellow-400', pulse: false };
    case 'RECEDING':     return { label: 'RECEDING',        cls: 'bg-cyan-500/20 border-cyan-500/40 text-cyan-400', pulse: false };
    default:             return { label: 'NORMAL',          cls: 'bg-gray-500/10 border-gray-600/30 text-gray-500', pulse: false };
  }
}

function markerSizeForStatus(status: SituationStatus): number {
  switch (status) {
    case 'FLOODING_NOW': return 14;
    case 'IMMINENT':     return 13;
    case 'WATCH':        return 11;
    case 'RECEDING':     return 10;
    default:             return 9;
  }
}

function markerColorForStatus(status: SituationStatus): string {
  switch (status) {
    case 'FLOODING_NOW': return '#ef4444';
    case 'IMMINENT':     return '#f97316';
    case 'WATCH':        return '#eab308';
    case 'RECEDING':     return '#22d3ee';
    default:             return '#4b5563';
  }
}

// ═══════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════
export default function GeospatialEngine() {
  // ── Data State (from API) ──
  const [regions, setRegions] = useState<Region[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [latestRisk, setLatestRisk] = useState<RiskAssessment | null>(null);
  const [riskHistory, setRiskHistory] = useState<RiskAssessment[]>([]);
  const [changes, setChanges] = useState<ChangeEvent[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [validation, setValidation] = useState<ValidationData | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [explanation, setExplanation] = useState<ExplainData | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  // showIndependence removed — section was removed from UI
  const [loading, setLoading] = useState(true);

  // ── Forecast & NLG State ──
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [nlgSummary, setNlgSummary] = useState<NLGSummary | null>(null);
  const [nlgLoading, setNlgLoading] = useState(false);
  const [showForecast, setShowForecast] = useState(false);
  const [showAiInsights, setShowAiInsights] = useState(false);

  // ── Advanced Features State ──
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [fusionData, setFusionData] = useState<any>(null);
  const [fusionLoading, setFusionLoading] = useState(false);
  const [showFusion, setShowFusion] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [compoundData, setCompoundData] = useState<any>(null);
  const [compoundLoading, setCompoundLoading] = useState(false);
  const [showCompound, setShowCompound] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [orbAssessment, setOrbAssessment] = useState<any>(null);
  const [orbAssessmentLoading, setOrbAssessmentLoading] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [financialData, setFinancialData] = useState<any>(null);
  const [financialLoading, setFinancialLoading] = useState(false);
  const [showFinancial, setShowFinancial] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  // ── Ad-hoc Location State ──
  const [adHocLocation, setAdHocLocation] = useState<{ lat: number; lon: number; name: string } | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [adHocData, setAdHocData] = useState<any>(null);
  const [adHocExplanation, setAdHocExplanation] = useState<ExplainData | null>(null);
  const [adHocLoading, setAdHocLoading] = useState(false);
  /** Real Open-Meteo ERA5 monthly trend data for the current ad-hoc location */
  const [adHocTrendData, setAdHocTrendData] = useState<TrendData | null>(null);
  /** Real Open-Meteo ERA5 monthly trend data for the current registered region (vegetation orb) */
  const [regionTrendData, setRegionTrendData] = useState<TrendData | null>(null);
  const [mapClickPopup, setMapClickPopup] = useState<{ lat: number; lon: number; name?: string } | null>(null);

  // ── UI State ──
  const [searchQuery, setSearchQuery] = useState("");
  const [searchFocused, setSearchFocused] = useState(false);
  const [geoResults, setGeoResults] = useState<GeoResult[]>([]);
  const [geoLoading, setGeoLoading] = useState(false);
  const geoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const geoRequestId = useRef(0); // increments on every search to discard stale responses
  const [activeSource, setActiveSource] = useState("MODIS");
  const [activeOrb, setActiveOrb] = useState<OrbKey>("flood");
  const [hoverInfo, setHoverInfo] = useState<Region | null>(null);
  const [playing, setPlaying] = useState(false);
  const [showLogs, setShowLogs] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapRef>(null);

  // ── Auth State ──
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginLoading, setLoginLoading] = useState(false);

  // ── Trend Dashboard State ──
  const [showTrends, setShowTrends] = useState(false);
  const [trendData, setTrendData] = useState<TrendData | null>(null);
  const [trendLoading, setTrendLoading] = useState(false);

  // ── Scheduler State ──
  const [schedulerStatus, setSchedulerStatus] = useState<SchedulerStatus | null>(null);

  // Load scheduler status on mount
  useEffect(() => {
    fetchSchedulerStatus().then(s => setSchedulerStatus(s));
    const interval = setInterval(() => fetchSchedulerStatus().then(s => setSchedulerStatus(s)), 60000);
    return () => clearInterval(interval);
  }, []);

  // ── Situation Board State ──
  const [situationData, setSituationData] = useState<SituationData | null>(null);
  const [situationLoading, setSituationLoading] = useState(false);

  // Load situation data on mount, refresh every 5 minutes
  useEffect(() => {
    const load = async () => {
      setSituationLoading(true);
      const data = await fetchSituation();
      if (data) setSituationData(data);
      setSituationLoading(false);
    };
    load();
    const interval = setInterval(load, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Restore auth from localStorage on mount
  useEffect(() => {
    const token = localStorage.getItem('cosmeon_token');
    const user = localStorage.getItem('cosmeon_user');
    if (token && user) {
      try {
        setAuthToken(token);
        setAuthUser(JSON.parse(user));
      } catch { localStorage.removeItem('cosmeon_token'); localStorage.removeItem('cosmeon_user'); }
    }
  }, []);

  const handleLogin = async () => {
    setLoginLoading(true); setLoginError(null);
    const result = await authLogin(loginUsername, loginPassword);
    setLoginLoading(false);
    if (!result) { setLoginError('Invalid username or password'); return; }
    setAuthToken(result.access_token);
    setAuthUser(result.user);
    localStorage.setItem('cosmeon_token', result.access_token);
    localStorage.setItem('cosmeon_user', JSON.stringify(result.user));
    setShowLogin(false); setLoginUsername(''); setLoginPassword('');
  };

  const handleLogout = () => {
    setAuthToken(null); setAuthUser(null);
    localStorage.removeItem('cosmeon_token'); localStorage.removeItem('cosmeon_user');
  };

  const openTrends = async () => {
    if (selectedRegion) {
      setShowTrends(true); setTrendLoading(true);
      const d = await fetchTrends(selectedRegion.id, 24);
      setTrendData(d); setTrendLoading(false);
    } else if (adHocLocation) {
      setShowTrends(true); setTrendLoading(true);
      const d = await fetchTrendsLocation(adHocLocation.lat, adHocLocation.lon, 24);
      setTrendData(d); setTrendLoading(false);
    }
  };



  const currentOrb = ORB_DEFS[activeOrb];

  // ── Detect coordinate input in search ──
  const parsedCoords = useMemo(() => {
    const q = searchQuery.trim();
    const m = q.match(/^(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)$/);
    if (m) {
      const lat = parseFloat(m[1]);
      const lon = parseFloat(m[2]);
      if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) return { lat, lon };
    }
    return null;
  }, [searchQuery]);

  // ── Filter regions by search ──
  const filteredRegions = useMemo(() => {
    if (!searchQuery.trim()) return regions;
    if (parsedCoords) return [];
    const q = searchQuery.toLowerCase();
    return regions.filter(r => r.name.toLowerCase().includes(q));
  }, [regions, searchQuery, parsedCoords]);

  // ── Debounced geocoding search for place names ──
  useEffect(() => {
    const q = searchQuery.trim();

    // Clear immediately when query is empty or is a coordinate
    if (!q || q.length < 2 || parsedCoords) {
      setGeoResults([]);
      setGeoLoading(false); // always reset spinner
      if (geoTimerRef.current) { clearTimeout(geoTimerRef.current); geoTimerRef.current = null; }
      return;
    }

    // If the query already exactly matches a monitored region, skip geocoding
    const exactRegionMatch = regions.some(r => r.name.toLowerCase() === q.toLowerCase());
    if (exactRegionMatch) {
      setGeoResults([]);
      setGeoLoading(false);
      return;
    }

    // Clear any pending timer
    if (geoTimerRef.current) clearTimeout(geoTimerRef.current);

    // 500ms debounce — avoids hammering Nominatim (rate limit: 1 req/s)
    geoTimerRef.current = setTimeout(async () => {
      const reqId = ++geoRequestId.current; // increment before async call
      setGeoLoading(true);
      setGeoResults([]); // clear stale results while fetching

      try {
        const results = await geocodeSearch(q);
        // Only apply if this is still the latest request
        if (reqId === geoRequestId.current) {
          setGeoResults(results ?? []);
        }
      } catch {
        if (reqId === geoRequestId.current) setGeoResults([]);
      } finally {
        if (reqId === geoRequestId.current) setGeoLoading(false);
      }
    }, 500);

    return () => {
      if (geoTimerRef.current) { clearTimeout(geoTimerRef.current); geoTimerRef.current = null; }
    };
  }, [searchQuery, parsedCoords, regions]);

  // ── Select a region & fly to it ──
  const selectRegion = useCallback((reg: Region | null) => {
    setSelectedRegion(reg);
    setRegionTrendData(null);
    setAdHocLocation(null); // clear ad-hoc when selecting a real region
    setAdHocData(null);
    setAdHocExplanation(null);
    setAdHocTrendData(null);
    setForecastData(null);
    setNlgSummary(null);
    setShowForecast(false);
    setShowAiInsights(false);
    setFusionData(null);
    setCompoundData(null);
    setFinancialData(null);
    setShowFusion(false);
    setShowCompound(false);
    setShowFinancial(false);
    setShowFeedback(false);
    if (reg && mapRef.current) {
      const { lon, lat } = centerOf(reg.bbox);
      mapRef.current.flyTo({ center: [lon, lat], zoom: 6, duration: 1500 });
    }
  }, []);

  // ── Analyze ad-hoc location ──
  const analyzeAdHocLocation = useCallback(async (lat: number, lon: number, name?: string) => {
    const locName = name || `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
    setAdHocLocation({ lat, lon, name: locName });
    setSelectedRegion(null); // deselect any region
    setAdHocLoading(true);
    setAdHocExplanation(null);
    setAdHocTrendData(null); // reset trend — useEffect will re-fetch for new location
    setMapClickPopup(null);
    setFusionData(null);
    setCompoundData(null);
    setFinancialData(null);
    setShowFusion(false);
    setShowCompound(false);
    setShowFinancial(false);
    setForecastData(null);
    setNlgSummary(null);
    setShowForecast(false);
    setShowAiInsights(false);
    setShowFeedback(false);

    if (mapRef.current) {
      mapRef.current.flyTo({ center: [lon, lat], zoom: 6, duration: 1500 });
    }

    const data = await analyzeLocation(lat, lon, locName);
    if (data) setAdHocData(data);
    setAdHocLoading(false);
  }, []);

  // ── Fetch real ERA5 monthly trends whenever an ad-hoc location is set ──
  useEffect(() => {
    if (!adHocLocation) {
      setAdHocTrendData(null);
      return;
    }
    // Fire-and-forget: populate chart while other analysis runs in parallel
    fetchTrendsLocation(adHocLocation.lat, adHocLocation.lon, 12).then(d => {
      if (d && d.trend && d.trend.length > 0) setAdHocTrendData(d);
    });
  }, [adHocLocation]);

  // ── Fetch ERA5 trends for registered region (used by vegetation orb chart) ──
  useEffect(() => {
    if (!selectedRegion) { setRegionTrendData(null); return; }
    const lat = (selectedRegion.bbox[1] + selectedRegion.bbox[3]) / 2;
    const lon = (selectedRegion.bbox[0] + selectedRegion.bbox[2]) / 2;
    fetchTrendsLocation(lat, lon, 12).then(d => {
      if (d?.trend?.length) setRegionTrendData(d);
    });
  }, [selectedRegion]);

  // ── Fetch orb-specific assessment when switching to infra / veg orbs ──
  useEffect(() => {
    setOrbAssessment(null);
    if (activeOrb === 'flood') return;
    const doFetch = async () => {
      setOrbAssessmentLoading(true);
      let data = null;
      if (selectedRegion) {
        data = await fetchOrbAssessment(selectedRegion.id);
      } else if (adHocLocation) {
        data = await orbAssessmentLocation(adHocLocation.lat, adHocLocation.lon, adHocLocation.name);
      }
      setOrbAssessment(data);
      setOrbAssessmentLoading(false);
    };
    doFetch();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeOrb, selectedRegion?.id, adHocLocation?.lat]);

  // ── Load Initial Data ──
  useEffect(() => {
    async function load() {
      setLoading(true);
      const regData = await fetchRegions();
      if (regData?.regions) {
        setRegions(regData.regions);
        if (regData.regions.length > 0) {
          setSelectedRegion(regData.regions[0]);
        }
      }
      const logData = await fetchLogs(60);
      if (logData?.logs) setLogs(logData.logs.reverse());
      setLoading(false);
    }
    load();
  }, []);

  // ── Reusable function to load all region data ──
  const loadRegionData = useCallback(async (region: Region, autoAnalyze = false) => {
    const [riskData, histData, chgData, predData] = await Promise.all([
      fetchRegionRisk(region.id),
      fetchRegionHistory(region.id, 30),
      fetchChanges(50),
      fetchPrediction(region.id),
    ]);
    if (riskData && !riskData.message) setLatestRisk(riskData);
    else setLatestRisk(null);
    if (histData?.assessments) setRiskHistory(histData.assessments);
    if (chgData?.events) setChanges(chgData.events.filter((e: ChangeEvent) => e.region_id === region.id));
    if (predData && predData.predicted_risk_level) setPrediction(predData);
    else setPrediction(null);

    // Fetch live validation (GloFAS cross-check) — non-blocking
    fetchValidation(region.id).then(valData => {
      if (valData && valData.validation) setValidation(valData);
      else setValidation(null);
    });

  }, []);

  // ── Load Region-Specific Data when selection changes ──
  useEffect(() => {
    if (!selectedRegion) return;
    setExplanation(null);
    loadRegionData(selectedRegion, true); // auto-analyze on every region select
  }, [selectedRegion, loadRegionData]);



  // ── Periodic live refresh every 5 minutes ──
  useEffect(() => {
    if (!selectedRegion) return;
    const interval = setInterval(() => {
      loadRegionData(selectedRegion, true);
    }, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [selectedRegion, loadRegionData]);

  // ── Auto-scroll logs ──
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  // ── Derived data for charts (registered regions from DB history) ──
  const chartData = useMemo(() =>
    riskHistory
      .slice()
      .reverse()
      .map(a => {
        const vegStress = ((a.assessment_details as Record<string, unknown>)?.vegetation_stress as number) || 0;
        return {
          date: a.timestamp ? new Date(a.timestamp).toLocaleDateString("en-US", { month: "short", year: "2-digit" }) : "",
          flood: Math.round(a.flood_percentage * 100 * 100) / 100,
          confidence: Math.round(a.confidence_score * 100),
          water_change: Math.round(a.water_change_pct * 100 * 100) / 100,
          vegetation_stress: Math.round(vegStress * 100 * 100) / 100,
        };
      })
    , [riskHistory]);

  /**
   * displayChartData — the actual data fed to the bottom time-series chart.
   *
   * • Ad-hoc location: use real Open-Meteo ERA5 monthly trend data fetched by
   *   fetchTrendsLocation().  This fills the chart for ANY searched location.
   * • Registered region: use DB risk-history as before.
   */
  const displayChartData = useMemo(() => {
    if (adHocLocation && adHocTrendData?.trend?.length) {
      return adHocTrendData.trend.map(t => ({
        date: t.month_label,
        flood: t.avg_flood_pct,
        confidence: t.heavy_rain_days,
        water_change: t.avg_water_change_pct,
        vegetation_stress: t.avg_vegetation_stress,
      }));
    }
    // For registered regions in vegetation orb: use ERA5 trend data so values
    // reflect actual climatological variation instead of flat DB records.
    if (selectedRegion && activeOrb === 'veg' && regionTrendData?.trend?.length) {
      return regionTrendData.trend.map(t => ({
        date: t.month_label,
        flood: t.avg_flood_pct,
        confidence: t.heavy_rain_days,
        water_change: t.avg_water_change_pct,
        vegetation_stress: t.avg_vegetation_stress,
      }));
    }
    return chartData;
  }, [adHocLocation, adHocTrendData, selectedRegion, activeOrb, regionTrendData, chartData]);

  // ── Choose chart data key based on active orb ──
  const chartKey = activeOrb === "flood" ? "flood" : activeOrb === "infra" ? "water_change" : "vegetation_stress";
  const primaryColor = currentOrb.color;

  return (
    <main className="w-screen h-screen relative bg-[#0B0E11] overflow-hidden text-white font-sans">

      {/* ═══ BASE MAP LAYER ═══ */}
      <div className="absolute inset-0 z-0">
        <Map
          ref={mapRef}
          initialViewState={{ longitude: 85.3, latitude: 25.0, zoom: 3, pitch: 30 }}
          onClick={async (e) => {
            const { lng, lat } = e.lngLat;
            // Show popup with coordinates immediately, then resolve name
            setMapClickPopup({ lat, lon: lng });
            // Reverse geocode to get place name
            reverseGeocode(lat, lng).then(geo => {
              setMapClickPopup(prev => prev ? { ...prev, name: geo.short_name } : null);
            });
          }}
          mapStyle={{
            version: 8,
            sources: {
              'satellite': { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize: 256 },
              'dark-labels': { type: 'raster', tiles: ['https://a.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png'], tileSize: 256 }
            },
            layers: [
              { id: 'satellite-layer', type: 'raster', source: 'satellite', paint: { 'raster-opacity': 0.6 } },
              { id: 'labels-layer', type: 'raster', source: 'dark-labels', paint: { 'raster-opacity': 1 } }
            ]
          }}
        >
          {regions.map(reg => {
            const { lon, lat } = centerOf(reg.bbox);
            const isSelected = selectedRegion?.id === reg.id;
            const sitReg = situationData?.regions.find(r => r.id === reg.id);
            const status: SituationStatus = sitReg?.situation_status ?? 'NORMAL';
            const mColor = isSelected ? primaryColor : markerColorForStatus(status);
            const mSize = markerSizeForStatus(status);
            return (
              <Marker key={reg.id} longitude={lon} latitude={lat} anchor="center">
                <div className="flex flex-col items-center gap-0.5">
                  <div
                    className="rounded-full border-2 cursor-pointer transition-all duration-300 relative flex items-center justify-center"
                    style={{
                      width: mSize + 6,
                      height: mSize + 6,
                      backgroundColor: isSelected ? mColor : mColor + '30',
                      borderColor: mColor,
                      boxShadow: isSelected ? `0 0 15px ${mColor}` : status === 'FLOODING_NOW' ? `0 0 8px ${mColor}60` : 'none',
                    }}
                    onClick={() => selectRegion(reg)}
                    onMouseEnter={() => setHoverInfo(reg)}
                    onMouseLeave={() => setHoverInfo(null)}
                  >
                    {(isSelected || status === 'FLOODING_NOW') && (
                      <div className="absolute -inset-2 rounded-full animate-ping opacity-25"
                        style={{ border: `1px solid ${mColor}` }} />
                    )}
                  </div>
                  <span className="text-[9px] font-mono text-gray-400 text-center leading-none max-w-[60px] truncate"
                    style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}>
                    {reg.name.split(',')[0]}
                  </span>
                </div>
              </Marker>
            );
          })}

          {hoverInfo && (
            <Popup longitude={centerOf(hoverInfo.bbox).lon} latitude={centerOf(hoverInfo.bbox).lat} closeButton={false} closeOnClick={false} anchor="bottom" offset={20}>
              <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-1 min-w-[160px]">
                <span className={`${textMono} text-[12px] uppercase`} style={{ color: primaryColor }}>{hoverInfo.name}</span>
                <span className={`${textMono} text-[13px] text-gray-400`}>Click to analyze</span>
              </motion.div>
            </Popup>
          )}

          {/* Ad-hoc location marker (cyan) */}
          {adHocLocation && (
            <Marker longitude={adHocLocation.lon} latitude={adHocLocation.lat} anchor="center">
              <div className="relative flex items-center justify-center">
                <div className="w-7 h-7 rounded-full border-2 border-cyan-400 bg-cyan-400/20 flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.5)]">
                  <MapPin size={14} className="text-cyan-400" />
                </div>
                <div className="absolute -inset-2 rounded-full animate-ping opacity-30 border border-cyan-400" />
              </div>
            </Marker>
          )}

          {/* Map click popup — "Analyze this location?" */}
          {mapClickPopup && (
            <Popup longitude={mapClickPopup.lon} latitude={mapClickPopup.lat} closeButton={false} closeOnClick={true} anchor="bottom" offset={20}
              onClose={() => setMapClickPopup(null)}>
              <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-2 min-w-[220px] p-1.5">
                {mapClickPopup.name && (
                  <span className={`${textMono} text-[13px] text-white font-semibold`}>
                    {mapClickPopup.name}
                  </span>
                )}
                <span className={`${textMono} text-[11px] text-gray-400`}>
                  {mapClickPopup.lat.toFixed(4)}, {mapClickPopup.lon.toFixed(4)}
                </span>
                <button
                  onClick={() => analyzeAdHocLocation(mapClickPopup.lat, mapClickPopup.lon, mapClickPopup.name)}
                  className="w-full py-2.5 px-3 rounded-lg text-[13px] font-mono font-bold uppercase tracking-wider transition-all bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30"
                >
                  Analyze This Location
                </button>
              </motion.div>
            </Popup>
          )}
        </Map>
      </div>

      {/* ═══ A. TOP NAVIGATION BAR ═══ */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="absolute top-6 left-1/2 -translate-x-1/2 h-14 z-20 flex items-center justify-between w-[95%] pointer-events-none">
        <h1 className={`${textMono} text-lg font-bold flex items-center gap-2 pointer-events-auto select-none backdrop-blur-md bg-[#0B0E11]/50 px-4 py-2 rounded-xl border border-white/10 shadow-lg`}>
          <div className="w-8 h-8 rounded-lg bg-[#00E5FF]/10 flex items-center justify-center border border-[#00E5FF]/30"><MapPin size={16} className="text-[#00E5FF]" /></div>
          COSMEON
        </h1>

        <div className="pointer-events-auto absolute left-1/2 -translate-x-1/2" style={{ position: 'absolute' }}>
          <div className="relative">
            <div className={`h-12 flex items-center gap-3 px-2 py-1 ${glassClass} rounded-full`}
              style={{ borderColor: searchFocused ? 'rgba(0,229,255,0.3)' : undefined }}>
              <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center ml-1">
                {geoLoading
                  ? <div className="w-3 h-3 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                  : <Search size={14} className="text-gray-400" />
                }
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setSearchFocused(true)}
                // Use onMouseDown on dropdown items so blur fires AFTER selection
                onBlur={() => setTimeout(() => setSearchFocused(false), 250)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    setSearchQuery(""); setGeoResults([]); setSearchFocused(false);
                    (e.target as HTMLInputElement).blur();
                  } else if (e.key === 'Enter') {
                    // Enter with coords — analyze immediately
                    if (parsedCoords) {
                      analyzeAdHocLocation(parsedCoords.lat, parsedCoords.lon);
                      setSearchQuery(""); setGeoResults([]);
                    // Enter with a single geo result — select it
                    } else if (geoResults.length === 1) {
                      const geo = geoResults[0];
                      const parts = geo.display_name.split(', ');
                      analyzeAdHocLocation(geo.lat, geo.lon, parts.slice(0, 3).join(', '));
                      setSearchQuery(""); setGeoResults([]);
                    // Enter with a single region match — select it
                    } else if (filteredRegions.length === 1) {
                      selectRegion(filteredRegions[0]); setSearchQuery("");
                    }
                  }
                }}
                placeholder="Search any place on Earth... e.g., Tokyo, Mumbai, 26.0 85.5"
                className={`bg-transparent outline-none border-none text-sm w-96 text-white placeholder:text-gray-500 ${textMono}`}
              />
              {searchQuery && (
                <button
                  onMouseDown={(e) => { e.preventDefault(); setSearchQuery(""); setGeoResults([]); setGeoLoading(false); }}
                  className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center mr-1 hover:bg-white/20 transition-colors"
                >
                  <X size={10} className="text-gray-400" />
                </button>
              )}
            </div>
            {/* Search Dropdown */}
            <AnimatePresence>
              {searchQuery && searchFocused && (filteredRegions.length > 0 || parsedCoords || geoResults.length > 0 || geoLoading || searchQuery.trim().length >= 2) && (
                <motion.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.12 }}
                  className={`absolute top-14 left-0 right-0 ${glassClass} !rounded-xl py-2 max-h-[360px] overflow-y-auto z-50`}
                  style={{ minWidth: '420px' }}
                >
                  {/* Coordinate analysis option */}
                  {parsedCoords && (
                    <div
                      onMouseDown={(e) => { e.preventDefault(); analyzeAdHocLocation(parsedCoords.lat, parsedCoords.lon); setSearchQuery(""); setGeoResults([]); }}
                      className="px-4 py-3 flex items-center gap-3 cursor-pointer hover:bg-cyan-500/10 transition-colors border-b border-white/5"
                    >
                      <div className="w-7 h-7 rounded-full bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center shrink-0">
                        <MapPin size={14} className="text-cyan-400" />
                      </div>
                      <div className="flex flex-col">
                        <span className={`text-sm ${textMono} text-cyan-400`}>Analyze {parsedCoords.lat.toFixed(4)}, {parsedCoords.lon.toFixed(4)}</span>
                        <span className={`text-[11px] ${textMono} text-gray-500`}>Direct coordinate entry — press Enter or click</span>
                      </div>
                    </div>
                  )}

                  {/* Monitored regions */}
                  {filteredRegions.length > 0 && (
                    <>
                      <div className={`px-4 py-1.5 text-[11px] text-gray-600 uppercase ${textMono} tracking-widest`}>Monitored Regions</div>
                      {filteredRegions.map(reg => (
                        <div
                          key={reg.id}
                          onMouseDown={(e) => { e.preventDefault(); selectRegion(reg); setSearchQuery(""); setGeoResults([]); }}
                          className="px-4 py-2.5 flex items-center gap-3 cursor-pointer hover:bg-white/10 transition-colors"
                        >
                          <MapPin size={14} style={{ color: primaryColor }} />
                          <span className={`text-sm ${textMono} text-gray-300`}>{reg.name}</span>
                        </div>
                      ))}
                    </>
                  )}

                  {/* Geocoded place results */}
                  {geoLoading && (
                    <div className="px-4 py-3 flex items-center gap-3">
                      <div className="w-4 h-4 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin shrink-0" />
                      <span className={`text-[13px] ${textMono} text-gray-400`}>Searching worldwide places…</span>
                    </div>
                  )}
                  {!geoLoading && geoResults.length > 0 && (
                    <>
                      <div className={`px-4 py-1.5 text-[11px] text-gray-600 uppercase ${textMono} tracking-widest ${filteredRegions.length > 0 ? 'border-t border-white/5 mt-1' : ''}`}>
                        Places Worldwide
                      </div>
                      {geoResults.map((geo, i) => {
                        const parts = geo.display_name.split(', ');
                        const shortName = parts.slice(0, 3).join(', ');
                        return (
                          <div
                            key={`geo-${i}`}
                            onMouseDown={(e) => { e.preventDefault(); analyzeAdHocLocation(geo.lat, geo.lon, shortName); setSearchQuery(""); setGeoResults([]); }}
                            className="px-4 py-2.5 flex items-center gap-3 cursor-pointer hover:bg-cyan-500/10 transition-colors"
                          >
                            <div className="w-6 h-6 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shrink-0">
                              <MapPin size={12} className="text-cyan-400" />
                            </div>
                            <div className="flex flex-col min-w-0">
                              <span className={`text-sm ${textMono} text-gray-200 truncate`}>{shortName}</span>
                              <span className={`text-[11px] ${textMono} text-gray-500 truncate`}>{geo.lat.toFixed(4)}, {geo.lon.toFixed(4)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </>
                  )}
                  {/* No results state — only shown once loading is complete */}
                  {!geoLoading && geoResults.length === 0 && filteredRegions.length === 0 && !parsedCoords && searchQuery.trim().length >= 2 && (
                    <div className={`px-4 py-3 text-[13px] ${textMono} text-gray-500`}>
                      No results for &quot;{searchQuery}&quot; — try a different spelling or use lat, lon
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="flex items-center gap-4 pointer-events-auto">
          <button className={`h-10 px-4 flex items-center gap-2 text-xs uppercase ${textMono} text-gray-300 hover:text-white transition-colors hover:bg-white/5 rounded-lg`}>
            <Database size={14} className="text-[#00E5FF]" /> API Access
          </button>
          <div className={`h-10 px-4 flex items-center gap-3 ${glassClass} !rounded-full text-xs uppercase ${textMono} shadow-none`}>
            <div className="w-2 h-2 rounded-full bg-[#20E251] shadow-[0_0_8px_#20E251]" />
            System Live
          </div>
          {/* Scheduler status pill */}
          {schedulerStatus && (
            <div
              title={`Auto-monitoring: every ${schedulerStatus.interval_hours}h\nNext: ${schedulerStatus.next_run ? new Date(schedulerStatus.next_run + 'Z').toLocaleTimeString() : 'soon'}\nRuns: ${schedulerStatus.runs_completed}`}
              className={`h-10 px-3 flex items-center gap-2 rounded-lg text-[11px] font-mono border cursor-default ${schedulerStatus.enabled ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-gray-800/50 border-white/10 text-gray-500'
                }`}
            >
              <div className={`w-1.5 h-1.5 rounded-full ${schedulerStatus.enabled ? 'bg-emerald-400 animate-pulse' : 'bg-gray-600'}`} />
              AUTO {schedulerStatus.interval_hours}H
            </div>
          )}
          {/* Auth: user button or login */}
          {authUser ? (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 h-10 px-3 rounded-lg bg-white/5 border border-white/10">
                <span className={`text-[10px] uppercase font-mono px-1.5 py-0.5 rounded ${authUser.role === 'admin' ? 'bg-red-500/20 text-red-400' :
                  authUser.role === 'analyst' ? 'bg-amber-500/20 text-amber-400' :
                    'bg-blue-500/20 text-blue-400'
                  }`}>{authUser.role}</span>
                <span className="text-[13px] font-mono text-gray-200">{authUser.username}</span>
              </div>
              <button onClick={handleLogout} className="h-10 px-3 text-[11px] font-mono text-gray-500 hover:text-red-400 transition-colors rounded-lg hover:bg-red-500/10 border border-transparent hover:border-red-500/20">Logout</button>
            </div>
          ) : (
            <button onClick={() => setShowLogin(true)} className="w-10 h-10 rounded-full bg-gradient-to-br from-[#0B0E11] to-[#151A22] border border-white/20 flex items-center justify-center text-xs font-bold font-mono hover:border-cyan-500/50 transition-colors">PA</button>
          )}
        </div>
      </motion.div>

      {/* ═══ B. INTELLIGENCE HUB (LEFT) ═══ */}
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="absolute left-6 top-28 w-[340px] z-20 flex flex-col gap-4" style={{ bottom: showLogs ? '200px' : '70px' }}>

        <div className={`${glassClass} p-5 flex flex-col gap-4`}>
          <h2 className={`text-[13px] uppercase text-gray-400 ${textMono} tracking-widest`}>Data Source</h2>
          <div className="flex bg-[#151A22] rounded-lg p-1 border border-white/10 text-[13px] font-mono">
            {['MODIS', 'ERA5', 'GloFAS'].map(s => (
              <button key={s} onClick={() => setActiveSource(s)} className={`flex-1 py-1.5 rounded-md transition-all ${activeSource === s ? 'bg-white/10 text-white shadow-md' : 'text-gray-500 hover:text-white/80'}`}>{s}</button>
            ))}
          </div>
        </div>

        <div className={`${glassClass} p-5 flex-grow flex flex-col overflow-y-auto`}>
          <h2 className={`text-[13px] uppercase text-gray-400 ${textMono} tracking-widest mb-4 flex items-center justify-between`}>
            Active Region (AOI)
            <ChevronDown size={12} className="text-gray-500" />
          </h2>

          {/* ── Situation Board ── */}
          <div className="flex flex-col gap-2 mb-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-1">
              <span className={`text-[11px] uppercase ${textMono} tracking-widest text-gray-500`}>Situation Board</span>
              {situationData && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-cyan-500/30 bg-cyan-500/10 text-cyan-400">
                  {(situationData.summary.flooding_now + situationData.summary.imminent)} active / {situationData.summary.total}
                </span>
              )}
            </div>

            {/* Loading skeletons */}
            {situationLoading && !situationData && (
              <div className="flex flex-col gap-1.5 animate-pulse">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="bg-[#151A22]/80 rounded-lg p-3 h-[68px]" />
                ))}
              </div>
            )}

            {/* Empty state */}
            {situationData && situationData.regions.length === 0 && (
              <div className="text-[12px] text-gray-600 font-mono p-3">All regions nominal — no elevated risk detected</div>
            )}

            {/* Situation rows */}
            {(situationData?.regions ?? [])
              .filter(r => !searchQuery.trim() || r.name.toLowerCase().includes(searchQuery.toLowerCase()))
              .map(sit => {
                const reg = regions.find(r => r.id === sit.id);
                if (!reg) return null;
                const isSelected = selectedRegion?.id === sit.id;
                const sm = situationMeta(sit.situation_status);
                const riskCol = riskColor(sit.risk_level);
                const barWidth = Math.min(sit.flood_percentage * 500, 100); // scale: 20% flood = full bar
                const trendArrow = sit.trend === 'escalating' ? '▲' : sit.trend === 'improving' ? '▼' : '→';
                const trendCls = sit.trend === 'escalating' ? 'text-red-400' : sit.trend === 'improving' ? 'text-emerald-400' : 'text-gray-500';
                return (
                  <div
                    key={sit.id}
                    onClick={() => selectRegion(reg)}
                    className={`rounded-lg border cursor-pointer transition-all overflow-hidden ${isSelected ? 'bg-white/5' : 'border-white/5 hover:bg-white/5'}`}
                    style={{ borderColor: isSelected ? riskCol + '33' : undefined }}
                  >
                    {/* Severity bar */}
                    <div className="w-full h-[3px] bg-[#0B0E11]">
                      <div className="h-full transition-all duration-700" style={{ width: `${barWidth}%`, backgroundColor: riskCol }} />
                    </div>
                    <div className="p-2.5 flex flex-col gap-1">
                      {/* Row 1: name + badge + trend */}
                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-[13px] font-mono truncate ${isSelected ? 'text-white' : 'text-gray-200'}`}>{sit.name}</span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <span className="text-[10px] font-mono font-bold" style={{ color: riskCol }}>{sit.risk_level}</span>
                          <span className={`text-[11px] font-mono font-bold ${trendCls}`}>{trendArrow}</span>
                        </div>
                      </div>
                      {/* Row 2: metrics + situation badge */}
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-mono text-gray-500 truncate">
                          {sit.flood_area_km2.toFixed(0)} km² · {sit.discharge_anomaly_sigma >= 0 ? '+' : ''}{sit.discharge_anomaly_sigma.toFixed(1)}σ
                        </span>
                        {sit.situation_status !== 'NORMAL' && (
                          <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${sm.cls} ${sm.pulse ? 'animate-pulse' : ''}`}>
                            {sm.label}
                          </span>
                        )}
                      </div>
                      {/* Row 3: timestamp */}
                      <span className="text-[10px] font-mono text-gray-600">Updated {timeAgo(sit.last_assessed)}</span>
                    </div>
                  </div>
                );
              })}

            {/* Fallback: show plain list if situation data not loaded yet */}
            {!situationData && !situationLoading && filteredRegions.map(reg => (
              <div key={reg.id} onClick={() => selectRegion(reg)} className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center gap-3
                ${selectedRegion?.id === reg.id ? 'bg-white/5' : 'border-white/5 hover:bg-white/5'}`}
                style={{ borderColor: selectedRegion?.id === reg.id ? primaryColor + '33' : undefined }}
              >
                <MapPin size={14} style={{ color: selectedRegion?.id === reg.id ? primaryColor : '#6b7280' }} />
                <span className={`text-[14px] ${textMono} ${selectedRegion?.id === reg.id ? 'text-white' : 'text-gray-400'}`}>{reg.name}</span>
              </div>
            ))}
          </div>

          <h2 className={`text-[13px] uppercase text-gray-400 ${textMono} tracking-widest mb-4`}>Analysis Orbs</h2>
          <div className="flex flex-col gap-3">
            {(Object.values(ORB_DEFS) as typeof ORB_DEFS[OrbKey][]).map(orb => {
              const isActive = activeOrb === orb.id;
              return (
                <button key={orb.id} onClick={() => setActiveOrb(orb.id as OrbKey)} className="flex items-center gap-3 p-3 rounded-xl border transition-all"
                  style={{
                    borderColor: isActive ? orb.color + '80' : 'rgba(255,255,255,0.1)',
                    backgroundColor: isActive ? orb.color + '15' : 'rgba(21,26,34,0.5)',
                  }}
                >
                  <div className="w-8 h-8 rounded-full flex items-center justify-center border"
                    style={{
                      backgroundColor: isActive ? 'rgba(255,255,255,0.1)' : '#0B0E11',
                      borderColor: isActive ? orb.color + '40' : 'rgba(255,255,255,0.05)',
                      color: isActive ? orb.color : '#6b7280',
                    }}
                  >
                    <orb.icon size={14} />
                  </div>
                  <div className="flex flex-col items-start gap-1">
                    <span className={`text-[13px] uppercase ${textMono} ${isActive ? 'text-white' : 'text-gray-400'}`}>{orb.label}</span>
                    {isActive && <div className="text-[12px] font-mono animate-pulse" style={{ color: orb.color }}>Active</div>}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </motion.div>

      {/* ═══ C. STRUCTURED INSIGHTS OUTPUT (RIGHT) ═══ */}
      <AnimatePresence>
        {selectedRegion && (
          <motion.div key={`insights-${selectedRegion.id}-${activeOrb}`} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }} className="absolute right-6 top-28 bottom-32 w-[400px] z-20 flex flex-col gap-4 overflow-hidden">
            <div className={`${glassClass} flex flex-col h-full overflow-hidden`}>

              {/* Header */}
              <div className="p-6 border-b border-white/10 bg-black/40 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col gap-1">
                    {latestRisk ? (
                      <>
                        {(() => {
                          const liveLevel = prediction?.predicted_risk_level ?? latestRisk.risk_level;
                          const displayLevel = activeOrb === 'infra' ? (orbAssessment?.infra?.risk_level ?? liveLevel) : activeOrb === 'veg' ? (orbAssessment?.veg?.risk_level ?? liveLevel) : liveLevel;
                          return (
                            <span className={`text-[16px] uppercase tracking-widest ${textMono} font-bold flex items-center gap-2`} style={{ color: riskColor(displayLevel), textShadow: `0 0 12px ${riskColor(displayLevel)}66` }}>
                              <AlertTriangle size={16} className={(displayLevel === 'HIGH' || displayLevel === 'CRITICAL') ? 'animate-pulse' : ''} />
                              {displayLevel} RISK
                              <span className="px-2 py-0.5 rounded text-[10px] uppercase font-mono border ml-1 tracking-wider" style={{ borderColor: riskColor(displayLevel) + '40', backgroundColor: riskColor(displayLevel) + '20' }}>
                                {activeOrb === 'infra' ? 'INFRA EXPOSURE' : activeOrb === 'veg' ? 'VEG STRESS' : (prediction ? 'ML PREDICTION' : latestRisk.change_type)}
                              </span>
                            </span>
                          );
                        })()}
                        {(() => {
                          const sit = situationData?.regions.find(r => r.id === selectedRegion.id);
                          if (!sit || sit.situation_status === 'NORMAL') return null;
                          const sm = situationMeta(sit.situation_status);
                          return (
                            <span className={`text-[10px] font-mono px-2 py-0.5 rounded border w-fit ${sm.cls} ${sm.pulse ? 'animate-pulse' : ''}`}>
                              {sm.label}
                            </span>
                          );
                        })()}
                      </>
                    ) : (
                      <div className="animate-pulse flex items-center gap-2">
                        <div className="h-4 w-4 bg-gray-600 rounded" />
                        <div className="h-4 w-32 bg-gray-700 rounded" />
                        <div className="h-4 w-16 bg-gray-700 rounded" />
                      </div>
                    )}
                    <span className="text-white text-[15px] font-semibold mt-1 flex items-center justify-between">
                      <span>{selectedRegion.name} — {currentOrb.panelTitle}</span>
                      {latestRisk?.timestamp && <span className="text-[10px] text-gray-500 font-mono tracking-widest uppercase truncate ml-2">Last assessed: {timeAgo(latestRisk.timestamp)}</span>}
                    </span>
                  </div>
                  <button onClick={() => setSelectedRegion(null)} className="text-gray-500 hover:text-white"><X size={18} /></button>
                </div>

              </div>

              {/* Data Grid */}
              <div className="flex-grow p-6 flex flex-col gap-6 overflow-y-auto">
                {!latestRisk ? (
                  <div className="animate-pulse flex flex-col gap-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[#151A22]/80 rounded-xl p-4 h-20" />
                      <div className="bg-[#151A22]/80 rounded-xl p-4 h-20" />
                    </div>
                    <div className="bg-[#151A22]/80 rounded-xl p-4 h-40">
                      <div className="h-3 w-32 bg-gray-700 rounded mb-4" />
                      {[...Array(4)].map((_, i) => (
                        <div key={i} className="flex justify-between mb-3">
                          <div className="h-3 w-28 bg-gray-700 rounded" />
                          <div className="h-3 w-16 bg-gray-600 rounded" />
                        </div>
                      ))}
                    </div>
                    <p className={`text-[13px] text-gray-500 text-center font-mono`}>Loading risk data…</p>
                  </div>
                ) : (
                <>
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-[#151A22]/80 rounded-xl p-3 flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-gray-500 font-mono">FLOOD AREA</span>
                    <span className="text-[18px] font-bold font-mono text-white">{(latestRisk.flood_area_km2).toFixed(0)} km²</span>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="h-[3px] flex-1 bg-[#1a1f2e] rounded-full overflow-hidden">
                        <div className="h-full transition-all duration-700 ease-out" style={{ width: `${Math.min(latestRisk.flood_percentage * 500, 100)}%`, backgroundColor: riskColor(latestRisk.risk_level) }} />
                      </div>
                      <span className="text-[10px] font-mono text-gray-400">{(latestRisk.flood_percentage * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="bg-[#151A22]/80 rounded-xl p-3 flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-gray-500 font-mono">RISK PROB.</span>
                    {prediction ? (
                      <>
                        <span className="text-[18px] font-bold font-mono text-white">{(prediction.flood_probability * 100).toFixed(0)}%</span>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="h-[3px] flex-1 bg-[#1a1f2e] rounded-full overflow-hidden">
                            <div className="h-full transition-all duration-700 ease-out" style={{ width: `${prediction.flood_probability * 100}%`, backgroundColor: riskColor(prediction.predicted_risk_level) }} />
                          </div>
                          <span className="text-[10px] font-mono" style={{ color: riskColor(prediction.predicted_risk_level) }}>{prediction.predicted_risk_level}</span>
                        </div>
                      </>
                    ) : (
                      <span className="text-[18px] font-bold font-mono text-gray-600 opacity-50">--</span>
                    )}
                  </div>
                  <div className="bg-[#151A22]/80 rounded-xl p-3 flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-gray-500 font-mono">CONFIDENCE</span>
                    {prediction ? (
                      <>
                        <span className="text-[18px] font-bold font-mono text-white">{(prediction.confidence * 100).toFixed(0)}%</span>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="h-[3px] flex-1 bg-[#1a1f2e] rounded-full overflow-hidden">
                            <div className="h-full transition-all duration-700 ease-out bg-cyan-400" style={{ width: `${prediction.confidence * 100}%` }} />
                          </div>
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          <span className="text-[10px] font-mono text-cyan-400">T{((prediction as any).feature_values?.glofas_flood_risk) ? Math.min(Math.round((prediction as any).feature_values.glofas_flood_risk), 3) : 2}</span>
                        </div>
                      </>
                    ) : (
                      <span className="text-[18px] font-bold font-mono text-gray-600 opacity-50">--</span>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[12px] font-medium font-mono uppercase tracking-widest text-gray-500">Assessment Details</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-white/5 text-gray-500 bg-white/5">[ERA5 + GloFAS]</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">{currentOrb.metricLabel}</span>
                      {orbAssessmentLoading && activeOrb !== 'flood' ? (
                        <span className="text-gray-500 text-[16px] font-mono">loading...</span>
                      ) : (() => {
                        const valPct = activeOrb === 'infra' && orbAssessment?.infra ? orbAssessment.infra.exposure_score * 100 : activeOrb === 'veg' && orbAssessment?.veg ? orbAssessment.veg.stress_index * 100 : latestRisk.flood_percentage * 100;
                        const col = valPct > 10 ? 'text-red-400' : valPct > 5 ? 'text-orange-400' : valPct > 2 ? 'text-yellow-400' : 'text-emerald-400';
                        return (
                          <>
                            <span className={`text-[16px] font-bold font-mono ${col}`}>{valPct.toFixed(1)}%</span>
                            <div className={`h-[2px] rounded-full mt-2 w-full bg-[#1a1f2e]`}><div className={`h-full transition-all duration-700 ease-out bg-current ${col}`} style={{ width: `${Math.min(valPct, 100)}%` }} /></div>
                          </>
                        );
                      })()}
                    </div>

                    <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Water Change</span>
                      <div className="flex items-center gap-2">
                        <span className={`text-[16px] font-bold font-mono ${latestRisk.water_change_pct > 0 ? 'text-red-400' : 'text-emerald-400'}`}>{latestRisk.water_change_pct > 0 ? '+' : ''}{(latestRisk.water_change_pct * 100).toFixed(1)}%</span>
                        <span className={`text-[10px] font-mono ${latestRisk.water_change_pct > 0 ? 'text-red-400' : latestRisk.water_change_pct < 0 ? 'text-emerald-400' : 'text-gray-500'}`}>{latestRisk.water_change_pct > 0 ? '▲ expanding' : latestRisk.water_change_pct < 0 ? '▼ receding' : '→ stable'}</span>
                      </div>
                    </div>

                    {activeOrb === 'infra' && orbAssessment?.infra && (
                      <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Soil Saturation</span>
                        <span className={`text-[16px] font-bold font-mono ${orbAssessment.infra.soil_saturation > 0.8 ? 'text-red-400' : orbAssessment.infra.soil_saturation > 0.6 ? 'text-orange-400' : orbAssessment.infra.soil_saturation > 0.4 ? 'text-yellow-400' : 'text-emerald-400'}`}>{(orbAssessment.infra.soil_saturation * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {activeOrb === 'veg' && orbAssessment?.veg && (
                      <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">ET₀ / Precip</span>
                        <span className="text-[16px] font-bold font-mono text-emerald-400">{orbAssessment.veg.et0_mm_day.toFixed(1)} / {orbAssessment.veg.precip_mm_day.toFixed(1)} mm</span>
                      </div>
                    )}

                    <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Total Area</span>
                      <span className="text-[16px] font-bold font-mono text-white">{(latestRisk.total_area_km2).toFixed(0)} km²</span>
                    </div>

                    <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Discharge Anomaly</span>
                      {(() => {
                        const anom = situationData?.regions.find(r => r.id === selectedRegion.id)?.discharge_anomaly_sigma ?? 0;
                        const anomCol = anom > 2 ? 'text-red-400' : anom > 1 ? 'text-orange-400' : anom > 0.5 ? 'text-yellow-400' : 'text-emerald-400';
                        return (
                          <>
                            <span className={`text-[16px] font-bold font-mono ${anomCol}`}>{anom >= 0 ? '+' : ''}{anom.toFixed(1)}σ</span>
                            <div className={`h-[2px] rounded-full mt-2 w-full bg-[#1a1f2e]`}><div className={`h-full transition-all duration-700 ease-out bg-current ${anomCol}`} style={{ width: `${Math.min(Math.abs(anom) * 25, 100)}%` }} /></div>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                  
                  {((activeOrb === 'infra' && orbAssessment?.infra?.description) || (activeOrb === 'veg' && orbAssessment?.veg?.condition)) && (
                    <div className="mt-2 p-3 bg-white/[0.02] border border-white/5 rounded-lg text-[11px] text-gray-500 font-mono">
                      {activeOrb === 'infra' ? orbAssessment?.infra?.description : orbAssessment?.veg?.condition}
                    </div>
                  )}
                </div>

                {/* Prediction Block */}
                {prediction && (
                  <div className={`bg-[#151A22]/80 border rounded-xl p-3 flex flex-col gap-1 transition-all duration-700 relative overflow-hidden group ${['CRITICAL', 'HIGH'].includes(prediction.predicted_risk_level) ? 'animate-[pulse_4s_ease-in-out_infinite]' : ''}`} style={{ borderColor: ['CRITICAL', 'HIGH'].includes(prediction.predicted_risk_level) ? riskColor(prediction.predicted_risk_level) + '4d' : 'rgba(255,255,255,0.05)', boxShadow: ['CRITICAL', 'HIGH'].includes(prediction.predicted_risk_level) ? `0 0 15px ${riskColor(prediction.predicted_risk_level)}20` : 'none' }}>
                    <div className="absolute top-0 right-0 w-32 h-32 rounded-full blur-[40px] transition-colors" style={{ backgroundColor: riskColor(prediction.predicted_risk_level ?? 'LOW') + '1A', top: '-20px', right: '-20px' }} />
                    <div className="flex items-center justify-between z-10">
                      <span className="text-[10px] font-medium font-mono uppercase tracking-widest text-gray-500 flex items-center gap-1.5"><Activity size={10} className="text-gray-400" /> ML PREDICTION</span>
                      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                      <span className="text-[9px] text-gray-600 font-mono">Model: {(prediction as any).model_version || 'GradientBoosting_v2'}</span>
                    </div>
                    <div className="flex items-center justify-between z-10">
                      <span className="text-[18px] font-bold font-mono uppercase tracking-widest" style={{ color: riskColor(prediction.predicted_risk_level ?? 'LOW'), textShadow: `0 0 12px ${riskColor(prediction.predicted_risk_level ?? 'LOW')}66` }}>
                        {(prediction.predicted_risk_level ?? 'UNKNOWN')} RISK
                      </span>
                      <span className="text-[11px] font-mono text-gray-400">
                        {((prediction.flood_probability ?? 0) * 100).toFixed(1)}% prob · {((prediction.confidence ?? 0) * 100).toFixed(1)}% conf
                      </span>
                    </div>
                  </div>
                )}


                {/* ── Prediction Explainability Panel ── */}
                <button
                  onClick={async () => {
                    if (!selectedRegion || explainLoading) return;
                    setExplainLoading(true);
                    const data = await fetchExplanation(
                      selectedRegion.id,
                      prediction?.predicted_risk_level ?? undefined,
                      prediction?.flood_probability ?? undefined,
                    );
                    if (data && data.ml_prediction) setExplanation(data);
                    setExplainLoading(false);
                  }}
                  disabled={explainLoading}
                  className={`w-full py-3 rounded-xl border text-[13px] uppercase tracking-widest font-mono font-bold transition-all ${explainLoading
                    ? 'border-cyan-500/30 text-cyan-400 bg-cyan-500/10 animate-pulse'
                    : explanation
                      ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20'
                      : 'border-violet-500/40 text-violet-300 bg-violet-500/10 hover:bg-violet-500/20'
                    }`}
                >
                  {explainLoading ? 'Analyzing live data...' : explanation ? '↻ Refresh Explanation' : 'Explain Prediction'}
                </button>

                {explanation && (() => {
                  const ml = explanation.ml_prediction;
                  const fv = ml.feature_values ?? {};
                  const glofasIdx = Math.min(Math.round(fv.glofas_flood_risk ?? 0), 3);
                  const glofasLabels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] as const;
                  const glofasLevel = glofasLabels[glofasIdx];
                  const anomaly = fv.discharge_anomaly_sigma ?? 0;
                  const precip7d = fv.precip_7d_mm ?? 0;
                  const precipAnom = fv.precip_anomaly ?? 0;
                  const soil = fv.soil_saturation ?? 0;
                  const hist = fv.mean_flood_pct ?? 0;
                  type SigStatus = 'elevated' | 'normal' | 'low';
                  const convSignals: { name: string; sub: string; val: string; status: SigStatus }[] = [
                    { name: 'GloFAS v4', sub: 'River discharge', val: `${glofasLevel} • ${anomaly >= 0 ? '+' : ''}${anomaly.toFixed(1)}σ`, status: glofasIdx >= 2 ? 'elevated' : glofasIdx === 1 ? 'normal' : 'low' },
                    { name: 'ERA5 Precipitation', sub: '7-day rainfall', val: `${precip7d.toFixed(0)}mm (${precipAnom >= 0 ? '+' : ''}${precipAnom.toFixed(1)}σ)`, status: precipAnom > 1.0 ? 'elevated' : precipAnom < -1.0 ? 'low' : 'normal' },
                    { name: 'Soil Saturation', sub: 'ERA5/ECMWF IFS', val: `${(soil * 100).toFixed(0)}% saturated`, status: soil > 0.6 ? 'elevated' : soil > 0.3 ? 'normal' : 'low' },
                    { name: 'Historical Baseline', sub: 'Regional DB records', val: hist > 0 ? `avg ${hist.toFixed(1)}% coverage` : 'No history available', status: hist > 10 ? 'elevated' : hist > 2 ? 'normal' : 'low' },
                  ];
                  const elevCount = convSignals.filter(s => s.status === 'elevated').length;
                  const predHigh = ['HIGH', 'CRITICAL'].includes(ml.risk_level);
                  const consensusMsg = predHigh && elevCount >= 3 ? `${elevCount}/4 independent signals elevated — prediction well-supported`
                    : predHigh && elevCount === 2 ? `${elevCount}/4 signals elevated — moderate confidence, plausible`
                    : predHigh && elevCount <= 1 ? `Only ${elevCount}/4 signals elevated — single-source signal, treat with caution`
                    : !predHigh && elevCount === 0 ? '0/4 signals elevated — LOW prediction is well-supported'
                    : `${elevCount}/4 signals elevated — prediction consistent with data`;
                  const consensusColors = predHigh && elevCount >= 3 ? ['text-emerald-400', 'border-emerald-500/30 bg-emerald-500/10 shadow-[0_0_8px_rgba(16,185,129,0.15)]']
                    : predHigh && elevCount <= 1 ? ['text-orange-400', 'border-orange-500/30 bg-orange-500/10 shadow-[0_0_8px_rgba(249,115,22,0.15)]']
                    : ['text-cyan-400', 'border-cyan-500/30 bg-cyan-500/10'];
                  const sigBadge = (s: SigStatus) => s === 'elevated' ? 'text-red-400 border-red-500/40 bg-red-500/20' : s === 'low' ? 'text-cyan-400 border-cyan-500/20 bg-cyan-500/5' : 'text-gray-500 border-gray-600/40 bg-gray-600/10';
                  const sigLabel = (s: SigStatus) => s === 'elevated' ? '↑ Elevated' : s === 'low' ? '↓ Low' : '→ Normal';
                  const sigBorderCol = (s: SigStatus) => s === 'elevated' ? 'border-l-red-500' : s === 'low' ? 'border-l-cyan-500' : 'border-l-gray-600';
                  const sigBarWidth = (name: string) => {
                    if (name.includes('GloFAS')) return `${(glofasIdx / 3) * 100}%`;
                    if (name.includes('Precipitation')) return `${Math.min(Math.abs(precipAnom) / 3 * 100, 100)}%`;
                    if (name.includes('Soil')) return `${soil * 100}%`;
                    if (name.includes('Historical')) return `${Math.min(hist / 20 * 100, 100)}%`;
                    return '0%';
                  };
                  const sigBarCol = (s: SigStatus) => s === 'elevated' ? '#ef4444' : s === 'low' ? '#06b6d4' : '#4b5563';
                  return (
                    <>
                      {/* Prediction explanation text */}
                      <div className="bg-[#0A1628]/90 border border-white/10 rounded-xl p-4 flex flex-col gap-2">
                        <div className="flex items-center gap-2">
                          <span className={`text-[12px] font-medium text-gray-500 uppercase ${textMono} tracking-widest`}>
                            Prediction Explanation
                          </span>
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-cyan-500/30 text-cyan-400">
                            GloFAS Integrated
                          </span>
                        </div>
                        <p className={`text-[13px] text-gray-300 font-sans leading-relaxed`}>{ml.explanation}</p>
                        <p className={`text-[10px] text-gray-600 ${textMono}`}>{ml.model_inputs_source}</p>
                      </div>

                      {/* Signal Convergence Panel */}
                      <div className="bg-[#0B1320]/90 border border-white/[0.06] border-t-white/[0.08] rounded-xl p-4 flex flex-col gap-3">
                        <div className="flex items-center justify-between">
                          <span className={`text-[12px] font-medium text-gray-500 uppercase ${textMono} tracking-widest`}>Signal Verification</span>
                          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${consensusColors[1]} ${consensusColors[0]}`}>{elevCount}/4 agree</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          {convSignals.map(sig => (
                            <div key={sig.name} className={`bg-[#151A22] rounded-lg p-2.5 flex flex-col gap-1 border border-white/5 border-l-[3px] ${sigBorderCol(sig.status)}`}>
                              <div className="flex items-center justify-between gap-1 overflow-hidden">
                                <span className="text-[11px] font-mono text-gray-300 truncate">{sig.name}</span>
                                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${sigBadge(sig.status)}`}>{sigLabel(sig.status)}</span>
                              </div>
                              <span className="text-[10px] font-mono text-gray-600">{sig.sub}</span>
                              <div className="flex items-center justify-between mt-1">
                                <div className="h-[2px] w-12 bg-[#0B0E11] rounded-full overflow-hidden shrink-0 hidden md:block"><div className="h-full rounded-full" style={{ width: sigBarWidth(sig.name), backgroundColor: sigBarCol(sig.status) }} /></div>
                                <span className="text-[11px] font-mono text-gray-400 text-right">{sig.val}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className={`text-[12px] font-mono px-3 py-2 rounded-lg border ${consensusColors[1]} ${consensusColors[0]}`}>{consensusMsg}</div>
                      </div>

                      {/* Top Contributing Factors */}
                      <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-2.5">
                        <div className="flex items-center justify-between">
                          <span className={`text-[12px] font-medium text-gray-500 uppercase ${textMono} tracking-widest`}>
                            PREDICTION DRIVERS
                          </span>
                        </div>
                        <span className="text-[10px] text-gray-600 font-mono -mt-1">
                          Ranked by signal strength
                        </span>
                        {ml.top_drivers.map((d: { feature: string; importance: number; influence: string }) => {
                          const isWater = d.feature.startsWith('discharge') || d.feature.startsWith('forecast_max') || d.feature === 'glofas_flood_risk';
                          const isRain = d.feature.startsWith('precip_');
                          const isSoil = d.feature === 'soil_saturation';
                          const isHist = d.feature === 'mean_flood_pct';
                          const valRaw = ((fv as Record<string, number>)[d.feature] ?? 0);
                          const valFmt = isWater ? `${valRaw >= 0 ? '+' : ''}${valRaw.toFixed(1)}σ` : isRain ? `${valRaw.toFixed(1)}mm` : isSoil ? `${(valRaw * 100).toFixed(0)}%` : isHist ? `${valRaw.toFixed(1)}%` : valRaw.toFixed(1);
                          return (
                          <div key={d.feature} className="flex flex-col gap-0.5 mt-1 hover:bg-white/[0.03] transition-colors duration-200 p-1 -mx-1 rounded">
                            <div className="flex items-center justify-between mb-0.5">
                              <span className={`text-[11px] flex items-center gap-1.5 ${isWater ? 'text-cyan-400' : isRain ? 'text-indigo-400' : isSoil ? 'text-orange-400' : 'text-gray-400'} ${textMono}`}>
                                {isWater ? <Droplets size={12} /> : isRain ? <CloudRain size={12} /> : isSoil ? <Mountain size={12} /> : isHist ? <BarChart3 size={12} /> : <Activity size={12} />}
                                {d.feature.replace(/_/g, ' ')}
                              </span>
                              <div className="flex items-center gap-2">
                                <span className="text-[12px] font-mono text-gray-400 mr-2">{valFmt}</span>
                                <div className="w-16 md:w-20 h-2.5 bg-[#0B0E11] rounded-full overflow-hidden shrink-0 flex">
                                  <div className="h-full rounded-full transition-all duration-500 ease-out"
                                    style={{
                                      width: `${Math.min(d.importance * 400, 100)}%`,
                                      backgroundColor: isWater ? '#22d3ee' : d.importance > 0.15 ? '#f59e0b' : d.importance > 0.08 ? primaryColor : '#4b5563',
                                      boxShadow: (isWater && d.importance > 0.15) ? '0 0 4px rgba(34,211,238,0.3)' : 'none'
                                    }}
                                  />
                                </div>
                                <span className={`text-[12px] text-gray-300 ${textMono} font-bold w-9 text-right`}>{(d.importance * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                            <span className={`text-[11px] text-gray-600 ${textMono} ml-4`}>{d.influence}</span>
                          </div>
                          );
                        })}
                      </div>

                    </>
                  );
                })()}

                {/* Risk History Mini Chart */}
                {(chartData.length > 0 || (activeOrb === 'veg' && regionTrendData)) && (
                  <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4">
                    <span className={`text-[13px] text-gray-500 uppercase ${textMono} block mb-1`}>{currentOrb.chartLabel}</span>
                    <span className="text-[11px] text-gray-600 font-mono block mb-3">
                      {activeOrb === 'flood' ? 'Historical flood coverage from satellite analysis over the past 12 months' : activeOrb === 'infra' ? 'Water change percentage impacting infrastructure over time' : 'Model confidence trend based on data quality and prediction accuracy'}
                    </span>
                    <div className="h-[120px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={displayChartData}>
                          <defs>
                            <linearGradient id="miniGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={primaryColor} stopOpacity={0.3} />
                              <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="date" hide />
                          <YAxis hide />
                          <Area type="monotone" dataKey={chartKey} stroke={primaryColor} fill="url(#miniGrad)" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* ── Historical Trends Button ── */}
                <div className="bg-[#0A1628]/80 border border-cyan-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={openTrends}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-cyan-300 flex items-center gap-2`}>
                      <TrendingUp size={14} className="text-cyan-400" /> Historical Trend Analysis
                    </span>
                    <ChevronRight size={14} className="text-gray-500" />
                  </button>
                </div>

                {/* ── Forecast Panel ── */}
                <div className="bg-[#0A1628]/80 border border-violet-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={async () => {
                      if (!selectedRegion) return;
                      setShowForecast(prev => !prev);
                      if (!forecastData && !forecastLoading) {
                        setForecastLoading(true);
                        const data = await fetchForecast(selectedRegion.id);
                        if (data && !('error' in data)) setForecastData(data);
                        setForecastLoading(false);
                      }
                    }}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <div className="flex flex-col items-start gap-0.5">
                      <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-violet-300 flex items-center gap-2`}>
                        <TrendingUp size={14} className="text-violet-400" /> 6-Month {currentOrb.chartLabel} Forecast
                      </span>
                      <span className="text-[10px] text-gray-600 font-mono mt-0.5">Projected risk trajectory — ERA5 + GloFAS</span>
                    </div>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showForecast ? 'rotate-90' : ''}`} />
                  </button>

                  <AnimatePresence>
                    {showForecast && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-3">
                          {forecastLoading && (
                            <div className="flex items-center justify-center py-8">
                              <div className="w-8 h-8 border-2 border-violet-400/30 border-t-violet-400 rounded-full animate-spin" />
                            </div>
                          )}

                          {forecastData && !forecastLoading && forecastData.summary && forecastData.monthly_forecast && (
                            <>
                              {/* Summary Strip */}
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className={`text-[12px] font-mono px-2.5 py-1 rounded-full border ${(forecastData.summary.overall_trend ?? 'stable') === 'escalating' ? 'bg-red-500/15 text-red-400 border-red-500/30'
                                  : (forecastData.summary.overall_trend ?? 'stable') === 'declining' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                    : 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
                                  }`}>
                                  Trend: {(forecastData.summary.overall_trend ?? 'STABLE').toUpperCase()}
                                </span>
                                <span className="text-[12px] font-mono text-gray-500">
                                  Peak: {forecastData.summary.peak_risk_month} ({(forecastData.summary.peak_probability * 100).toFixed(0)}%)
                                </span>
                              </div>

                              {/* Peak Alert Banner */}
                              {forecastData.summary.peak_probability > 0.6 && (
                                <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-[11px] font-mono text-red-300">
                                  Peak risk expected in {forecastData.summary.peak_risk_month}: {(forecastData.summary.peak_probability * 100).toFixed(0)}%
                                </div>
                              )}

                              {/* Orb-specific Forecast Chart */}
                              <div className="h-[140px]">
                                <ResponsiveContainer width="100%" height="100%">
                                  <AreaChart data={forecastData.monthly_forecast.map(m => {
                                    const val = activeOrb === 'flood' ? m.risk_probability
                                      : activeOrb === 'infra' ? (m.infra_exposure ?? m.risk_probability)
                                      : (m.vegetation_stress_index ?? m.risk_probability);
                                    return {
                                      month: m.month_name.split(' ')[0].slice(0, 3),
                                      risk: Math.round(val * 100),
                                      lower: Math.round((m.confidence_lower ?? val * 0.7) * 100),
                                      upper: Math.round((m.confidence_upper ?? val * 1.3) * 100),
                                    };
                                  })}>
                                    <defs>
                                      <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={primaryColor} stopOpacity={0.4} />
                                        <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
                                      </linearGradient>
                                      <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={primaryColor} stopOpacity={0.1} />
                                        <stop offset="95%" stopColor={primaryColor} stopOpacity={0.05} />
                                      </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="month" tick={{ fill: '#6b7280', fontSize: 11 }} />
                                    <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} domain={[0, 100]} unit="%" />
                                    <RechartsTooltip
                                      contentStyle={{ background: '#0A1628', border: `1px solid ${primaryColor}44`, borderRadius: 8, fontSize: 12, pointerEvents: 'none' }}
                                      position={{ y: 0 }}
                                      offset={15}
                                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                      formatter={(value: any, name: any) => {
                                        if (name === 'upper' || name === 'lower') return [null, null];
                                        return [`${value ?? 0}%`, currentOrb.chartLabel];
                                      }}
                                    />
                                    <Area type="monotone" dataKey="upper" stroke="none" fill="url(#confGrad)" name="upper" />
                                    <Area type="monotone" dataKey="lower" stroke="none" fill="transparent" name="lower" />
                                    <Area type="monotone" dataKey="risk" stroke={primaryColor} fill="url(#forecastGrad)" strokeWidth={2.5} dot={{ r: 3, fill: primaryColor }} name="risk" />
                                  </AreaChart>
                                </ResponsiveContainer>
                              </div>

                              {/* Monthly Cards (6-col) */}
                              <div className="grid grid-cols-6 gap-1">
                                {forecastData.monthly_forecast.slice(0, 6).map(m => {
                                  const val = activeOrb === 'flood' ? m.risk_probability
                                    : activeOrb === 'infra' ? (m.infra_exposure ?? m.risk_probability)
                                    : (m.vegetation_stress_index ?? m.risk_probability);
                                  const level = val >= 0.70 ? 'CRITICAL' : val >= 0.45 ? 'HIGH' : val >= 0.20 ? 'MEDIUM' : 'LOW';
                                  const lvlCol = level === 'CRITICAL' ? '#ef4444' : level === 'HIGH' ? '#f59e0b' : level === 'MEDIUM' ? '#eab308' : '#22c55e';
                                  return (
                                    <div key={m.month} className="bg-[#151A22] rounded-lg p-1.5 flex flex-col items-center gap-0.5 border border-white/5 transition-all duration-300 hover:bg-white/[0.04]">
                                      <div className="text-[9px] text-gray-500 font-mono">{m.month_name.split(' ')[0].slice(0, 3)}</div>
                                      <div className="w-3 rounded-t-sm flex items-end justify-center bg-[#0B0E11] h-[40px] mt-1 overflow-hidden">
                                        <div className="w-full rounded-t-sm transition-all duration-700 ease-out" style={{ backgroundColor: lvlCol, height: `${val * 100}%` }} />
                                      </div>
                                      <div className="text-[11px] font-bold font-mono mt-1" style={{ color: lvlCol }}>
                                        {(val * 100).toFixed(0)}%
                                      </div>
                                      <div className="text-[8px] text-gray-600 font-mono">{level}</div>
                                    </div>
                                  );
                                })}
                              </div>
                            </>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* ── AI Insights Panel ── */}
                <div className="bg-[#0A1628]/80 border border-amber-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={async () => {
                      if (!selectedRegion) return;
                      setShowAiInsights(prev => !prev);
                      if (!nlgSummary && !nlgLoading) {
                        setNlgLoading(true);
                        const data = await fetchNLGSummary(selectedRegion.id);
                        if (data && !('error' in data)) setNlgSummary(data);
                        setNlgLoading(false);
                      }
                    }}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <div className="flex flex-col items-start gap-0.5">
                      <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-amber-300 flex items-center gap-2`}>
                        <Sparkles size={14} className="text-amber-400" /> AI Insights
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-amber-500/15 text-amber-400/50 ml-2">GloFAS v4 + ERA5</span>
                      </span>
                      <span className="text-[10px] text-gray-600 font-mono mt-0.5">Natural language situation analysis</span>
                    </div>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showAiInsights ? 'rotate-90' : ''}`} />
                  </button>

                  <AnimatePresence>
                    {showAiInsights && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-3">
                          {nlgLoading && (
                            <div className="flex items-center justify-center py-8">
                              <div className="w-8 h-8 border-2 border-amber-400/30 border-t-amber-400 rounded-full animate-spin" />
                            </div>
                          )}

                          {nlgSummary && !nlgLoading && (
                            <>
                              <div className="text-[10px] uppercase tracking-widest text-amber-500/50 font-mono mb-1.5">SITUATION</div>
                              {/* Narrative */}
                              <div className="text-[13px] text-gray-300 leading-relaxed font-sans whitespace-pre-line -mt-1">
                                {nlgSummary.narrative.split('**').map((part, i) =>
                                  i % 2 === 1 ? <strong key={i} className="text-white">{part}</strong> : <span key={i}>{part}</span>
                                )}
                              </div>

                              <div className="text-[10px] uppercase tracking-widest text-amber-500/50 font-mono mb-1.5 mt-3">KEY FINDINGS</div>
                              {/* Highlights */}
                              <div className="flex flex-col gap-1.5 -mt-1">
                                {nlgSummary.highlights.map((h, i) => (
                                  <div key={i} className="flex items-start gap-2 text-[13px] font-sans text-gray-300 leading-relaxed">
                                    <span className="text-amber-500/60 mt-0.5">—</span>
                                    {h}
                                  </div>
                                ))}
                              </div>

                              {/* Trend */}
                              {nlgSummary.trend_narrative && (
                                <>
                                  <div className="text-[10px] uppercase tracking-widest text-amber-500/50 font-mono mb-1.5 mt-3">TREND</div>
                                  <div className="text-[13px] text-gray-300 leading-relaxed font-sans -mt-1">
                                    {nlgSummary.trend_narrative.split('**').map((part, i) =>
                                      i % 2 === 1 ? <strong key={i} className="text-white">{part}</strong> : <span key={i}>{part}</span>
                                    )}
                                  </div>
                                </>
                              )}

                              {/* Meta */}
                              <div className="flex items-center justify-between text-[9px] text-gray-700 font-mono mt-2 pt-2 border-t border-white/5">
                                <span>{nlgSummary.engine === 'gpt-4o-mini' ? '✨ GPT-4o' : '⚙ Template Model'}</span>
                                <span>Generated {new Date(nlgSummary.generated_at).toLocaleTimeString()}</span>
                              </div>
                            </>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* ── Multi-Source Analysis Panel ── */}
                <div className="bg-[#0A1628]/80 border border-cyan-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={async () => {
                      if (!selectedRegion) return;
                      setShowFusion(prev => !prev);
                      if (!fusionData && !fusionLoading) {
                        setFusionLoading(true);
                        const data = await fetchFusionAnalysis(selectedRegion.id);
                        if (data && !('error' in data)) setFusionData(data);
                        setFusionLoading(false);
                      }
                    }}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <div className="flex flex-col items-start gap-0.5">
                      <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-cyan-300 flex items-center gap-2`}>
                        <Radio size={14} className="text-cyan-400" /> Multi-Source Analysis
                      </span>
                      <span className="text-[10px] text-gray-600 font-mono mt-0.5">Sensor fusion: optical, thermal, SAR, soil</span>
                    </div>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFusion ? 'rotate-90' : ''}`} />
                  </button>
                  <AnimatePresence>
                    {showFusion && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-2.5">
                          <div className="text-[11px] text-gray-500 font-mono leading-relaxed">
                            Combines NASA MODIS satellite NDVI, ERA5 thermal reanalysis, Open-Meteo weather, and model-estimated radar data using adaptive weighting for comprehensive flood detection across data sources.
                          </div>
                          {fusionLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" /></div>}
                          {fusionData && !fusionLoading && (
                            <>
                              <div className="grid grid-cols-2 gap-2">
                                {[
                                  { label: 'Flood Conf.', value: fusionData.flood_confidence, color: '#00E5FF', desc: 'Combined multi-sensor flood likelihood' },
                                  { label: 'Veg Stress', value: fusionData.vegetation_stress, color: '#4ade80', desc: 'Vegetation health anomaly from optical' },
                                  { label: 'Soil Sat.', value: fusionData.soil_saturation, color: '#c084fc', desc: 'Ground water saturation from weather' },
                                  { label: 'Water Extent', value: fusionData.surface_water_extent_pct, color: '#38bdf8', desc: 'Surface water coverage estimate' },
                                  { label: 'Quality', value: fusionData.quality_score, color: '#facc15', desc: 'Overall data reliability score' },
                                ].map(s => (
                                  <div key={s.label} className="bg-[#151A22] rounded-lg p-2 border border-white/5" title={s.desc}>
                                    <div className="text-[10px] text-gray-500 font-mono">{s.label}</div>
                                    <div className="flex items-center gap-2 mt-1">
                                      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                        <div className="h-full rounded-full" style={{ width: `${(s.value * 100)}%`, backgroundColor: s.color }} />
                                      </div>
                                      <span className="text-[11px] font-mono text-gray-400">{(s.value * 100).toFixed(0)}%</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                              {/* Fusion Weights Breakdown */}
                              {fusionData.fusion_weights && Object.keys(fusionData.fusion_weights).length > 0 && (
                                <div className="bg-[#151A22] rounded-lg p-2.5 border border-white/5">
                                  <div className="text-[12px] text-gray-500 font-mono mb-1.5">ADAPTIVE FUSION WEIGHTS</div>
                                  <div className="text-[12px] text-gray-600 font-mono mb-1">How much each sensor contributes — auto-adjusted based on data quality and cloud cover</div>
                                  {Object.entries(fusionData.fusion_weights).map(([sensor, weight]) => (
                                    <div key={sensor} className="flex items-center gap-2 mb-1">
                                      <span className="text-[13px] font-mono text-gray-400 min-w-[140px] shrink-0">{sensor}</span>
                                      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                        <div className="h-full rounded-full bg-cyan-400" style={{ width: `${(Number(weight) * 100)}%` }} />
                                      </div>
                                      <span className="text-[13px] font-mono text-gray-500 w-10 text-right">{(Number(weight) * 100).toFixed(0)}%</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                              {fusionData.cloud_penetration_used && (
                                <div className="text-[11px] font-mono text-cyan-400 flex items-center gap-1.5">
                                  <Radio size={10} /> Cloud-penetration model active — optical data limited, model-based radar estimate used
                                </div>
                              )}
                              <div className="text-[10px] font-mono text-gray-600">
                                Sensors: {fusionData.sensors_fused?.join(', ')}
                              </div>
                              {fusionData.thermal_anomaly !== 0 && (
                                <div className="text-[11px] font-mono text-gray-500">
                                  Thermal: {fusionData.thermal_anomaly > 0 ? '+' : ''}{fusionData.thermal_anomaly?.toFixed(1)}°C anomaly — {fusionData.thermal_anomaly > 2 ? 'significant heat stress detected' : fusionData.thermal_anomaly < -2 ? 'cooling anomaly detected' : 'within normal range'}
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* ── Compound Risk Panel ── */}
                <div className="bg-[#0A1628]/80 border border-rose-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={async () => {
                      if (!selectedRegion) return;
                      setShowCompound(prev => !prev);
                      if (!compoundData && !compoundLoading) {
                        setCompoundLoading(true);
                        const data = await fetchCompoundRisk(selectedRegion.id);
                        if (data && !('error' in data)) setCompoundData(data);
                        setCompoundLoading(false);
                      }
                    }}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <div className="flex flex-col items-start gap-0.5">
                      <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-rose-300 flex items-center gap-2`}>
                        <Shield size={14} className="text-rose-400" /> Compound Risk
                      </span>
                      <span className="text-[10px] text-gray-600 font-mono mt-0.5">Multi-hazard cascading risk assessment</span>
                    </div>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showCompound ? 'rotate-90' : ''}`} />
                  </button>
                  <AnimatePresence>
                    {showCompound && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-2.5">
                          {compoundLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-rose-400/30 border-t-rose-400 rounded-full animate-spin" /></div>}
                          {compoundData && !compoundLoading && (
                            <>
                              <div className="flex items-center gap-2">
                                <span className={`text-[20px] font-bold font-mono ${compoundData.compound_level === 'CRITICAL' ? 'text-red-400' :
                                  compoundData.compound_level === 'HIGH' ? 'text-orange-400' :
                                    compoundData.compound_level === 'MEDIUM' ? 'text-yellow-400' : 'text-emerald-400'
                                  }`}>{(compoundData.compound_score * 100).toFixed(0)}%</span>
                                <span className="text-[12px] font-mono text-gray-500">{compoundData.compound_level}</span>
                                {compoundData.cascading_amplification > 1.05 && (
                                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-red-500/30 text-red-400 bg-red-500/10">
                                    ×{compoundData.cascading_amplification.toFixed(2)} cascade
                                  </span>
                                )}
                              </div>
                              <div className="flex flex-col gap-1">
                                {compoundData.hazard_layers?.map((h: { name: string; severity: number; status: string; description: string }) => (
                                  <div key={h.name} className="flex items-center gap-2">
                                    <div className={`w-1.5 h-1.5 rounded-full ${h.status === 'active' ? 'bg-red-400' : h.status === 'warning' ? 'bg-yellow-400' : 'bg-gray-600'}`} />
                                    <span className="text-[11px] font-mono text-gray-400 flex-1">{h.name.replace('_', ' ')}</span>
                                    <span className="text-[11px] font-mono text-gray-500">{(h.severity * 100).toFixed(0)}%</span>
                                  </div>
                                ))}
                              </div>
                              {compoundData.interaction_effects?.length > 0 && (
                                <div className="bg-[#151A22] rounded-lg p-2 border border-white/5">
                                  <div className="text-[10px] text-gray-500 font-mono mb-1">INTERACTIONS</div>
                                  {compoundData.interaction_effects.map((ie: { effect: string; amplification: number }, i: number) => (
                                    <div key={i} className="text-[11px] text-rose-300 font-mono">⚡ {ie.effect}</div>
                                  ))}
                                </div>
                              )}
                              {compoundData.recommendations?.length > 0 && (
                                <div className="flex flex-col gap-1 mt-1">
                                  {compoundData.recommendations.slice(0, 3).map((r: string, i: number) => (
                                    <div key={i} className="text-[11px] text-gray-400 flex gap-1.5 items-start">
                                      <span className="text-rose-400">→</span> {r}
                                    </div>
                                  ))}
                                </div>
                              )}
                              {compoundData.data_sources && (
                                <div className="text-[10px] text-gray-600 font-mono mt-1 pt-1 border-t border-white/5">
                                  {compoundData.data_sources.methodology} · {Object.entries(compoundData.data_sources).filter(([k]) => !['methodology', 'flood_probability'].includes(k)).map(([, v]) => v).join(' · ')}
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* ── Financial Impact Panel ── */}
                <div className="bg-[#0A1628]/80 border border-emerald-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={async () => {
                      if (!selectedRegion) return;
                      setShowFinancial(prev => !prev);
                      if (!financialData && !financialLoading) {
                        setFinancialLoading(true);
                        const data = await fetchFinancialImpact(selectedRegion.id);
                        if (data && !('error' in data)) setFinancialData(data);
                        setFinancialLoading(false);
                      }
                    }}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <div className="flex flex-col items-start gap-0.5">
                      <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-emerald-300 flex items-center gap-2`}>
                        <DollarSign size={14} className="text-emerald-400" /> Financial Impact
                      </span>
                      <span className="text-[10px] text-gray-600 font-mono mt-0.5">Asset exposure & recovery cost estimates</span>
                    </div>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFinancial ? 'rotate-90' : ''}`} />
                  </button>
                  <AnimatePresence>
                    {showFinancial && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-2.5">
                          {financialLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" /></div>}
                          {financialData && !financialLoading && (
                            <>
                              {financialData.scenario_based && (
                                <div className="text-[10px] font-mono text-amber-400/80 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1 text-center">
                                  Scenario-based estimate (no active flood detected)
                                </div>
                              )}
                              <div className="text-center">
                                <div className="text-[10px] text-gray-500 font-mono">{financialData.scenario_based ? 'POTENTIAL EXPOSURE' : 'TOTAL EXPOSURE'}</div>
                                <div className="text-[22px] font-bold font-mono text-emerald-400">
                                  {fmtUSD(financialData.total_impact_usd ?? 0)}
                                </div>
                                {financialData.confidence && (
                                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${financialData.confidence === 'high' ? 'border-emerald-500/30 text-emerald-400' : financialData.confidence === 'medium' ? 'border-yellow-500/30 text-yellow-400' : 'border-gray-500/30 text-gray-400'}`}>
                                    {financialData.confidence.toUpperCase()} CONFIDENCE
                                  </span>
                                )}
                              </div>
                              <div className="grid grid-cols-3 gap-1.5">
                                {[
                                  { label: 'Direct', value: financialData.direct_damage_usd },
                                  { label: 'Indirect', value: financialData.indirect_costs_usd },
                                  { label: 'Recovery', value: financialData.recovery_cost_usd },
                                ].map(m => (
                                  <div key={m.label} className="bg-[#151A22] rounded-lg p-2 text-center border border-white/5">
                                    <div className="text-[10px] text-gray-500 font-mono">{m.label}</div>
                                    <div className="text-[12px] font-mono text-gray-300">{fmtUSD(m.value ?? 0)}</div>
                                  </div>
                                ))}
                              </div>
                              <div className="flex items-center gap-3 text-[11px] font-mono text-gray-500">
                                <span>🏥 {financialData.affected_population?.toLocaleString()} affected</span>
                                <span>📊 GDP: {financialData.gdp_impact_pct?.toFixed(3)}%</span>
                              </div>
                              {financialData.mitigation_roi?.length > 0 && (
                                <div className="bg-[#151A22] rounded-lg p-2.5 border border-white/5">
                                  <div className="text-[10px] text-gray-500 font-mono mb-1.5">MITIGATION ROI</div>
                                  {financialData.mitigation_roi.slice(0, 3).map((m: { measure: string; roi_pct: number }, i: number) => (
                                    <div key={i} className="flex items-center justify-between text-[11px] font-mono">
                                      <span className="text-gray-400">{m.measure}</span>
                                      <span className={m.roi_pct > 200 ? 'text-emerald-400' : 'text-yellow-400'}>{m.roi_pct}% ROI</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* ── Model Feedback Panel ── */}
                <div className="bg-[#0A1628]/80 border border-gray-500/20 rounded-xl overflow-hidden shrink-0">
                  <button
                    onClick={() => setShowFeedback(prev => !prev)}
                    className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                  >
                    <div className="flex flex-col items-start gap-0.5">
                      <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-gray-400 flex items-center gap-2`}>
                        <ThumbsUp size={14} className="text-gray-500" /> Model Feedback
                      </span>
                      <span className="text-[10px] text-gray-600 font-mono mt-0.5">Help improve detection accuracy</span>
                    </div>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFeedback ? 'rotate-90' : ''}`} />
                  </button>
                  <AnimatePresence>
                    {showFeedback && selectedRegion && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-3">
                          <div className="text-[12px] text-gray-500 font-mono">
                            Was the flood detection for {selectedRegion.name} accurate?
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={async () => {
                                await submitFeedback({
                                  detection_type: 'flood',
                                  model_prediction: latestRisk?.risk_level || 'UNKNOWN',
                                  user_verdict: 'correct',
                                  region_id: selectedRegion.id,
                                });
                                setShowFeedback(false);
                              }}
                              className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-[12px] font-mono text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                            >
                              <ThumbsUp size={12} /> Correct
                            </button>
                            <button
                              onClick={async () => {
                                await submitFeedback({
                                  detection_type: 'flood',
                                  model_prediction: latestRisk?.risk_level || 'UNKNOWN',
                                  user_verdict: 'incorrect',
                                  region_id: selectedRegion.id,
                                });
                                setShowFeedback(false);
                              }}
                              className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-[12px] font-mono text-red-400 hover:bg-red-500/20 transition-colors"
                            >
                              <ThumbsDown size={12} /> Incorrect
                            </button>
                          </div>
                          <div className="text-[10px] text-gray-600 font-mono">
                            Feedback improves model accuracy through continuous learning
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
                </>
                )}

              </div>

              {/* Actions Footer */}
              <div className="p-4 border-t border-white/10 bg-[#0B1320] flex gap-2 shrink-0">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(window.location.href);
                    // Optional: user could add toast, but simple copy suffices
                  }}
                  className="flex-1 flex items-center justify-center gap-2 py-3 bg-[#151A22] hover:bg-white/10 text-white font-mono text-[11px] uppercase tracking-widest border border-white/10 hover:border-cyan-500/40 rounded-lg transition-colors group"
                >
                  <Share2 size={14} className="text-gray-400 group-hover:text-cyan-400 transition-colors" /> Copy Link
                </button>
                <a href={selectedRegion ? getReportDownloadUrl(selectedRegion.id) : '#'} target="_blank" rel="noopener noreferrer"
                  className="flex-[2] flex items-center justify-center gap-2 py-3 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 font-mono text-[11px] uppercase tracking-widest border border-cyan-500/30 hover:border-cyan-400/50 rounded-lg transition-colors group">
                  <Download size={14} className="text-cyan-500 group-hover:text-cyan-400 transition-colors" /> PDF Report
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ AD-HOC LOCATION PANEL (RIGHT) ═══ */}
      <AnimatePresence>
        {adHocLocation && !selectedRegion && (
          <motion.div key={`adhoc-${adHocLocation.lat}-${adHocLocation.lon}`} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }} className="absolute right-6 top-28 bottom-32 w-[400px] z-20 flex flex-col gap-4 overflow-hidden">
            <div className={`${glassClass} flex flex-col h-full overflow-hidden`}>

              {/* Header */}
              <div className="p-6 border-b border-white/10 bg-black/40 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col gap-1">
                    {adHocData ? (
                      (() => {
                        const liveLevel = adHocData.prediction?.predicted_risk_level ?? adHocData.detection?.detected_risk_level ?? 'LOW';
                        const displayLevel = activeOrb === 'infra' ? (orbAssessment?.infra?.risk_level ?? liveLevel) : activeOrb === 'veg' ? (orbAssessment?.veg?.risk_level ?? liveLevel) : liveLevel;
                        return (
                          <span className={`text-[16px] uppercase tracking-widest ${textMono} font-bold flex items-center gap-2`} style={{ color: riskColor(displayLevel), textShadow: `0 0 12px ${riskColor(displayLevel)}66` }}>
                            <AlertTriangle size={16} className={(displayLevel === 'HIGH' || displayLevel === 'CRITICAL') ? 'animate-pulse' : ''} />
                            {displayLevel} RISK
                            <span className="px-2 py-0.5 rounded text-[10px] uppercase font-mono border ml-1 tracking-wider" style={{ borderColor: riskColor(displayLevel) + '40', backgroundColor: riskColor(displayLevel) + '20' }}>
                              {activeOrb === 'infra' ? 'INFRA EXPOSURE' : activeOrb === 'veg' ? 'VEG STRESS' : (adHocData.prediction ? 'ML PREDICTION' : 'LIVE DETECTION')}
                            </span>
                          </span>
                        );
                      })()
                    ) : (
                      <span className={`text-[16px] uppercase tracking-widest ${textMono} font-bold text-cyan-400 flex items-center gap-2`} style={{ textShadow: `0 0 12px rgba(34,211,238,0.4)` }}>
                        <Activity size={16} className="animate-pulse" /> Analyzing...
                      </span>
                    )}
                    <span className="text-white text-[15px] font-semibold mt-1 flex items-center justify-between">
                      <span>{adHocLocation.name} — {currentOrb.panelTitle}</span>
                      {adHocData?.detection?.timestamp && <span className="text-[10px] text-gray-500 font-mono tracking-widest uppercase truncate ml-2">Last assessed: Just now</span>}
                    </span>
                  </div>
                  <button onClick={() => { setAdHocLocation(null); setAdHocData(null); setAdHocExplanation(null); setAdHocTrendData(null); setForecastData(null); setNlgSummary(null); setShowForecast(false); setShowAiInsights(false); setFusionData(null); setCompoundData(null); setFinancialData(null); setShowFusion(false); setShowCompound(false); setShowFinancial(false); setShowFeedback(false); }} className="text-gray-500 hover:text-white"><X size={18} /></button>
                </div>
              </div>

              {/* Content */}
              <div className="flex-grow p-6 flex flex-col gap-6 overflow-y-auto">
                {adHocLoading && (
                  <div className="flex items-center justify-center py-12">
                    <div className="w-10 h-10 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                  </div>
                )}

                {adHocData && !adHocLoading && (() => {
                  const det = adHocData.detection;
                  const pred = adHocData.prediction;
                  const val = adHocData.validation;
                  return (
                    <>
                      {/* Data Grid — matches active region */}
                      {det && (
                        <div className="grid grid-cols-3 gap-3">
                          <div className="bg-[#151A22]/80 rounded-xl p-3 flex flex-col gap-1">
                            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-mono">FLOOD AREA</span>
                            <span className="text-[18px] font-bold font-mono text-white">{det.flood_area_km2?.toFixed(0) || '—'} km²</span>
                            <div className="flex items-center gap-2 mt-1">
                              <div className="h-[3px] flex-1 bg-[#1a1f2e] rounded-full overflow-hidden">
                                <div className="h-full transition-all duration-700 ease-out" style={{ width: `${Math.min((det.flood_probability ?? 0) * 500, 100)}%`, backgroundColor: riskColor(det.detected_risk_level) }} />
                              </div>
                              <span className="text-[10px] font-mono text-gray-400">{((det.flood_probability ?? 0) * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                          <div className="bg-[#151A22]/80 rounded-xl p-3 flex flex-col gap-1">
                            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-mono">RISK PROB.</span>
                            {pred ? (
                              <>
                                <span className="text-[18px] font-bold font-mono text-white">{(pred.flood_probability * 100).toFixed(0)}%</span>
                                <div className="flex items-center gap-2 mt-1">
                                  <div className="h-[3px] flex-1 bg-[#1a1f2e] rounded-full overflow-hidden">
                                    <div className="h-full transition-all duration-700 ease-out" style={{ width: `${pred.flood_probability * 100}%`, backgroundColor: riskColor(pred.predicted_risk_level) }} />
                                  </div>
                                  <span className="text-[10px] font-mono" style={{ color: riskColor(pred.predicted_risk_level) }}>{pred.predicted_risk_level}</span>
                                </div>
                              </>
                            ) : (
                              <span className="text-[18px] font-bold font-mono text-gray-600 opacity-50">--</span>
                            )}
                          </div>
                          <div className="bg-[#151A22]/80 rounded-xl p-3 flex flex-col gap-1">
                            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-mono">CONFIDENCE</span>
                            {pred ? (
                              <>
                                <span className="text-[18px] font-bold font-mono text-white">{(pred.confidence * 100).toFixed(0)}%</span>
                                <div className="flex items-center gap-2 mt-1">
                                  <div className="h-[3px] flex-1 bg-[#1a1f2e] rounded-full overflow-hidden">
                                    <div className="h-full transition-all duration-700 ease-out bg-cyan-400" style={{ width: `${pred.confidence * 100}%` }} />
                                  </div>
                                  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                  <span className="text-[10px] font-mono text-cyan-400">T{((pred as any).feature_values?.glofas_flood_risk) ? Math.min(Math.round((pred as any).feature_values.glofas_flood_risk), 3) : 2}</span>
                                </div>
                              </>
                            ) : (
                              <span className="text-[18px] font-bold font-mono text-gray-600 opacity-50">--</span>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Assessment Details — matches active region */}
                      {det && (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[12px] font-medium font-mono uppercase tracking-widest text-gray-500">Assessment Details</span>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-white/5 text-gray-500 bg-white/5">[ERA5 + GloFAS]</span>
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                              <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">{currentOrb.metricLabel}</span>
                              {orbAssessmentLoading && activeOrb !== 'flood' ? (
                                <span className="text-gray-500 text-[16px] font-mono">loading...</span>
                              ) : (() => {
                                const valPct = activeOrb === 'infra' && orbAssessment?.infra ? orbAssessment.infra.exposure_score * 100 : activeOrb === 'veg' && orbAssessment?.veg ? orbAssessment.veg.stress_index * 100 : (det.flood_probability ?? 0) * 100;
                                const col = valPct > 10 ? 'text-red-400' : valPct > 5 ? 'text-orange-400' : valPct > 2 ? 'text-yellow-400' : 'text-emerald-400';
                                return (
                                  <>
                                    <span className={`text-[16px] font-bold font-mono ${col}`}>{valPct.toFixed(1)}%</span>
                                    <div className={`h-[2px] rounded-full mt-2 w-full bg-[#1a1f2e]`}><div className={`h-full transition-all duration-700 ease-out bg-current ${col}`} style={{ width: `${Math.min(valPct, 100)}%` }} /></div>
                                  </>
                                );
                              })()}
                            </div>

                            <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                              <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Water Change</span>
                              <div className="flex items-center gap-2">
                                <span className={`text-[16px] font-bold font-mono ${det.discharge_anomaly_sigma > 0 ? 'text-red-400' : 'text-emerald-400'}`}>{det.discharge_anomaly_sigma > 0 ? '+' : ''}{(det.discharge_anomaly_sigma * 10).toFixed(1)}%</span>
                                <span className={`text-[10px] font-mono ${det.discharge_anomaly_sigma > 0 ? 'text-red-400' : det.discharge_anomaly_sigma < 0 ? 'text-emerald-400' : 'text-gray-500'}`}>{det.discharge_anomaly_sigma > 0 ? '▲ expanding' : det.discharge_anomaly_sigma < 0 ? '▼ receding' : '→ stable'}</span>
                              </div>
                            </div>

                            {activeOrb === 'infra' && orbAssessment?.infra && (
                              <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                                <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Soil Saturation</span>
                                <span className={`text-[16px] font-bold font-mono ${orbAssessment.infra.soil_saturation > 0.8 ? 'text-red-400' : orbAssessment.infra.soil_saturation > 0.6 ? 'text-orange-400' : orbAssessment.infra.soil_saturation > 0.4 ? 'text-yellow-400' : 'text-emerald-400'}`}>{(orbAssessment.infra.soil_saturation * 100).toFixed(0)}%</span>
                              </div>
                            )}
                            {activeOrb === 'veg' && orbAssessment?.veg && (
                              <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                                <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">ET₀ / Precip</span>
                                <span className="text-[16px] font-bold font-mono text-emerald-400">{orbAssessment.veg.et0_mm_day.toFixed(1)} / {orbAssessment.veg.precip_mm_day.toFixed(1)} mm</span>
                              </div>
                            )}

                            <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                              <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Total Area</span>
                              <span className="text-[16px] font-bold font-mono text-white">{det.flood_area_km2?.toFixed(0) || '—'} km²</span>
                            </div>

                            <div className="bg-[#151A22]/60 hover:bg-[#151A22]/80 transition-colors duration-200 rounded-lg p-3 flex flex-col">
                              <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono mb-1">Discharge Anomaly</span>
                              <span className={`text-[16px] font-bold font-mono ${det.discharge_anomaly_sigma > 2 ? 'text-red-400' : det.discharge_anomaly_sigma > 1 ? 'text-orange-400' : det.discharge_anomaly_sigma > 0.5 ? 'text-yellow-400' : 'text-emerald-400'}`}>{det.discharge_anomaly_sigma >= 0 ? '+' : ''}{det.discharge_anomaly_sigma.toFixed(1)}σ</span>
                              <div className={`h-[2px] rounded-full mt-2 w-full bg-[#1a1f2e]`}><div className={`h-full transition-all duration-700 ease-out bg-current ${det.discharge_anomaly_sigma > 1.5 ? 'text-red-400' : 'text-emerald-400'}`} style={{ width: `${Math.min(Math.abs(det.discharge_anomaly_sigma) * 25, 100)}%` }} /></div>
                            </div>
                          </div>
                          
                          {((activeOrb === 'infra' && orbAssessment?.infra?.description) || (activeOrb === 'veg' && orbAssessment?.veg?.condition)) && (
                            <div className="mt-2 p-3 bg-white/[0.02] border border-white/5 rounded-lg text-[11px] text-gray-500 font-mono">
                              {activeOrb === 'infra' ? orbAssessment?.infra?.description : orbAssessment?.veg?.condition}
                            </div>
                          )}
                        </div>
                      )}

                      {/* ML Prediction */}
                      {pred && (
                        <div className={`bg-[#151A22]/80 border rounded-xl p-3 flex flex-col gap-1 transition-all duration-700 relative overflow-hidden group ${['CRITICAL', 'HIGH'].includes(pred.predicted_risk_level ?? pred.risk_level) ? 'animate-[pulse_4s_ease-in-out_infinite]' : ''}`} style={{ borderColor: ['CRITICAL', 'HIGH'].includes(pred.predicted_risk_level ?? pred.risk_level) ? riskColor(pred.predicted_risk_level ?? pred.risk_level) + '4d' : 'rgba(255,255,255,0.05)', boxShadow: ['CRITICAL', 'HIGH'].includes(pred.predicted_risk_level ?? pred.risk_level) ? `0 0 15px ${riskColor(pred.predicted_risk_level ?? pred.risk_level)}20` : 'none' }}>
                          <div className="absolute top-0 right-0 w-32 h-32 rounded-full blur-[40px] transition-colors" style={{ backgroundColor: riskColor(pred.predicted_risk_level ?? pred.risk_level ?? 'LOW') + '1A', top: '-20px', right: '-20px' }} />
                          <div className="flex items-center justify-between z-10">
                            <span className="text-[10px] font-medium font-mono uppercase tracking-widest text-gray-500 flex items-center gap-1.5"><Activity size={10} className="text-gray-400" /> ML PREDICTION</span>
                            <span className="text-[9px] text-gray-600 font-mono">Model: {pred.model_version || 'GradientBoosting_v2'}</span>
                          </div>
                          <div className="flex items-center justify-between z-10">
                            <span className="text-[18px] font-bold font-mono uppercase tracking-widest" style={{ color: riskColor(pred.predicted_risk_level ?? pred.risk_level ?? 'LOW'), textShadow: `0 0 12px ${riskColor(pred.predicted_risk_level ?? pred.risk_level ?? 'LOW')}66` }}>
                              {(pred.predicted_risk_level ?? pred.risk_level ?? 'UNKNOWN')} RISK
                            </span>
                            <span className="text-[11px] font-mono text-gray-400">
                              {((pred.flood_probability ?? pred.probability ?? 0) * 100).toFixed(1)}% prob · {((pred.confidence ?? 0) * 100).toFixed(1)}% conf
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Automated Detection */}
                      {det && (
                        <div className="bg-[#0A1628]/90 border rounded-xl p-4 flex flex-col gap-3" style={{ borderColor: riskColor(det.detected_risk_level) + '40' }}>
                          <span className={`text-[13px] text-gray-500 uppercase ${textMono} flex items-center gap-2`}>
                            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" /> Automated Detection
                          </span>
                          <div className="flex items-center justify-between">
                            <span className="text-[13px] text-gray-400 font-mono">Detected Risk</span>
                            <span className="text-[15px] font-bold font-mono" style={{ color: riskColor(det.detected_risk_level) }}>{det.detected_risk_level}</span>
                          </div>
                          <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-[13px] font-mono">
                            <span className="text-gray-500">River Discharge</span>
                            <span className="text-right text-cyan-400">{det.river_discharge_m3s} m³/s</span>
                            <span className="text-gray-500">Discharge Anomaly</span>
                            <span className={`text-right ${det.discharge_anomaly_sigma > 1.5 ? 'text-red-400' : det.discharge_anomaly_sigma > 0.8 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                              {det.discharge_anomaly_sigma > 0 ? '+' : ''}{det.discharge_anomaly_sigma}σ
                            </span>
                            <span className="text-gray-500">7-Day Rainfall</span>
                            <span className="text-right text-blue-400">{det.rainfall_7d_mm} mm</span>
                            <span className="text-gray-500">Forecast Rain</span>
                            <span className="text-right text-blue-300">{det.rainfall_forecast_mm} mm</span>
                            <span className="text-gray-500">Elevation</span>
                            <span className="text-right text-gray-400">{det.elevation_m} m</span>
                            <span className="text-gray-500">Confidence</span>
                            <span className="text-right text-gray-400">{(det.confidence_score * 100).toFixed(0)}%</span>
                          </div>
                          {det.alert_triggered && (
                            <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-[13px] text-red-300 font-mono">
                              {det.alert_message}
                            </div>
                          )}
                          <div className="text-[12px] text-gray-600 font-mono">Sources: {det.data_sources?.join(', ')}</div>
                        </div>
                      )}

                      {/* Prediction Explainability Button */}
                      <button
                        onClick={async () => {
                          if (!adHocLocation || explainLoading) return;
                          setExplainLoading(true);
                          const data = await explainLocation(
                            adHocLocation.lat,
                            adHocLocation.lon,
                            adHocData?.prediction?.predicted_risk_level ?? undefined,
                            adHocData?.prediction?.flood_probability ?? undefined,
                          );
                          if (data && data.ml_prediction) setAdHocExplanation(data);
                          setExplainLoading(false);
                        }}
                        disabled={explainLoading}
                        className={`w-full py-3 rounded-xl border text-[13px] uppercase tracking-widest font-mono font-bold transition-all ${explainLoading
                          ? 'border-cyan-500/30 text-cyan-400 bg-cyan-500/10 animate-pulse'
                          : adHocExplanation
                            ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20'
                            : 'border-violet-500/40 text-violet-300 bg-violet-500/10 hover:bg-violet-500/20'
                          }`}
                      >
                        {explainLoading ? 'Analyzing live data...' : adHocExplanation ? '↻ Refresh Explanation' : 'Explain Prediction'}
                      </button>

                      {/* Explanation Panels */}
                      {adHocExplanation && (() => {
                        const ml = adHocExplanation.ml_prediction;
                        const fv = ml.feature_values ?? {};
                        const glofasIdx = Math.min(Math.round(fv.glofas_flood_risk ?? 0), 3);
                        const glofasLabels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] as const;
                        const glofasLevel = glofasLabels[glofasIdx];
                        const anomaly = fv.discharge_anomaly_sigma ?? 0;
                        const precip7d = fv.precip_7d_mm ?? 0;
                        const precipAnom = fv.precip_anomaly ?? 0;
                        const soil = fv.soil_saturation ?? 0;
                        const hist = fv.mean_flood_pct ?? 0;
                        type SigStatus2 = 'elevated' | 'normal' | 'low';
                        const convSignals2: { name: string; sub: string; val: string; status: SigStatus2 }[] = [
                          { name: 'GloFAS v4', sub: 'River discharge', val: `${glofasLevel} • ${anomaly >= 0 ? '+' : ''}${anomaly.toFixed(1)}σ`, status: glofasIdx >= 2 ? 'elevated' : glofasIdx === 1 ? 'normal' : 'low' },
                          { name: 'ERA5 Precipitation', sub: '7-day rainfall', val: `${precip7d.toFixed(0)}mm (${precipAnom >= 0 ? '+' : ''}${precipAnom.toFixed(1)}σ)`, status: precipAnom > 1.0 ? 'elevated' : precipAnom < -1.0 ? 'low' : 'normal' },
                          { name: 'Soil Saturation', sub: 'ERA5/ECMWF IFS', val: `${(soil * 100).toFixed(0)}% saturated`, status: soil > 0.6 ? 'elevated' : soil > 0.3 ? 'normal' : 'low' },
                          { name: 'Historical Baseline', sub: 'Regional DB records', val: hist > 0 ? `avg ${hist.toFixed(1)}% coverage` : 'No history available', status: hist > 10 ? 'elevated' : hist > 2 ? 'normal' : 'low' },
                        ];
                        const elevCount2 = convSignals2.filter(s => s.status === 'elevated').length;
                        const predHigh2 = ['HIGH', 'CRITICAL'].includes(ml.risk_level);
                        const consensusMsg2 = predHigh2 && elevCount2 >= 3 ? `${elevCount2}/4 independent signals elevated — prediction well-supported`
                          : predHigh2 && elevCount2 === 2 ? `${elevCount2}/4 signals elevated — moderate confidence, plausible`
                          : predHigh2 && elevCount2 <= 1 ? `Only ${elevCount2}/4 signals elevated — single-source signal, treat with caution`
                          : !predHigh2 && elevCount2 === 0 ? '0/4 signals elevated — LOW prediction is well-supported'
                          : `${elevCount2}/4 signals elevated — prediction consistent with data`;
                        const consensusColors2 = predHigh2 && elevCount2 >= 3 ? ['text-emerald-400', 'border-emerald-500/30 bg-emerald-500/10']
                          : predHigh2 && elevCount2 <= 1 ? ['text-orange-400', 'border-orange-500/30 bg-orange-500/10']
                          : ['text-cyan-400', 'border-cyan-500/30 bg-cyan-500/10'];
                        const sigBadge2 = (s: SigStatus2) => s === 'elevated' ? 'text-red-400 border-red-500/30 bg-red-500/10' : s === 'low' ? 'text-cyan-400 border-cyan-500/20 bg-cyan-500/5' : 'text-gray-500 border-gray-600/40 bg-gray-600/10';
                        const sigLabel2 = (s: SigStatus2) => s === 'elevated' ? '↑ Elevated' : s === 'low' ? '↓ Low' : '→ Normal';
                        return (
                          <>
                            {/* Prediction explanation text */}
                            <div className="bg-[#0A1628]/90 border border-white/10 rounded-xl p-4 flex flex-col gap-2">
                              <div className="flex items-center gap-2">
                                <span className={`text-[13px] text-gray-500 uppercase ${textMono} tracking-widest`}>
                                  Prediction Explanation
                                </span>
                                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-cyan-500/30 text-cyan-400">
                                  GloFAS Integrated
                                </span>
                              </div>
                              <p className={`text-[13px] text-gray-300 ${textMono} leading-relaxed`}>{ml.explanation}</p>
                              <p className={`text-[11px] text-gray-600 ${textMono}`}>{ml.model_inputs_source}</p>
                            </div>

                            {/* Signal Verification Panel */}
                            <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-3">
                              <div className="flex items-center justify-between">
                                <span className={`text-[12px] font-medium font-mono uppercase tracking-widest text-gray-500 mb-1`}>Signal Verification</span>
                                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${consensusColors2[1]} ${consensusColors2[0]}`}>{elevCount2}/4 agree</span>
                              </div>
                              <div className="grid grid-cols-2 gap-2">
                                {convSignals2.map(sig => {
                                  const sigBorderCol = sig.status === 'elevated' ? 'border-red-500/50' : sig.status === 'low' ? 'border-cyan-500/50' : 'border-gray-500/50';
                                  const sigBarWidth = sig.status === 'elevated' ? '100%' : sig.status === 'low' ? '30%' : '60%';
                                  const sigBarCol = sig.status === 'elevated' ? 'bg-red-400' : sig.status === 'low' ? 'bg-cyan-400' : 'bg-gray-500';
                                  
                                  return (
                                    <div key={sig.name} className={`bg-[#0A1628]/60 rounded-lg p-2.5 flex flex-col gap-1.5 border-l-2 ${sigBorderCol} border-y border-y-white/5 border-r border-r-white/5 relative overflow-hidden`}>
                                      <div className="absolute bottom-0 left-0 h-[2px] bg-white/5 w-full">
                                        <div className={`h-full ${sigBarCol} transition-all duration-700 ease-out opacity-20`} style={{ width: sigBarWidth }} />
                                      </div>
                                      <div className="flex items-center justify-between gap-1 relative z-10">
                                        <span className="text-[11px] font-mono text-gray-300 truncate">{sig.name}</span>
                                        <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${sigBadge2(sig.status)}`}>{sigLabel2(sig.status)}</span>
                                      </div>
                                      <span className="text-[10px] font-mono text-gray-500 relative z-10">{sig.sub}</span>
                                      <span className="text-[12px] font-mono font-bold text-white relative z-10">{sig.val}</span>
                                    </div>
                                  );
                                })}
                              </div>
                              <div className={`text-[11px] uppercase tracking-widest font-mono font-bold px-3 py-2.5 rounded-lg border flex items-center gap-2 ${consensusColors2[1]} ${consensusColors2[0]} ${predHigh2 ? 'animate-pulse shadow-[0_0_15px_rgba(16,185,129,0.15)]' : ''}`}>
                                {predHigh2 ? <Activity size={14} /> : <ThumbsUp size={14} />} {consensusMsg2}
                              </div>
                            </div>

                            {/* Prediction Drivers Block */}
                            <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-3">
                              <div className="flex items-center justify-between">
                                <span className={`text-[12px] font-medium font-mono uppercase tracking-widest text-gray-500`}>Prediction Drivers</span>
                                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-white/5 text-gray-500 bg-white/5">9 signals from GloFAS v4 + ERA5 + ECMWF</span>
                              </div>
                              <div className="flex flex-col gap-2.5 mt-1">
                                {ml.top_drivers.map((d: { feature: string; importance: number; influence: string }) => {
                                  let Icon = Activity;
                                  if (d.feature.includes('discharge')) Icon = Droplets;
                                  else if (d.feature.includes('precip') || d.feature.includes('rainfall')) Icon = CloudRain;
                                  else if (d.feature.includes('elevation') || d.feature.includes('slope')) Icon = Mountain;
                                  else if (d.feature.includes('glofas')) Icon = BarChart3;

                                  const isWet = d.feature.startsWith('discharge') || d.feature.startsWith('forecast_max') || d.feature === 'glofas_flood_risk' || d.feature.includes('precip');
                                  const labelCol = isWet ? 'text-cyan-400' : 'text-gray-300';
                                  const barCol = isWet ? '#22d3ee' : d.importance > 0.15 ? '#f59e0b' : d.importance > 0.08 ? primaryColor : '#4b5563';

                                  return (
                                    <div key={d.feature} className="flex flex-col gap-1 group">
                                      <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                          <Icon size={12} className={labelCol} />
                                          <span className={`text-[12px] font-medium ${labelCol} ${textMono} group-hover:text-white transition-colors`}>
                                            {d.feature.replace(/_/g, ' ')}
                                          </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                          <span className={`text-[10px] text-gray-500 ${textMono} uppercase tracking-wider`}>{d.influence}</span>
                                          <span className={`text-[13px] text-white ${textMono} font-bold bg-[#0A1628] px-1.5 py-0.5 rounded`}>{(d.importance * 100).toFixed(1)}%</span>
                                        </div>
                                      </div>
                                      <div className="w-full h-1.5 bg-[#0B0E11] rounded-full overflow-hidden mt-0.5">
                                        <div className="h-full rounded-full transition-all duration-700 ease-out" style={{
                                          width: `${Math.min(d.importance * 400, 100)}%`,
                                          backgroundColor: barCol,
                                          boxShadow: `0 0 8px ${barCol}40`
                                        }} />
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          </>
                        );
                      })()}

                      {/* ── Risk History Mini Chart (Ad-Hoc, from ERA5 trend data) ── */}
                      {displayChartData.length > 0 && (
                        <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4">
                          <span className={`text-[13px] text-gray-500 uppercase ${textMono} block mb-1`}>{currentOrb.chartLabel}</span>
                          <span className="text-[11px] text-gray-600 font-mono block mb-3">
                            {activeOrb === 'flood' ? 'Historical flood coverage from satellite analysis over the past 12 months' : activeOrb === 'infra' ? 'Water change percentage impacting infrastructure over time' : 'Model confidence trend based on data quality and prediction accuracy'}
                          </span>
                          <div className="h-[120px]">
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart data={displayChartData}>
                                <defs>
                                  <linearGradient id="miniGradAdHoc" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={primaryColor} stopOpacity={0.3} />
                                    <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
                                  </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="date" hide />
                                <YAxis hide />
                                <Area type="monotone" dataKey={chartKey} stroke={primaryColor} fill="url(#miniGradAdHoc)" strokeWidth={2} />
                              </AreaChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      )}

                      {/* ── Historical Trends Button (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-cyan-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={openTrends}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <span className={`text-[13px] uppercase ${textMono} tracking-widest text-cyan-300 flex items-center gap-2`}>
                            <TrendingUp size={14} className="text-cyan-400" /> Historical Trend Analysis
                          </span>
                          <ChevronRight size={14} className="text-gray-500" />
                        </button>
                      </div>

                      {/* ── 6-Month Forecast Panel (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-violet-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={async () => {
                            if (!adHocLocation) return;
                            setShowForecast(prev => !prev);
                            if (!forecastData && !forecastLoading) {
                              setForecastLoading(true);
                              const data = await forecastLocation(adHocLocation.lat, adHocLocation.lon, adHocLocation.name);
                              if (data && !('error' in data)) setForecastData(data);
                              setForecastLoading(false);
                            }
                          }}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-violet-300 flex items-center gap-2`}>
                              <TrendingUp size={14} className="text-violet-400" /> 6-Month {currentOrb.chartLabel} Forecast
                            </span>
                            <span className="text-[10px] text-gray-600 font-mono mt-0.5">Medium-range seasonal projections</span>
                          </div>
                          <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showForecast ? 'rotate-90' : ''}`} />
                        </button>
                        <AnimatePresence>
                          {showForecast && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                              <div className="px-4 pb-4 flex flex-col gap-3">
                                {forecastLoading && (
                                  <div className="flex items-center justify-center py-8">
                                    <div className="w-8 h-8 border-2 border-violet-400/30 border-t-violet-400 rounded-full animate-spin" />
                                  </div>
                                )}
                                {forecastData && !forecastLoading && forecastData.summary && forecastData.monthly_forecast && (
                                  <>
                                    <div className="flex items-center gap-2 flex-wrap">
                                      <span className={`text-[12px] font-mono px-2.5 py-1 rounded-full border ${(forecastData.summary.overall_trend ?? 'stable') === 'escalating' ? 'bg-red-500/15 text-red-400 border-red-500/30'
                                        : (forecastData.summary.overall_trend ?? 'stable') === 'declining' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                          : 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
                                        }`}>
                                        Trend: {(forecastData.summary.overall_trend ?? 'STABLE').toUpperCase()}
                                      </span>
                                      <span className="text-[12px] font-mono px-2.5 py-1 rounded-full border border-orange-500/30 text-orange-400 bg-orange-500/15 flex items-center gap-1.5">
                                        <AlertTriangle size={12} /> PEAK ALERT: {forecastData.summary.peak_risk_month} ({(forecastData.summary.peak_probability * 100).toFixed(0)}%)
                                      </span>
                                    </div>
                                    <div className="h-[140px]">
                                      <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={forecastData.monthly_forecast.map(m => {
                                          const val = activeOrb === 'flood' ? m.risk_probability
                                            : activeOrb === 'infra' ? (m.infra_exposure ?? m.risk_probability)
                                            : (m.vegetation_stress_index ?? m.risk_probability);
                                          return {
                                            month: m.month_name.split(' ')[0].slice(0, 3),
                                            risk: Math.round(val * 100),
                                          };
                                        })}>
                                          <defs>
                                            <linearGradient id="forecastGradAdhoc" x1="0" y1="0" x2="0" y2="1">
                                              <stop offset="5%" stopColor={primaryColor} stopOpacity={0.4} />
                                              <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
                                            </linearGradient>
                                          </defs>
                                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                          <XAxis dataKey="month" tick={{ fill: '#6b7280', fontSize: 11 }} />
                                          <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} domain={[0, 100]} unit="%" />
                                          <RechartsTooltip
                                            contentStyle={{ background: '#0A1628', border: `1px solid ${primaryColor}44`, borderRadius: 8, fontSize: 12, pointerEvents: 'none' }}
                                            position={{ y: 0 }}
                                            offset={15}
                                            formatter={(value: number | string | undefined) => [`${value ?? 0}%`, currentOrb.chartLabel]}
                                          />
                                          <Area type="monotone" dataKey="risk" stroke={primaryColor} fill="url(#forecastGradAdhoc)" strokeWidth={2.5} dot={{ r: 3, fill: primaryColor }} />
                                        </AreaChart>
                                      </ResponsiveContainer>
                                    </div>
                                    <div className="grid grid-cols-6 gap-1.5 mt-2">
                                      {forecastData.monthly_forecast.slice(0, 6).map(m => {
                                        const val = activeOrb === 'flood' ? m.risk_probability
                                          : activeOrb === 'infra' ? (m.infra_exposure ?? m.risk_probability)
                                          : (m.vegetation_stress_index ?? m.risk_probability);
                                        const level = val >= 0.70 ? 'CRITICAL' : val >= 0.45 ? 'HIGH' : val >= 0.20 ? 'MEDIUM' : 'LOW';
                                        const hPct = Math.max(Math.min(val * 100, 100), 10);
                                        const col = level === 'CRITICAL' ? '#ef4444' : level === 'HIGH' ? '#f59e0b' : level === 'MEDIUM' ? '#eab308' : '#22c55e';
                                        
                                        return (
                                          <div key={m.month} className="bg-[#151A22]/50 hover:bg-[#151A22] transition-colors rounded-lg p-1.5 flex flex-col items-center border border-white/5 relative group pb-6">
                                            <div className="text-[9px] text-gray-500 font-mono mt-0.5 text-center">{m.month_name.split(' ')[0].slice(0, 3)}</div>
                                            <div className="h-12 w-full flex items-end justify-center mt-2 mb-1">
                                              <div className="w-[14px] rounded-sm transition-all duration-700 ease-out" style={{ height: `${hPct}%`, backgroundColor: col, opacity: 0.8 }} />
                                            </div>
                                            <div className="text-[10px] font-bold font-mono text-center absolute bottom-1.5" style={{ color: col }}>
                                              {(val * 100).toFixed(0)}
                                            </div>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {/* ── AI Insights Panel (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-amber-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={async () => {
                            if (!adHocLocation) return;
                            setShowAiInsights(prev => !prev);
                            if (!nlgSummary && !nlgLoading) {
                              setNlgLoading(true);
                              const data = await nlgSummaryLocation(adHocLocation.lat, adHocLocation.lon, adHocLocation.name);
                              if (data && !('error' in data)) setNlgSummary(data);
                              setNlgLoading(false);
                            }
                          }}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-amber-300 flex items-center gap-2`}>
                              <Sparkles size={14} className="text-amber-400" /> AI Insights
                              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-amber-500/15 text-amber-400/50 ml-2">GloFAS v4 + ERA5</span>
                            </span>
                            <span className="text-[10px] text-gray-600 font-mono mt-0.5">Natural language situation analysis</span>
                          </div>
                          <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showAiInsights ? 'rotate-90' : ''}`} />
                        </button>
                        <AnimatePresence>
                          {showAiInsights && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                              <div className="px-4 pb-4 flex flex-col gap-3">
                                {nlgLoading && (
                                  <div className="flex items-center justify-center py-8">
                                    <div className="w-8 h-8 border-2 border-amber-400/30 border-t-amber-400 rounded-full animate-spin" />
                                  </div>
                                )}
                                {nlgSummary && !nlgLoading && (
                                  <>
                                    <div className="text-[10px] uppercase tracking-widest text-amber-500/50 font-mono mb-1.5">SITUATION</div>
                                    <div className="text-[13px] text-gray-300 leading-relaxed font-sans whitespace-pre-line -mt-1">
                                      {nlgSummary.narrative.split('**').map((part, i) =>
                                        i % 2 === 1 ? <strong key={i} className="text-white">{part}</strong> : <span key={i}>{part}</span>
                                      )}
                                    </div>

                                    <div className="text-[10px] uppercase tracking-widest text-amber-500/50 font-mono mb-1.5 mt-3">KEY FINDINGS</div>
                                    <div className="flex flex-col gap-1.5 -mt-1">
                                      {nlgSummary.highlights.map((h, i) => (
                                        <div key={i} className="flex items-start gap-2 text-[13px] font-sans text-gray-300 leading-relaxed">
                                          <span className="text-amber-500/60 mt-0.5">—</span>
                                          {h}
                                        </div>
                                      ))}
                                    </div>
                                    
                                    {nlgSummary.trend_narrative && (
                                      <>
                                        <div className="text-[10px] uppercase tracking-widest text-amber-500/50 font-mono mb-1.5 mt-3">TREND</div>
                                        <div className="text-[13px] text-gray-300 leading-relaxed font-sans -mt-1">
                                          {nlgSummary.trend_narrative.split('**').map((part, i) =>
                                            i % 2 === 1 ? <strong key={i} className="text-white">{part}</strong> : <span key={i}>{part}</span>
                                          )}
                                        </div>
                                      </>
                                    )}

                                    <div className="flex items-center justify-between text-[9px] text-gray-700 font-mono mt-2 pt-2 border-t border-white/5">
                                      <span>{nlgSummary.engine === 'gpt-4o-mini' ? '✨ GPT-4o' : '⚙ Template Model'}</span>
                                      <span>Generated {new Date(nlgSummary.generated_at).toLocaleTimeString()}</span>
                                    </div>
                                  </>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {/* ── Multi-Source Analysis Panel (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-cyan-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={async () => {
                            if (!adHocLocation) return;
                            setShowFusion(prev => !prev);
                            if (!fusionData && !fusionLoading) {
                              setFusionLoading(true);
                              const data = await fusionLocation(adHocLocation.lat, adHocLocation.lon, adHocLocation.name);
                              if (data && !('error' in data)) setFusionData(data);
                              setFusionLoading(false);
                            }
                          }}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-cyan-300 flex items-center gap-2`}>
                              <Radio size={14} className="text-cyan-400" /> Multi-Source Analysis
                            </span>
                            <span className="text-[10px] text-gray-600 font-mono mt-0.5">Adaptive optical & radar sensor fusion</span>
                          </div>
                          <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFusion ? 'rotate-90' : ''}`} />
                        </button>
                        <AnimatePresence>
                          {showFusion && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                              <div className="px-4 pb-4 flex flex-col gap-2.5">
                                <div className="text-[11px] text-gray-500 font-mono leading-relaxed">
                                  Combines NASA MODIS satellite NDVI, ERA5 thermal reanalysis, Open-Meteo weather, and model-estimated radar data using adaptive weighting for comprehensive flood detection across data sources.
                                </div>
                                {fusionLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" /></div>}
                                {fusionData && !fusionLoading && (
                                  <>
                                    <div className="grid grid-cols-2 gap-2">
                                      {[
                                        { label: 'Flood Conf.', value: fusionData.flood_confidence, color: '#00E5FF', desc: 'Combined multi-sensor flood likelihood' },
                                        { label: 'Veg Stress', value: fusionData.vegetation_stress, color: '#4ade80', desc: 'Vegetation health anomaly from optical' },
                                        { label: 'Soil Sat.', value: fusionData.soil_saturation, color: '#c084fc', desc: 'Ground water saturation from weather' },
                                        { label: 'Water Extent', value: fusionData.surface_water_extent_pct, color: '#38bdf8', desc: 'Surface water coverage estimate' },
                                        { label: 'Quality', value: fusionData.quality_score, color: '#facc15', desc: 'Overall data reliability score' },
                                      ].map(s => (
                                        <div key={s.label} className="bg-[#151A22] rounded-lg p-2 border border-white/5" title={s.desc}>
                                          <div className="text-[10px] text-gray-500 font-mono">{s.label}</div>
                                          <div className="flex items-center gap-2 mt-1">
                                            <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                              <div className="h-full rounded-full" style={{ width: `${(s.value * 100)}%`, backgroundColor: s.color }} />
                                            </div>
                                            <span className="text-[11px] font-mono text-gray-400">{(s.value * 100).toFixed(0)}%</span>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                    {fusionData.fusion_weights && Object.keys(fusionData.fusion_weights).length > 0 && (
                                      <div className="bg-[#151A22] rounded-lg p-2.5 border border-white/5">
                                        <div className="text-[12px] text-gray-500 font-mono mb-1.5">ADAPTIVE FUSION WEIGHTS</div>
                                        <div className="text-[12px] text-gray-600 font-mono mb-1">How much each sensor contributes — auto-adjusted based on data quality and cloud cover</div>
                                        {Object.entries(fusionData.fusion_weights).map(([sensor, weight]) => (
                                          <div key={sensor} className="flex items-center gap-2 mb-1">
                                            <span className="text-[13px] font-mono text-gray-400 min-w-[140px] shrink-0">{sensor}</span>
                                            <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                              <div className="h-full rounded-full bg-cyan-400" style={{ width: `${(Number(weight) * 100)}%` }} />
                                            </div>
                                            <span className="text-[13px] font-mono text-gray-500 w-10 text-right">{(Number(weight) * 100).toFixed(0)}%</span>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                    {fusionData.cloud_penetration_used && (
                                      <div className="text-[11px] font-mono text-cyan-400 flex items-center gap-1.5">
                                        <Radio size={10} /> Cloud-penetration model active — optical data limited, model-based radar estimate used
                                      </div>
                                    )}
                                    <div className="text-[10px] font-mono text-gray-600">
                                      Sensors: {fusionData.sensors_fused?.join(', ')}
                                    </div>
                                    {fusionData.thermal_anomaly !== 0 && (
                                      <div className="text-[11px] font-mono text-gray-500">
                                        Thermal: {fusionData.thermal_anomaly > 0 ? '+' : ''}{fusionData.thermal_anomaly?.toFixed(1)}°C anomaly — {fusionData.thermal_anomaly > 2 ? 'significant heat stress detected' : fusionData.thermal_anomaly < -2 ? 'cooling anomaly detected' : 'within normal range'}
                                      </div>
                                    )}
                                  </>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {/* ── Compound Risk Panel (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-rose-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={async () => {
                            if (!adHocLocation) return;
                            setShowCompound(prev => !prev);
                            if (!compoundData && !compoundLoading) {
                              setCompoundLoading(true);
                              const data = await compoundRiskLocation(adHocLocation.lat, adHocLocation.lon, adHocLocation.name);
                              if (data && !('error' in data)) setCompoundData(data);
                              setCompoundLoading(false);
                            }
                          }}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-rose-300 flex items-center gap-2`}>
                              <Shield size={14} className="text-rose-400" /> Compound Risk
                            </span>
                            <span className="text-[10px] text-gray-600 font-mono mt-0.5">Cascading multi-hazard interactions</span>
                          </div>
                          <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showCompound ? 'rotate-90' : ''}`} />
                        </button>
                        <AnimatePresence>
                          {showCompound && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                              <div className="px-4 pb-4 flex flex-col gap-2.5">
                                {compoundLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-rose-400/30 border-t-rose-400 rounded-full animate-spin" /></div>}
                                {compoundData && !compoundLoading && (
                                  <>
                                    <div className="flex items-center gap-2">
                                      <span className={`text-[20px] font-bold font-mono ${compoundData.compound_level === 'CRITICAL' ? 'text-red-400' :
                                        compoundData.compound_level === 'HIGH' ? 'text-orange-400' :
                                          compoundData.compound_level === 'MEDIUM' ? 'text-yellow-400' : 'text-emerald-400'
                                        }`}>{(compoundData.compound_score * 100).toFixed(0)}%</span>
                                      <span className="text-[12px] font-mono text-gray-500">{compoundData.compound_level}</span>
                                      {compoundData.cascading_amplification > 1.05 && (
                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-red-500/30 text-red-400 bg-red-500/10">
                                          ×{compoundData.cascading_amplification.toFixed(2)} cascade
                                        </span>
                                      )}
                                    </div>
                                    <div className="flex flex-col gap-1">
                                      {compoundData.hazard_layers?.map((h: { name: string; severity: number; status: string; description: string }) => (
                                        <div key={h.name} className="flex items-center gap-2">
                                          <div className={`w-1.5 h-1.5 rounded-full ${h.status === 'active' ? 'bg-red-400' : h.status === 'warning' ? 'bg-yellow-400' : 'bg-gray-600'}`} />
                                          <span className="text-[11px] font-mono text-gray-400 flex-1">{h.name.replace('_', ' ')}</span>
                                          <span className="text-[11px] font-mono text-gray-500">{(h.severity * 100).toFixed(0)}%</span>
                                        </div>
                                      ))}
                                    </div>
                                    {compoundData.interaction_effects?.length > 0 && (
                                      <div className="bg-[#151A22] rounded-lg p-2 border border-white/5">
                                        <div className="text-[10px] text-gray-500 font-mono mb-1">INTERACTIONS</div>
                                        {compoundData.interaction_effects.map((ie: { effect: string; amplification: number }, i: number) => (
                                          <div key={i} className="text-[11px] text-rose-300 font-mono">⚡ {ie.effect}</div>
                                        ))}
                                      </div>
                                    )}
                                    {compoundData.recommendations?.length > 0 && (
                                      <div className="flex flex-col gap-1 mt-1">
                                        {compoundData.recommendations.slice(0, 3).map((r: string, i: number) => (
                                          <div key={i} className="text-[11px] text-gray-400 flex gap-1.5 items-start">
                                            <span className="text-rose-400">→</span> {r}
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                    {compoundData.data_sources && (
                                      <div className="text-[10px] text-gray-600 font-mono mt-1 pt-1 border-t border-white/5">
                                        {compoundData.data_sources.methodology} · {Object.entries(compoundData.data_sources).filter(([k]) => !['methodology', 'flood_probability'].includes(k)).map(([, v]) => v).join(' · ')}
                                      </div>
                                    )}
                                  </>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {/* ── Financial Impact Panel (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-emerald-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={async () => {
                            if (!adHocLocation) return;
                            setShowFinancial(prev => !prev);
                            if (!financialData && !financialLoading) {
                              setFinancialLoading(true);
                              const data = await financialImpactLocation(adHocLocation.lat, adHocLocation.lon, adHocLocation.name);
                              if (data && !('error' in data)) setFinancialData(data);
                              setFinancialLoading(false);
                            }
                          }}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-emerald-300 flex items-center gap-2`}>
                              <DollarSign size={14} className="text-emerald-400" /> Financial Impact
                            </span>
                            <span className="text-[10px] text-gray-600 font-mono mt-0.5">Estimated economic exposure & ROI</span>
                          </div>
                          <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFinancial ? 'rotate-90' : ''}`} />
                        </button>
                        <AnimatePresence>
                          {showFinancial && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                              <div className="px-4 pb-4 flex flex-col gap-2.5">
                                {financialLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" /></div>}
                                {financialData && !financialLoading && (
                                  <>
                                    {financialData.scenario_based && (
                                      <div className="text-[10px] font-mono text-amber-400/80 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1 text-center">
                                        Scenario-based estimate (no active flood detected)
                                      </div>
                                    )}
                                    <div className="text-center">
                                      <div className="text-[10px] text-gray-500 font-mono">{financialData.scenario_based ? 'POTENTIAL EXPOSURE' : 'TOTAL EXPOSURE'}</div>
                                      <div className="text-[22px] font-bold font-mono text-emerald-400">
                                        {fmtUSD(financialData.total_impact_usd ?? 0)}
                                      </div>
                                      {financialData.confidence && (
                                        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${financialData.confidence === 'high' ? 'border-emerald-500/30 text-emerald-400' : financialData.confidence === 'medium' ? 'border-yellow-500/30 text-yellow-400' : 'border-gray-500/30 text-gray-400'}`}>
                                          {financialData.confidence.toUpperCase()} CONFIDENCE
                                        </span>
                                      )}
                                    </div>
                                    <div className="grid grid-cols-3 gap-1.5">
                                      {[
                                        { label: 'Direct', value: financialData.direct_damage_usd },
                                        { label: 'Indirect', value: financialData.indirect_costs_usd },
                                        { label: 'Recovery', value: financialData.recovery_cost_usd },
                                      ].map(m => (
                                        <div key={m.label} className="bg-[#151A22] rounded-lg p-2 text-center border border-white/5">
                                          <div className="text-[10px] text-gray-500 font-mono">{m.label}</div>
                                          <div className="text-[12px] font-mono text-gray-300">{fmtUSD(m.value ?? 0)}</div>
                                        </div>
                                      ))}
                                    </div>
                                    <div className="flex items-center gap-3 text-[11px] font-mono text-gray-500">
                                      <span>🏥 {financialData.affected_population?.toLocaleString()} affected</span>
                                      <span>📊 GDP: {financialData.gdp_impact_pct?.toFixed(3)}%</span>
                                    </div>
                                    {financialData.mitigation_roi?.length > 0 && (
                                      <div className="bg-[#151A22] rounded-lg p-2.5 border border-white/5">
                                        <div className="text-[10px] text-gray-500 font-mono mb-1.5">MITIGATION ROI</div>
                                        {financialData.mitigation_roi.slice(0, 3).map((m: { measure: string; roi_pct: number }, i: number) => (
                                          <div key={i} className="flex items-center justify-between text-[11px] font-mono">
                                            <span className="text-gray-400">{m.measure}</span>
                                            <span className={m.roi_pct > 200 ? 'text-emerald-400' : 'text-yellow-400'}>{m.roi_pct}% ROI</span>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {/* ── Model Feedback Panel (Ad-Hoc) ── */}
                      <div className="bg-[#0A1628]/80 border border-gray-500/20 rounded-xl overflow-hidden shrink-0">
                        <button
                          onClick={() => setShowFeedback(prev => !prev)}
                          className="w-full p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span className={`text-[12px] font-medium uppercase ${textMono} tracking-widest text-gray-400 flex items-center gap-2`}>
                              <ThumbsUp size={14} className="text-gray-500" /> Model Feedback
                            </span>
                            <span className="text-[10px] text-gray-600 font-mono mt-0.5">Help train the prediction engine</span>
                          </div>
                          <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFeedback ? 'rotate-90' : ''}`} />
                        </button>
                        <AnimatePresence>
                          {showFeedback && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                              <div className="px-4 pb-4 flex flex-col gap-3">
                                <div className="text-[12px] text-gray-500 font-mono">
                                  Was the flood detection for {adHocLocation.name} accurate?
                                </div>
                                <div className="flex gap-2">
                                  <button
                                    onClick={async () => {
                                      await submitFeedback({
                                        detection_type: 'flood',
                                        model_prediction: pred?.predicted_risk_level || det?.detected_risk_level || 'UNKNOWN',
                                        user_verdict: 'correct',
                                        location: { lat: adHocLocation.lat, lon: adHocLocation.lon, name: adHocLocation.name },
                                      });
                                      setShowFeedback(false);
                                    }}
                                    className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-[12px] font-mono text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                                  >
                                    <ThumbsUp size={12} /> Correct
                                  </button>
                                  <button
                                    onClick={async () => {
                                      await submitFeedback({
                                        detection_type: 'flood',
                                        model_prediction: pred?.predicted_risk_level || det?.detected_risk_level || 'UNKNOWN',
                                        user_verdict: 'incorrect',
                                        location: { lat: adHocLocation.lat, lon: adHocLocation.lon, name: adHocLocation.name },
                                      });
                                      setShowFeedback(false);
                                    }}
                                    className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-[12px] font-mono text-red-400 hover:bg-red-500/20 transition-colors"
                                  >
                                    <ThumbsDown size={12} /> Incorrect
                                  </button>
                                </div>
                                <div className="text-[10px] text-gray-600 font-mono">
                                  Feedback improves model accuracy through continuous learning
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </>
                  );
                })()}

              </div>

              {/* Download Action */}
              <div className="p-4 border-t border-white/10 bg-black/40 backdrop-blur-md sticky bottom-0 z-20 flex gap-2 w-full">
                <button
                  onClick={() => {
                    const url = `${window.location.origin}/ad-hoc?lat=${adHocLocation.lat}&lon=${adHocLocation.lon}`;
                    navigator.clipboard.writeText(url);
                  }}
                  className="w-10 h-10 shrink-0 flex items-center justify-center bg-[#151A22] hover:bg-white/10 text-gray-400 hover:text-white border border-white/10 rounded-lg transition-colors group"
                  title="Copy link to this location"
                >
                  <Link size={14} className="group-hover:scale-110 transition-transform" />
                </button>
                <button
                  onClick={() => {
                    const reportUrl = `/api/reports/executive/0?lat=${adHocLocation.lat}&lon=${adHocLocation.lon}&name=${encodeURIComponent(adHocLocation.name)}`;
                    window.open(reportUrl, '_blank');
                  }}
                  className="flex-1 flex items-center justify-center gap-2 h-10 bg-[#151A22] hover:bg-white/10 text-white font-mono text-[12px] uppercase tracking-widest border border-white/10 hover:border-[#00E5FF]/40 rounded-lg transition-colors group"
                >
                  <Download size={14} className="text-gray-400 group-hover:text-[#00E5FF]" /> PDF Report
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>


      {/* ═══ D. TIME-SERIES (BOTTOM CENTER) ═══ */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`absolute left-[390px] right-[430px] bottom-6 h-[180px] z-20 ${glassClass} flex flex-col overflow-hidden`}>
        <div className="px-5 py-3 border-b border-white/10 flex justify-between items-center bg-black/40">
          <div className="flex flex-col">
            <h3 className={`text-[12px] uppercase text-gray-400 ${textMono} tracking-widest`}>
              {selectedRegion ? `${selectedRegion.name} — ${currentOrb.panelTitle}` : adHocLocation ? `${adHocLocation.name} — ${currentOrb.panelTitle}` : 'Time-Series & Change Detection'}
            </h3>
            <span className={`text-[10px] text-gray-600 ${textMono}`}>
              {adHocLocation
                ? 'Open-Meteo ERA5 reanalysis — real historical climate data for this location'
                : activeOrb === 'flood' ? 'Satellite-derived flood coverage trend from historical risk assessments'
                : activeOrb === 'infra' ? 'Infrastructure exposure trend based on water change analysis'
                : 'Vegetation anomaly index over time from NDVI analysis'}
            </span>
          </div>
        </div>
        <div className="flex-grow flex flex-col relative p-4 min-h-0">
          <div className="flex-grow w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={displayChartData.length > 0 ? displayChartData : [{ date: '--', flood: 0, confidence: 0, water_change: 0 }]}>
                <defs>
                  <linearGradient id="chartColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={primaryColor} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="chartColor2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} axisLine={false} tickLine={false} />
                <YAxis hide />
                <RechartsTooltip contentStyle={{ background: '#0B0E11', border: '1px solid rgba(255,255,255,0.1)', fontFamily: 'monospace', fontSize: '13px' }} itemStyle={{ color: primaryColor, fontWeight: 'bold' }} />
                <Area type="monotone" dataKey={chartKey} stroke={primaryColor} fill="url(#chartColor)" strokeWidth={2} name={currentOrb.chartLabel} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="h-8 mt-1 flex items-center gap-4 px-2">
            <button onClick={() => setPlaying(!playing)} className="w-7 h-7 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 flex items-center justify-center text-white transition-colors shrink-0"
              style={{ color: playing ? primaryColor : undefined }}>
              {playing ? <Pause size={10} /> : <Play size={10} className="ml-0.5" />}
            </button>
            <div className="flex-grow h-1 bg-[#151A22] border border-white/10 rounded-full relative cursor-pointer group">
              <div className="absolute top-0 left-0 bottom-0 rounded-full w-3/4" style={{ backgroundColor: primaryColor }} />
              <div className="absolute top-1/2 -mt-1 -ml-1 left-3/4 w-2 h-2 bg-white border-2 rounded-full" style={{ borderColor: primaryColor, boxShadow: `0 0 8px ${primaryColor}` }} />
            </div>
            <span className={`text-[11px] text-gray-400 shrink-0 ${textMono}`}>2024 — 2026</span>
          </div>
        </div>
      </motion.div>

      {/* ═══ E. PROCESSING LOGS TERMINAL (BOTTOM LEFT) ═══ */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`absolute left-6 bottom-6 w-[340px] z-20 ${glassClass} flex flex-col overflow-hidden`} style={{ height: showLogs ? '160px' : '32px' }}>
        <div className="h-8 bg-black/60 border-b border-white/10 flex items-center justify-between px-3 cursor-pointer shrink-0" onClick={() => setShowLogs(!showLogs)}>
          <div className="flex items-center gap-2">
            <Terminal size={12} className="text-gray-400" />
            <span className={`text-[11px] text-gray-400 ${textMono} uppercase tracking-widest`}>Processing Terminal</span>
          </div>
          <span className={`text-[11px] text-gray-500 ${textMono}`}>{logs.length} entries</span>
        </div>
        {showLogs && (
          <div ref={logRef} className={`flex-grow p-3 flex flex-col gap-0.5 overflow-y-auto ${textMono} text-[10px] text-[#20E251] bg-black/20`}>
            {logs.map((l, i) => (
              <div key={l.id || i} className="whitespace-nowrap opacity-80 flex gap-2">
                <span className="text-gray-600">[{l.step}]</span>
                <span className={l.status === 'completed' ? 'text-[#20E251]' : 'text-yellow-400'}>{l.details?.message || l.status}</span>
                {l.duration_ms && <span className="text-gray-600 ml-auto">{l.duration_ms}ms</span>}
              </div>
            ))}
            <div className="animate-pulse text-[#20E251]">_</div>
          </div>
        )}
      </motion.div>

      {/* ═══ LOGIN MODAL ═══ */}
      <AnimatePresence>
        {showLogin && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] bg-black/70 backdrop-blur-sm flex items-center justify-center"
            onClick={() => setShowLogin(false)}>
            <motion.div initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="w-[380px] bg-[#0D1117] border border-white/10 rounded-2xl p-8 shadow-2xl flex flex-col gap-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-mono font-bold text-white">COSMEON Login</h2>
                  <p className="text-[12px] text-gray-500 font-mono mt-0.5">Authenticate to access protected features</p>
                </div>
                <button onClick={() => setShowLogin(false)} className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center hover:bg-white/10 transition-colors"><X size={14} className="text-gray-400" /></button>
              </div>
              <div className="flex flex-col gap-3">
                <div className="flex flex-col gap-1.5">
                  <label className="text-[11px] font-mono text-gray-500 uppercase tracking-wider">Username</label>
                  <input
                    type="text" value={loginUsername} onChange={e => setLoginUsername(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleLogin()}
                    placeholder="admin"
                    className="w-full px-4 py-3 bg-[#151A22] border border-white/10 rounded-lg text-sm font-mono text-white placeholder-gray-600 focus:outline-none focus:border-cyan-500/50 transition-colors"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-[11px] font-mono text-gray-500 uppercase tracking-wider">Password</label>
                  <input
                    type="password" value={loginPassword} onChange={e => setLoginPassword(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleLogin()}
                    placeholder="••••••••"
                    className="w-full px-4 py-3 bg-[#151A22] border border-white/10 rounded-lg text-sm font-mono text-white placeholder-gray-600 focus:outline-none focus:border-cyan-500/50 transition-colors"
                  />
                </div>
                {loginError && <p className="text-[12px] font-mono text-red-400">{loginError}</p>}
              </div>
              <button onClick={handleLogin} disabled={loginLoading || !loginUsername || !loginPassword}
                className="w-full py-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-mono text-sm font-bold text-black">
                {loginLoading ? 'Authenticating...' : 'Login'}
              </button>
              <div className="text-[11px] text-gray-600 font-mono text-center">
                Default: <span className="text-gray-400">admin / admin123</span> · <span className="text-gray-400">analyst / analyst123</span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ TREND DASHBOARD MODAL ═══ */}
      <AnimatePresence>
        {showTrends && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] bg-black/80 backdrop-blur-md flex items-center justify-center p-6"
            onClick={() => setShowTrends(false)}>
            <motion.div initial={{ scale: 0.95, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="w-full max-w-5xl max-h-[90vh] bg-[#0D1117] border border-white/10 rounded-2xl flex flex-col overflow-hidden shadow-2xl">
              {/* Header */}
              <div className="px-8 py-5 border-b border-white/5 flex items-center justify-between bg-[#0B0E11]">
                <div>
                  <h2 className="text-lg font-mono font-bold text-white flex items-center gap-2">
                    <TrendingUp size={18} className="text-cyan-400" />
                    Historical Risk Trend — {trendData?.region_name || selectedRegion?.name || adHocLocation?.name}
                  </h2>
                  <p className="text-[12px] font-mono text-gray-500 mt-0.5">
                    {trendData ? `${trendData.data_points} monthly snapshots · 24-month view` : 'Loading trend data...'}
                  </p>
                </div>
                <button onClick={() => setShowTrends(false)} className="w-9 h-9 rounded-full bg-white/5 flex items-center justify-center hover:bg-white/10 transition-colors">
                  <X size={16} className="text-gray-400" />
                </button>
              </div>
              {/* Content */}
              <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-8">
                {trendLoading ? (
                  <div className="flex-1 flex items-center justify-center gap-3">
                    <div className="w-6 h-6 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                    <span className="font-mono text-gray-400 text-sm">Loading trend data...</span>
                  </div>
                ) : trendData && trendData.trend.length > 0 ? (
                  <>
                    {/* Summary row */}
                    <div className="grid grid-cols-4 gap-4">
                      {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map(level => {
                        const count = trendData.trend.reduce((s, m) => s + (m.risk_distribution[level as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'] || 0), 0);
                        const color = level === 'CRITICAL' ? '#ef4444' : level === 'HIGH' ? '#f97316' : level === 'MEDIUM' ? '#eab308' : '#22c55e';
                        return (
                          <div key={level} className="bg-[#151A22] border border-white/5 rounded-xl p-4 flex flex-col gap-1">
                            <span className="text-[10px] font-mono uppercase tracking-widest text-gray-500">{level}</span>
                            <span className="text-2xl font-mono font-bold" style={{ color }}>{count}</span>
                            <span className="text-[11px] text-gray-600 font-mono">months</span>
                          </div>
                        );
                      })}
                    </div>

                    {/* Flood % over time */}
                    <div className="bg-[#151A22] border border-white/5 rounded-xl p-6">
                      <h3 className="text-[12px] font-mono uppercase tracking-widest text-gray-400 mb-2">{currentOrb.chartLabel} % — Monthly Average vs Peak</h3>
                      <div className="flex items-center gap-4 mb-2">
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-0.5 rounded" style={{ backgroundColor: primaryColor }} />
                          <span className="text-[10px] font-mono text-gray-500">Monthly Avg (ERA5 percentile)</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-0.5 rounded bg-orange-500" style={{ borderTop: '1px dashed #f97316' }} />
                          <span className="text-[10px] font-mono text-gray-500">Monthly Peak</span>
                        </div>
                      </div>
                      <div className="h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={trendData.trend} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                            <defs>
                              <linearGradient id="trendGradAvg" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={primaryColor} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
                              </linearGradient>
                              <linearGradient id="trendGradPeak" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f97316" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="month_label" tick={{ fill: '#4b5563', fontSize: 10, fontFamily: 'monospace' }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: '#4b5563', fontSize: 10, fontFamily: 'monospace' }} tickLine={false} axisLine={false} />
                            <RechartsTooltip contentStyle={{ backgroundColor: '#0D1117', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontFamily: 'monospace', fontSize: 12, color: '#e2e8f0' }} formatter={(v, name) => [`${Number(v).toFixed(1)}%`, name === 'Monthly Average' ? 'Monthly Avg' : 'Monthly Peak']} />
                            <Area type="monotone" dataKey={currentOrb.id === 'flood' ? 'avg_flood_pct' : currentOrb.id === 'infra' ? 'avg_water_change_pct' : 'avg_vegetation_stress'} stroke={primaryColor} fill="url(#trendGradAvg)" strokeWidth={2} name="Monthly Average" dot={false} />
                            <Area type="monotone" dataKey={currentOrb.id === 'flood' ? 'max_flood_pct' : currentOrb.id === 'infra' ? 'max_water_change_pct' : 'max_vegetation_stress'} stroke="#f97316" fill="url(#trendGradPeak)" strokeWidth={1.5} strokeDasharray="4 2" name="Monthly Peak" dot={false} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Monthly Precipitation + Heavy Rain Days side by side */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[#151A22] border border-white/5 rounded-xl p-5">
                        <h3 className="text-[12px] font-mono uppercase tracking-widest text-gray-400 mb-4">Monthly Precipitation mm</h3>
                        <div className="h-[160px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={trendData.trend} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.04)" />
                              <XAxis dataKey="month_label" tick={{ fill: '#4b5563', fontSize: 9, fontFamily: 'monospace' }} tickLine={false} axisLine={false} />
                              <YAxis tick={{ fill: '#4b5563', fontSize: 9, fontFamily: 'monospace' }} tickLine={false} axisLine={false} />
                              <RechartsTooltip contentStyle={{ backgroundColor: '#0D1117', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontFamily: 'monospace', fontSize: 11 }} formatter={(v) => [`${Number(v).toFixed(1)} mm`]} />
                              <Bar dataKey="total_precip_mm" fill={primaryColor} fillOpacity={0.8} radius={[2, 2, 0, 0]} name="Precipitation mm" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      <div className="bg-[#151A22] border border-white/5 rounded-xl p-5">
                        <h3 className="text-[12px] font-mono uppercase tracking-widest text-gray-400 mb-1">Heavy Rain Days (&gt;20mm/day)</h3>
                        <p className="text-[9px] font-mono text-gray-600 mb-3">Days where single-day ERA5 precipitation exceeded 20mm</p>
                        <div className="h-[160px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={trendData.trend} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.04)" />
                              <XAxis dataKey="month_label" tick={{ fill: '#4b5563', fontSize: 9, fontFamily: 'monospace' }} tickLine={false} axisLine={false} />
                              <YAxis tick={{ fill: '#4b5563', fontSize: 9, fontFamily: 'monospace' }} tickLine={false} axisLine={false} allowDecimals={false} />
                              <RechartsTooltip contentStyle={{ backgroundColor: '#0D1117', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontFamily: 'monospace', fontSize: 11 }} formatter={(v) => [`${Number(v)} days`]} />
                              <Bar dataKey="heavy_rain_days" fill="#8b5cf6" fillOpacity={0.8} radius={[2, 2, 0, 0]} name="Heavy Rain Days" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>

                    {/* Monthly table */}
                    <div className="bg-[#151A22] border border-white/5 rounded-xl overflow-hidden">
                      <table className="w-full text-[12px] font-mono">
                        <thead>
                          <tr className="border-b border-white/5 bg-black/20">
                            {['Month', 'Dominant Risk', `Avg ${currentOrb.chartLabel}`, `Peak ${currentOrb.chartLabel}`, 'Precip mm', 'Heavy Rain Days', 'Assessments'].map(h => (
                              <th key={h} className="px-4 py-2.5 text-left text-[10px] uppercase tracking-widest text-gray-600">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[...trendData.trend].reverse().map((m, i) => {
                            const rc = m.dominant_risk_level === 'CRITICAL' ? '#ef4444' : m.dominant_risk_level === 'HIGH' ? '#f97316' : m.dominant_risk_level === 'MEDIUM' ? '#eab308' : '#22c55e';
                            const avgValue = currentOrb.id === 'flood' ? m.avg_flood_pct : currentOrb.id === 'infra' ? (m.avg_water_change_pct || m.avg_flood_pct) : (m.avg_vegetation_stress || m.avg_flood_pct);
                            const maxValue = currentOrb.id === 'flood' ? m.max_flood_pct : currentOrb.id === 'infra' ? (m.max_water_change_pct || m.max_flood_pct) : (m.max_vegetation_stress || m.max_flood_pct);
                            return (
                              <tr key={m.month} className={`border-b border-white/5 ${i % 2 === 0 ? 'bg-black/10' : ''} hover:bg-white/3 transition-colors`}>
                                <td className="px-4 py-2.5 text-gray-300">{m.month_label}</td>
                                <td className="px-4 py-2.5"><span className="font-bold" style={{ color: rc }}>{m.dominant_risk_level}</span></td>
                                <td className="px-4 py-2.5 text-gray-400">{avgValue.toFixed(1)}%</td>
                                <td className="px-4 py-2.5 text-gray-400">{maxValue.toFixed(1)}%</td>
                                <td className="px-4 py-2.5 text-gray-400">{m.total_precip_mm.toFixed(1)}</td>
                                <td className="px-4 py-2.5 text-gray-400">{m.heavy_rain_days}</td>
                                <td className="px-4 py-2.5 text-gray-500">{m.assessment_count}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-gray-500 font-mono text-sm">No trend data available for this region.</div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </main>
  );
}
