"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Map, { Marker, Popup, MapRef } from "react-map-gl/maplibre";
import 'maplibre-gl/dist/maplibre-gl.css';
import { motion, AnimatePresence } from "framer-motion";
import { Search, Satellite, Database, Activity, Layers, Download, SlidersHorizontal, ChevronDown, Terminal, Play, Pause, MapPin, X, AlertTriangle, Leaf, Building2, Sparkles, TrendingUp, ChevronRight, Shield, DollarSign, Radio, ThumbsUp, ThumbsDown } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip } from "recharts";

import { fetchRegions, fetchRegionRisk, fetchRegionHistory, fetchChanges, fetchLogs, getReportDownloadUrl, fetchPrediction, fetchExternalFactors, fetchValidation, fetchDetection, triggerAnalysis, fetchExplanation, analyzeLocation, explainLocation, geocodeSearch, reverseGeocode, GeoResult, fetchForecast, fetchNLGSummary, ForecastData, NLGSummary, fetchFusionAnalysis, fetchCompoundRisk, fetchFinancialImpact, submitFeedback } from "@/lib/api";

// ─── Types ───
interface Region {
  id: number;
  name: string;
  bbox: number[]; // [west, south, east, north]
}

interface RiskAssessment {
  risk_level: string;
  flood_area_km2: number;
  total_area_km2: number;
  flood_percentage: number;
  confidence_score: number;
  change_type: string;
  water_change_pct: number;
  timestamp: string;
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
  glofas_assessment: {
    risk_level: string;
    discharge_m3s: number;
    anomaly_sigma: number;
    mean_discharge_m3s: number;
    explanation: string;
  };
  comparison: {
    agreement: boolean;
    agreement_score: number;
    summary: string;
    difference_reasons: string[];
    our_methodology: string;
    glofas_methodology: string;
  };
  independence_proof: {
    model_uses: string;
    model_does_not_use: string;
    glofas_uses: string;
    how_training_works: string;
    verification: string;
    feature_data_sources: Record<string, string[]>;
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

// ─── Helper ───
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
  const [liveDetection, setLiveDetection] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [explanation, setExplanation] = useState<ExplainData | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [showIndependence, setShowIndependence] = useState(false);
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
  const [mapClickPopup, setMapClickPopup] = useState<{ lat: number; lon: number; name?: string } | null>(null);

  // ── UI State ──
  const [searchQuery, setSearchQuery] = useState("");
  const [searchFocused, setSearchFocused] = useState(false);
  const [geoResults, setGeoResults] = useState<GeoResult[]>([]);
  const [geoLoading, setGeoLoading] = useState(false);
  const geoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [activeSource, setActiveSource] = useState("Sentinel-2");
  const [activeOrb, setActiveOrb] = useState<OrbKey>("flood");
  const [hoverInfo, setHoverInfo] = useState<Region | null>(null);
  const [playing, setPlaying] = useState(false);
  const [showLogs, setShowLogs] = useState(true);
  const [comparisonMode, setComparisonMode] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapRef>(null);



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
    if (!q || q.length < 2 || parsedCoords) {
      setGeoResults([]);
      return;
    }
    // Don't geocode if it matches an existing region exactly
    const matchesRegion = regions.some(r => r.name.toLowerCase().includes(q.toLowerCase()));
    if (matchesRegion && q.length < 4) {
      setGeoResults([]);
      return;
    }
    // Debounce: wait 350ms after last keystroke
    if (geoTimerRef.current) clearTimeout(geoTimerRef.current);
    geoTimerRef.current = setTimeout(async () => {
      setGeoLoading(true);
      const results = await geocodeSearch(q);
      setGeoResults(results);
      setGeoLoading(false);
    }, 350);
    return () => { if (geoTimerRef.current) clearTimeout(geoTimerRef.current); };
  }, [searchQuery, parsedCoords, regions]);

  // ── Select a region & fly to it ──
  const selectRegion = useCallback((reg: Region | null) => {
    setSelectedRegion(reg);
    setAdHocLocation(null); // clear ad-hoc when selecting a real region
    setAdHocData(null);
    setAdHocExplanation(null);
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
    setMapClickPopup(null);

    if (mapRef.current) {
      mapRef.current.flyTo({ center: [lon, lat], zoom: 6, duration: 1500 });
    }

    const data = await analyzeLocation(lat, lon, locName);
    if (data) setAdHocData(data);
    setAdHocLoading(false);
  }, []);

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

    // Fetch latest live detection result
    fetchDetection(region.id).then(detData => {
      if (detData && detData.detection) setLiveDetection(detData.detection);
      else setLiveDetection(null);
    });

    // Auto-trigger a fresh live analysis so timestamps are always current
    if (autoAnalyze) {
      triggerAnalysis(region.id).then(async (result) => {
        if (result?.detection) setLiveDetection(result.detection);
        // Re-fetch risk to update "Last Analyzed" timestamp
        const freshRisk = await fetchRegionRisk(region.id);
        if (freshRisk && !freshRisk.message) setLatestRisk(freshRisk);
        // Re-fetch prediction with fresh data
        const freshPred = await fetchPrediction(region.id);
        if (freshPred && freshPred.predicted_risk_level) setPrediction(freshPred);
      });
    }
  }, []);

  // ── Load Region-Specific Data when selection changes ──
  useEffect(() => {
    if (!selectedRegion) return;
    setExplanation(null);
    setShowIndependence(false);
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

  // ── Derived data for charts ──
  const chartData = useMemo(() =>
    riskHistory
      .slice()
      .reverse()
      .map(a => ({
        date: a.timestamp ? new Date(a.timestamp).toLocaleDateString("en-US", { month: "short", year: "2-digit" }) : "",
        flood: Math.round(a.flood_percentage * 100 * 100) / 100,
        confidence: Math.round(a.confidence_score * 100),
        water_change: Math.round(a.water_change_pct * 100 * 100) / 100,
      }))
    , [riskHistory]);

  // ── Choose chart data key based on active orb ──
  const chartKey = activeOrb === "flood" ? "flood" : activeOrb === "infra" ? "water_change" : "confidence";
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
            return (
              <Marker key={reg.id} longitude={lon} latitude={lat} anchor="center">
                <div
                  className="w-6 h-6 rounded-full border-2 cursor-pointer transition-all duration-300 relative flex items-center justify-center"
                  style={{
                    backgroundColor: isSelected ? primaryColor : 'rgba(11,14,17,0.8)',
                    borderColor: isSelected ? primaryColor : 'rgba(255,255,255,0.3)',
                    boxShadow: isSelected ? `0 0 15px ${primaryColor}` : 'none',
                  }}
                  onClick={() => selectRegion(reg)}
                  onMouseEnter={() => setHoverInfo(reg)}
                  onMouseLeave={() => setHoverInfo(null)}
                >
                  {isSelected && (
                    <div className="absolute -inset-2 rounded-full animate-ping opacity-30"
                      style={{ border: `1px solid ${primaryColor}` }} />
                  )}
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
            <div className={`h-12 flex items-center gap-3 px-2 py-1 ${glassClass} rounded-full`}>
              <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center ml-1"><Search size={14} className="text-gray-400" /></div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setSearchFocused(true)}
                onBlur={() => setTimeout(() => setSearchFocused(false), 200)}
                placeholder="Search any place on Earth... e.g., Tokyo, Mumbai, 26.0 85.5"
                className={`bg-transparent outline-none border-none text-sm w-96 text-white placeholder:text-gray-500 ${textMono}`}
              />
              {searchQuery && (
                <button onClick={() => setSearchQuery("")} className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center mr-1 hover:bg-white/20 transition-colors">
                  <X size={10} className="text-gray-400" />
                </button>
              )}
            </div>
            {/* Search Dropdown */}
            <AnimatePresence>
              {searchQuery && searchFocused && (filteredRegions.length > 0 || parsedCoords || geoResults.length > 0 || geoLoading) && (
                <motion.div
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className={`absolute top-14 left-0 right-0 ${glassClass} !rounded-xl py-2 max-h-[340px] overflow-y-auto z-50`}
                >
                  {/* Coordinate analysis option */}
                  {parsedCoords && (
                    <div
                      onMouseDown={() => { analyzeAdHocLocation(parsedCoords.lat, parsedCoords.lon); setSearchQuery(""); }}
                      className="px-4 py-3 flex items-center gap-3 cursor-pointer hover:bg-cyan-500/10 transition-colors border-b border-white/5"
                    >
                      <div className="w-7 h-7 rounded-full bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
                        <MapPin size={14} className="text-cyan-400" />
                      </div>
                      <div className="flex flex-col">
                        <span className={`text-sm ${textMono} text-cyan-400`}>Analyze {parsedCoords.lat.toFixed(2)}, {parsedCoords.lon.toFixed(2)}</span>
                        <span className={`text-[11px] ${textMono} text-gray-500`}>Direct coordinate analysis</span>
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
                          onMouseDown={() => { selectRegion(reg); setSearchQuery(""); }}
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
                      <div className="w-4 h-4 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                      <span className={`text-[13px] ${textMono} text-gray-500`}>Searching places...</span>
                    </div>
                  )}
                  {geoResults.length > 0 && (
                    <>
                      <div className={`px-4 py-1.5 text-[11px] text-gray-600 uppercase ${textMono} tracking-widest ${filteredRegions.length > 0 ? 'border-t border-white/5 mt-1' : ''}`}>Places Worldwide</div>
                      {geoResults.map((geo, i) => {
                        const parts = geo.display_name.split(', ');
                        const shortName = parts.slice(0, 3).join(', ');
                        return (
                          <div
                            key={`geo-${i}`}
                            onMouseDown={() => { analyzeAdHocLocation(geo.lat, geo.lon, shortName); setSearchQuery(""); setGeoResults([]); }}
                            className="px-4 py-2.5 flex items-center gap-3 cursor-pointer hover:bg-cyan-500/10 transition-colors"
                          >
                            <div className="w-6 h-6 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center shrink-0">
                              <MapPin size={12} className="text-cyan-400" />
                            </div>
                            <div className="flex flex-col min-w-0">
                              <span className={`text-sm ${textMono} text-gray-200 truncate`}>{shortName}</span>
                              <span className={`text-[11px] ${textMono} text-gray-600 truncate`}>{geo.lat.toFixed(3)}, {geo.lon.toFixed(3)} — {geo.type}</span>
                            </div>
                          </div>
                        );
                      })}
                    </>
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
          <button className="w-10 h-10 rounded-full bg-gradient-to-br from-[#0B0E11] to-[#151A22] border border-white/20 flex items-center justify-center text-xs font-bold font-mono hover:border-white/40 transition-colors">PA</button>
        </div>
      </motion.div>

      {/* ═══ B. INTELLIGENCE HUB (LEFT) ═══ */}
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="absolute left-6 top-28 w-[340px] z-20 flex flex-col gap-4" style={{ bottom: showLogs ? '200px' : '70px' }}>

        <div className={`${glassClass} p-5 flex flex-col gap-4`}>
          <h2 className={`text-[12px] uppercase text-gray-400 ${textMono} tracking-widest`}>Data Source</h2>
          <div className="flex bg-[#151A22] rounded-lg p-1 border border-white/10 text-xs font-mono">
            {['Sentinel-1', 'Sentinel-2', 'Landsat'].map(s => (
              <button key={s} onClick={() => setActiveSource(s)} className={`flex-1 py-1.5 rounded-md transition-all ${activeSource === s ? 'bg-white/10 text-white shadow-md' : 'text-gray-500 hover:text-white/80'}`}>{s}</button>
            ))}
          </div>
        </div>

        <div className={`${glassClass} p-5 flex-grow flex flex-col overflow-y-auto`}>
          <h2 className={`text-[12px] uppercase text-gray-400 ${textMono} tracking-widest mb-4 flex items-center justify-between`}>
            Active Region (AOI)
            <ChevronDown size={12} className="text-gray-500" />
          </h2>

          {/* Region List — filtered by search */}
          <div className="flex flex-col gap-1.5 mb-6">
            {filteredRegions.length === 0 && (
              <div className="text-xs text-gray-500 font-mono p-3">No regions match &quot;{searchQuery}&quot;</div>
            )}
            {filteredRegions.map(reg => (
              <div key={reg.id} onClick={() => selectRegion(reg)} className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center gap-3
                ${selectedRegion?.id === reg.id ? 'bg-white/5' : 'border-white/5 hover:bg-white/5'}
              `}
                style={{
                  borderColor: selectedRegion?.id === reg.id ? primaryColor + '33' : undefined,
                }}
              >
                <MapPin size={14} style={{ color: selectedRegion?.id === reg.id ? primaryColor : '#6b7280' }} />
                <span className={`text-sm ${textMono} ${selectedRegion?.id === reg.id ? 'text-white' : 'text-gray-400'}`}>{reg.name}</span>
              </div>
            ))}
          </div>

          <h2 className={`text-[12px] uppercase text-gray-400 ${textMono} tracking-widest mb-4`}>Analysis Orbs</h2>
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
                    <span className={`text-xs uppercase ${textMono} ${isActive ? 'text-white' : 'text-gray-400'}`}>{orb.label}</span>
                    {isActive && <div className="text-[10px] font-mono animate-pulse" style={{ color: orb.color }}>Active</div>}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </motion.div>

      {/* ═══ C. STRUCTURED INSIGHTS OUTPUT (RIGHT) ═══ */}
      <AnimatePresence>
        {selectedRegion && latestRisk && (
          <motion.div key={`insights-${selectedRegion.id}-${activeOrb}`} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }} className="absolute right-6 top-28 bottom-32 w-[400px] z-20 flex flex-col gap-4 overflow-hidden">
            <div className={`${glassClass} flex flex-col h-full overflow-hidden`}>

              {/* Header */}
              <div className="p-6 border-b border-white/10 bg-black/40 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col gap-1">
                    <span className={`text-sm uppercase tracking-widest ${textMono} font-bold flex items-center gap-2`} style={{ color: riskColor(latestRisk.risk_level) }}>
                      <AlertTriangle size={16} />
                      {latestRisk.risk_level} RISK
                      <span className="px-2 py-0.5 rounded text-[13px] border" style={{ borderColor: riskColor(latestRisk.risk_level) + '40', backgroundColor: riskColor(latestRisk.risk_level) + '20' }}>
                        {latestRisk.change_type}
                      </span>
                    </span>
                    <span className="text-white text-[15px] font-semibold mt-1">{selectedRegion.name} — {currentOrb.panelTitle}</span>
                  </div>
                  <button onClick={() => setSelectedRegion(null)} className="text-gray-500 hover:text-white"><X size={18} /></button>
                </div>

              </div>

              {/* Data Grid */}
              <div className="flex-grow p-6 flex flex-col gap-6 overflow-y-auto">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-1">
                    <span className={`text-[13px] text-gray-500 uppercase ${textMono}`}>{currentOrb.areaLabel}</span>
                    <span className={`text-xl font-bold ${textMono} text-white`}>{latestRisk.flood_area_km2.toFixed(1)} <span className="text-sm text-gray-500">km²</span></span>
                  </div>
                  <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-1">
                    <span className={`text-[13px] text-gray-500 uppercase ${textMono}`}>Confidence</span>
                    <span className={`text-xl font-bold ${textMono}`} style={{ color: primaryColor }}>{(latestRisk.confidence_score * 100).toFixed(0)}%</span>
                  </div>
                </div>

                <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-3">
                  <span className={`text-[13px] text-gray-500 uppercase ${textMono}`}>Assessment Details</span>
                  <div className="flex justify-between items-center text-[14px] font-mono pb-2 border-b border-white/5">
                    <span className="text-gray-300">{currentOrb.metricLabel}</span>
                    <span style={{ color: riskColor(latestRisk.risk_level) }}>{(latestRisk.flood_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center text-[14px] font-mono pb-2 border-b border-white/5">
                    <span className="text-gray-300">Total Monitored Area</span>
                    <span className="text-white">{latestRisk.total_area_km2.toFixed(0)} km²</span>
                  </div>
                  <div className="flex justify-between items-center text-[14px] font-mono pb-2 border-b border-white/5">
                    <span className="text-gray-300">Water Change</span>
                    <span className={latestRisk.water_change_pct > 0 ? 'text-red-400' : 'text-green-400'}>
                      {latestRisk.water_change_pct > 0 ? '+' : ''}{(latestRisk.water_change_pct * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-[14px] font-mono pb-2 border-b border-white/5">
                    <span className="text-gray-300">Data Source</span>
                    <span className="text-gray-400">{activeSource}</span>
                  </div>
                  <div className="flex justify-between items-center text-[14px] font-mono">
                    <span className="text-gray-300">Last Analyzed</span>
                    <span className="text-gray-400">{latestRisk.timestamp ? new Date(latestRisk.timestamp).toLocaleDateString() : 'N/A'}</span>
                  </div>
                </div>

                {/* Prediction Block */}
                {prediction && (
                  <div className="bg-[#151A22]/80 border rounded-xl p-4 flex flex-col gap-2" style={{ borderColor: riskColor(prediction.predicted_risk_level) + '30' }}>
                    <span className={`text-[13px] text-gray-500 uppercase ${textMono} flex items-center gap-2`}>
                      <Activity size={12} style={{ color: primaryColor }} /> ML Prediction
                    </span>
                    <div className="flex justify-between items-center text-[14px] font-mono">
                      <span className="text-gray-300">Predicted Risk</span>
                      <span style={{ color: riskColor(prediction.predicted_risk_level) }} className="font-bold">{prediction.predicted_risk_level}</span>
                    </div>
                    <div className="flex justify-between items-center text-[14px] font-mono">
                      <span className="text-gray-300">Flood Probability</span>
                      <span style={{ color: primaryColor }}>{(prediction.flood_probability * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between items-center text-[14px] font-mono">
                      <span className="text-gray-300">Model Confidence</span>
                      <span className="text-gray-400">{(prediction.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                )}

                {/* ── Live Automated Detection ── */}
                {liveDetection && (
                  <div className="bg-[#0A1628]/90 border rounded-xl p-4 flex flex-col gap-3" style={{ borderColor: riskColor(liveDetection.detected_risk_level) + '40' }}>
                    <div className="flex items-center justify-between">
                      <span className={`text-[13px] text-gray-500 uppercase ${textMono} flex items-center gap-2`}>
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" /> Automated Detection
                      </span>
                      <button
                        onClick={async () => {
                          if (!selectedRegion || analyzing) return;
                          setAnalyzing(true);
                          const result = await triggerAnalysis(selectedRegion.id);
                          if (result?.detection) setLiveDetection(result.detection);
                          const freshRisk = await fetchRegionRisk(selectedRegion.id);
                          if (freshRisk && !freshRisk.message) setLatestRisk(freshRisk);
                          const freshPred = await fetchPrediction(selectedRegion.id);
                          if (freshPred && freshPred.predicted_risk_level) setPrediction(freshPred);
                          setAnalyzing(false);
                        }}
                        className={`text-[12px] font-mono px-2.5 py-1 rounded border transition-colors ${analyzing ? 'border-cyan-500/30 text-cyan-500 animate-pulse' : 'border-white/10 text-gray-400 hover:border-cyan-500/40 hover:text-cyan-400'}`}
                      >
                        {analyzing ? '⟳ Analyzing...' : '↻ Re-Analyze'}
                      </button>
                    </div>

                    {/* Detected risk */}
                    <div className="flex items-center justify-between">
                      <span className="text-[13px] text-gray-400 font-mono">Detected Risk</span>
                      <span className="text-[15px] font-bold font-mono" style={{ color: riskColor(liveDetection.detected_risk_level) }}>
                        {liveDetection.detected_risk_level}
                      </span>
                    </div>

                    {/* Live data grid */}
                    <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-[13px] font-mono">
                      <span className="text-gray-500">River Discharge</span>
                      <span className="text-right text-cyan-400">{liveDetection.river_discharge_m3s} m³/s</span>
                      <span className="text-gray-500">Discharge Anomaly</span>
                      <span className={`text-right ${liveDetection.discharge_anomaly_sigma > 1.5 ? 'text-red-400' : liveDetection.discharge_anomaly_sigma > 0.8 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                        {liveDetection.discharge_anomaly_sigma > 0 ? '+' : ''}{liveDetection.discharge_anomaly_sigma}σ
                      </span>
                      <span className="text-gray-500">7-Day Rainfall</span>
                      <span className="text-right text-blue-400">{liveDetection.rainfall_7d_mm} mm</span>
                      <span className="text-gray-500">Forecast Rain</span>
                      <span className="text-right text-blue-300">{liveDetection.rainfall_forecast_mm} mm</span>
                      <span className="text-gray-500">Elevation</span>
                      <span className="text-right text-gray-400">{liveDetection.elevation_m} m</span>
                      <span className="text-gray-500">Confidence</span>
                      <span className="text-right text-gray-400">{(liveDetection.confidence_score * 100).toFixed(0)}%</span>
                    </div>

                    {/* Alert */}
                    {liveDetection.alert_triggered && (
                      <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-[13px] text-red-300 font-mono">
                        {liveDetection.alert_message}
                      </div>
                    )}

                    <div className="text-[12px] text-gray-600 font-mono">
                      Sources: {liveDetection.data_sources?.join(', ')}
                    </div>
                  </div>
                )}

                {/* ── GloFAS Live Validation Panel ── */}
                {validation && (
                  <div className="bg-[#151A22]/80 border rounded-xl p-4 flex flex-col gap-2" style={{ borderColor: validation.validation.agreement ? '#22c55e30' : '#ef444430' }}>
                    <span className={`text-[13px] text-gray-500 uppercase ${textMono} flex items-center gap-2`}>
                      <Satellite size={12} className="text-emerald-400" /> Live Validation — GloFAS
                    </span>

                    {/* Agreement Badge */}
                    <div className="flex items-center gap-2">
                      <span className={`text-[13px] font-bold px-2.5 py-1 rounded ${validation.validation.agreement ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                        {validation.validation.agreement ? '✓ ALIGNED' : '⚠ DIVERGENT'}
                      </span>
                      <span className="text-[13px] text-gray-500 font-mono">
                        Score: {(validation.validation.agreement_score * 100).toFixed(0)}%
                      </span>
                    </div>

                    {/* Comparison table */}
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[13px] font-mono mt-1">
                      <span className="text-gray-500">Our Prediction</span>
                      <span style={{ color: riskColor(validation.validation.our_prediction) }} className="text-right font-bold">
                        {validation.validation.our_prediction}
                      </span>
                      <span className="text-gray-500">GloFAS Risk</span>
                      <span style={{ color: riskColor(validation.validation.glofas_risk_level) }} className="text-right font-bold">
                        {validation.validation.glofas_risk_level}
                      </span>
                      <span className="text-gray-500">River Discharge</span>
                      <span className="text-right text-cyan-400">
                        {validation.validation.glofas_discharge_m3s.toFixed(1)} m³/s
                      </span>
                      <span className="text-gray-500">Discharge Anomaly</span>
                      <span className={`text-right ${validation.validation.glofas_discharge_anomaly > 1.5 ? 'text-red-400' : validation.validation.glofas_discharge_anomaly > 0.8 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                        {validation.validation.glofas_discharge_anomaly > 0 ? '+' : ''}{validation.validation.glofas_discharge_anomaly.toFixed(2)}σ
                      </span>
                    </div>

                    <div className="text-[12px] text-gray-600 font-mono mt-1">
                      Source: {validation.validation.data_source}
                    </div>
                  </div>
                )}

                {/* ── ML vs GloFAS Explainability Panel ── */}
                <button
                  onClick={async () => {
                    if (!selectedRegion || explainLoading) return;
                    setExplainLoading(true);
                    const data = await fetchExplanation(selectedRegion.id);
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
                  {explainLoading ? 'Analyzing live data...' : explanation ? '↻ Re-Analyze ML vs GloFAS' : 'Analyze: ML Prediction vs GloFAS'}
                </button>

                {explanation && (() => {
                  const ml = explanation.ml_prediction;
                  const gf = explanation.glofas_assessment;
                  const comp = explanation.comparison;
                  const proof = explanation.independence_proof;
                  return (
                    <>
                      {/* Side-by-side comparison */}
                      <div className="bg-[#0A1628]/90 border border-white/10 rounded-xl p-4 flex flex-col gap-3">
                        <span className={`text-[13px] text-gray-500 uppercase ${textMono} tracking-widest`}>
                          ML Prediction vs GloFAS Ground Truth
                        </span>

                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-[#151A22] border border-violet-500/20 rounded-lg p-3 text-center">
                            <div className={`text-[12px] text-gray-500 uppercase ${textMono} mb-1`}>Our ML Model</div>
                            <span className="text-sm font-bold px-2 py-0.5 rounded"
                              style={{ color: riskColor(ml.risk_level), backgroundColor: riskColor(ml.risk_level) + '20', border: `1px solid ${riskColor(ml.risk_level)}40` }}>
                              {ml.risk_level}
                            </span>
                            <div className={`text-[13px] text-gray-400 ${textMono} mt-1.5`}>
                              Prob: <b className="text-white">{(ml.probability * 100).toFixed(0)}%</b>
                            </div>
                            <div className={`text-[12px] text-violet-400/60 ${textMono} mt-1`}>Weather + Terrain</div>
                          </div>
                          <div className="bg-[#151A22] border border-cyan-500/20 rounded-lg p-3 text-center">
                            <div className={`text-[12px] text-gray-500 uppercase ${textMono} mb-1`}>GloFAS Ground Truth</div>
                            <span className="text-sm font-bold px-2 py-0.5 rounded"
                              style={{ color: riskColor(gf.risk_level), backgroundColor: riskColor(gf.risk_level) + '20', border: `1px solid ${riskColor(gf.risk_level)}40` }}>
                              {gf.risk_level}
                            </span>
                            <div className={`text-[13px] text-gray-400 ${textMono} mt-1.5`}>
                              Discharge: <b className="text-cyan-400">{gf.discharge_m3s} m³/s</b>
                            </div>
                            <div className={`text-[12px] text-cyan-400/60 ${textMono} mt-1`}>River Discharge</div>
                          </div>
                        </div>

                        {/* Agreement badge */}
                        <div className="flex items-center justify-center gap-2 py-1">
                          <span className={`text-[13px] font-bold font-mono px-3 py-1 rounded-full border ${comp.agreement && comp.agreement_score >= 0.9
                            ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                            : comp.agreement
                              ? 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
                              : 'bg-red-500/15 text-red-400 border-red-500/30'
                            }`}>
                            {comp.agreement && comp.agreement_score >= 0.9 ? 'MATCH' : comp.agreement ? 'CLOSE' : 'DIFFERS'}
                            <span className="ml-1 text-[12px] opacity-70">{(comp.agreement_score * 100).toFixed(0)}%</span>
                          </span>
                        </div>
                      </div>

                      {/* Why they agree/differ */}
                      <div className="bg-[#0A1628]/90 border border-white/10 rounded-xl p-4 flex flex-col gap-2">
                        <span className={`text-[13px] uppercase ${textMono} tracking-widest ${comp.agreement && comp.agreement_score >= 0.9 ? 'text-emerald-400' : 'text-amber-400'
                          }`}>
                          {comp.agreement && comp.agreement_score >= 0.9 ? 'Why Both Sources Agree' : 'Why Predictions Differ'}
                        </span>
                        <p className={`text-sm text-gray-300 ${textMono} leading-relaxed`}>{comp.summary}</p>
                        {comp.difference_reasons.map((reason, i) => (
                          <div key={i} className={`text-[13px] text-gray-400 leading-relaxed p-2.5 rounded-lg border-l-2 bg-[#151A22]/80 ${comp.agreement && comp.agreement_score >= 0.9 ? 'border-emerald-500/40' : 'border-amber-500/40'
                            }`}>
                            {reason}
                          </div>
                        ))}
                      </div>

                      {/* Top Contributing Factors */}
                      <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-2.5">
                        <span className={`text-[13px] text-gray-500 uppercase ${textMono} tracking-widest`}>
                          Top Contributing Factors
                        </span>
                        {ml.top_drivers.slice(0, 5).map(d => (
                          <div key={d.feature} className="flex flex-col gap-0.5">
                            <div className="flex items-center justify-between">
                              <span className={`text-[13px] text-gray-400 ${textMono}`}>{d.feature.replace(/_/g, ' ')}</span>
                              <span className={`text-[13px] text-gray-300 ${textMono} font-bold`}>{(d.importance * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full h-2 bg-[#0B0E11] rounded-full overflow-hidden">
                              <div className="h-full rounded-full transition-all"
                                style={{
                                  width: `${Math.min(d.importance * 400, 100)}%`,
                                  backgroundColor: d.importance > 0.15 ? '#f59e0b' : d.importance > 0.08 ? primaryColor : '#4b5563'
                                }}
                              />
                            </div>
                            <span className={`text-[12px] text-gray-500 ${textMono}`}>{d.influence}</span>
                          </div>
                        ))}
                      </div>

                      {/* Model Independence Proof */}
                      <div className="bg-[#0A0E18] border border-blue-500/20 rounded-xl p-4 flex flex-col gap-2">
                        <button onClick={() => setShowIndependence(!showIndependence)}
                          className={`text-[13px] uppercase ${textMono} tracking-widest text-blue-400 flex items-center justify-between w-full`}>
                          Model Independence Proof
                          <ChevronDown size={14} className={`transition-transform ${showIndependence ? 'rotate-180' : ''}`} />
                        </button>
                        <p className={`text-[12px] text-gray-500 ${textMono}`}>
                          Proves the ML model predicts independently and does NOT copy GloFAS.
                        </p>

                        {showIndependence && (
                          <div className="flex flex-col gap-2 mt-1">
                            <div className="flex gap-2 items-start">
                              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shrink-0 mt-0.5">USES</span>
                              <span className={`text-[13px] text-gray-400 ${textMono} leading-relaxed`}>{proof.model_uses}</span>
                            </div>
                            <div className="flex gap-2 items-start">
                              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 shrink-0 mt-0.5">NO</span>
                              <span className={`text-[13px] text-gray-400 ${textMono} leading-relaxed`}>{proof.model_does_not_use}</span>
                            </div>
                            <div className="flex gap-2 items-start">
                              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shrink-0 mt-0.5">GLOFAS</span>
                              <span className={`text-[13px] text-gray-400 ${textMono} leading-relaxed`}>{proof.glofas_uses}</span>
                            </div>
                            <div className="flex gap-2 items-start">
                              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 shrink-0 mt-0.5">TRAIN</span>
                              <span className={`text-[13px] text-gray-400 ${textMono} leading-relaxed`}>{proof.how_training_works}</span>
                            </div>
                            <div className="flex gap-2 items-start">
                              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shrink-0 mt-0.5">VERIFY</span>
                              <span className={`text-[13px] text-gray-400 ${textMono} leading-relaxed`}>{proof.verification}</span>
                            </div>

                            {/* Methodology comparison */}
                            <div className="mt-2 pt-2 border-t border-white/5">
                              <span className={`text-[12px] text-gray-500 uppercase ${textMono}`}>Methodology Comparison</span>
                              <div className="mt-1.5 flex flex-col gap-1.5">
                                <div className="text-[13px]">
                                  <b className="text-violet-400">Our ML Model: </b>
                                  <span className="text-gray-500">{comp.our_methodology}</span>
                                </div>
                                <div className="text-[13px]">
                                  <b className="text-cyan-400">GloFAS System: </b>
                                  <span className="text-gray-500">{comp.glofas_methodology}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </>
                  );
                })()}

                {/* Risk History Mini Chart */}
                {chartData.length > 0 && (
                  <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4">
                    <span className={`text-[13px] text-gray-500 uppercase ${textMono} block mb-3`}>{currentOrb.chartLabel}</span>
                    <div className="h-[120px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
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
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-violet-300 flex items-center gap-2`}>
                      <TrendingUp size={14} className="text-violet-400" /> 6-Month Forecast
                    </span>
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

                          {forecastData && !forecastLoading && (
                            <>
                              {/* Summary Strip */}
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className={`text-[12px] font-mono px-2.5 py-1 rounded-full border ${forecastData.summary.overall_trend === 'escalating' ? 'bg-red-500/15 text-red-400 border-red-500/30'
                                  : forecastData.summary.overall_trend === 'declining' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                    : 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
                                  }`}>
                                  Trend: {forecastData.summary.overall_trend.toUpperCase()}
                                </span>
                                <span className="text-[12px] font-mono text-gray-500">
                                  Peak: {forecastData.summary.peak_risk_month} ({(forecastData.summary.peak_probability * 100).toFixed(0)}%)
                                </span>
                              </div>

                              {/* Risk Probability Chart */}
                              <div className="h-[140px]">
                                <ResponsiveContainer width="100%" height="100%">
                                  <AreaChart data={forecastData.monthly_forecast.map(m => ({
                                    month: m.month_name.split(' ')[0].slice(0, 3),
                                    risk: Math.round(m.risk_probability * 100),
                                    lower: Math.round(m.confidence_lower * 100),
                                    upper: Math.round(m.confidence_upper * 100),
                                  }))}>
                                    <defs>
                                      <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4} />
                                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                      </linearGradient>
                                      <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.1} />
                                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05} />
                                      </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="month" tick={{ fill: '#6b7280', fontSize: 11 }} />
                                    <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} domain={[0, 100]} unit="%" />
                                    <RechartsTooltip
                                      contentStyle={{ background: '#0A1628', border: '1px solid rgba(139,92,246,0.3)', borderRadius: 8, fontSize: 12 }}
                                      formatter={(value: number | string | undefined) => [`${value ?? 0}%`, 'Risk']}
                                    />
                                    <Area type="monotone" dataKey="upper" stroke="none" fill="url(#confGrad)" />
                                    <Area type="monotone" dataKey="lower" stroke="none" fill="transparent" />
                                    <Area type="monotone" dataKey="risk" stroke="#8b5cf6" fill="url(#forecastGrad)" strokeWidth={2.5} dot={{ r: 3, fill: '#8b5cf6' }} />
                                  </AreaChart>
                                </ResponsiveContainer>
                              </div>

                              {/* Monthly Cards */}
                              <div className="grid grid-cols-3 gap-1.5">
                                {forecastData.monthly_forecast.slice(0, 6).map(m => (
                                  <div key={m.month} className="bg-[#151A22] rounded-lg p-2 text-center border border-white/5">
                                    <div className="text-[11px] text-gray-500 font-mono">{m.month_name.split(' ')[0].slice(0, 3)}</div>
                                    <div className="text-[14px] font-bold font-mono mt-0.5" style={{ color: m.risk_level === 'CRITICAL' ? '#ef4444' : m.risk_level === 'HIGH' ? '#f59e0b' : m.risk_level === 'MEDIUM' ? '#eab308' : '#22c55e' }}>
                                      {(m.risk_probability * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-[10px] text-gray-600 font-mono">{m.risk_level}</div>
                                  </div>
                                ))}
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
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-amber-300 flex items-center gap-2`}>
                      <Sparkles size={14} className="text-amber-400" /> AI Insights
                    </span>
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
                              {/* Narrative */}
                              <div className="text-[13px] text-gray-300 leading-relaxed font-sans whitespace-pre-line">
                                {nlgSummary.narrative.split('**').map((part, i) =>
                                  i % 2 === 1 ? <strong key={i} className="text-white">{part}</strong> : <span key={i}>{part}</span>
                                )}
                              </div>

                              {/* Highlights */}
                              <div className="flex flex-col gap-1.5">
                                {nlgSummary.highlights.map((h, i) => (
                                  <div key={i} className="flex items-start gap-2 text-[12px] font-mono text-gray-400">
                                    <span className="text-amber-400 mt-0.5">▸</span>
                                    {h}
                                  </div>
                                ))}
                              </div>

                              {/* Trend */}
                              {nlgSummary.trend_narrative && (
                                <div className="bg-[#151A22] rounded-lg p-3 border border-white/5">
                                  <div className="text-[12px] text-gray-300 leading-relaxed font-sans">
                                    {nlgSummary.trend_narrative.split('**').map((part, i) =>
                                      i % 2 === 1 ? <strong key={i} className="text-white">{part}</strong> : <span key={i}>{part}</span>
                                    )}
                                  </div>
                                </div>
                              )}

                              {/* Meta */}
                              <div className="flex items-center gap-2 text-[11px] text-gray-600 font-mono">
                                <span className={`px-1.5 py-0.5 rounded text-[10px] border ${nlgSummary.engine === 'gpt-4o-mini' ? 'border-emerald-500/30 text-emerald-400' : 'border-gray-600 text-gray-500'
                                  }`}>
                                  {nlgSummary.engine === 'gpt-4o-mini' ? '✨ GPT-4' : '⚙ Template'}
                                </span>
                                <span>Generated {new Date(nlgSummary.generated_at).toLocaleTimeString()}</span>
                              </div>
                            </>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* ── Multi-Sensor Fusion Panel ── */}
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
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-cyan-300 flex items-center gap-2`}>
                      <Radio size={14} className="text-cyan-400" /> Sensor Fusion
                    </span>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFusion ? 'rotate-90' : ''}`} />
                  </button>
                  <AnimatePresence>
                    {showFusion && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-2.5">
                          {fusionLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" /></div>}
                          {fusionData && !fusionLoading && (
                            <>
                              <div className="grid grid-cols-2 gap-2">
                                {[
                                  { label: 'Flood Conf.', value: fusionData.flood_confidence, color: '#00E5FF' },
                                  { label: 'Veg Stress', value: fusionData.vegetation_stress, color: '#4ade80' },
                                  { label: 'Soil Sat.', value: fusionData.soil_saturation, color: '#c084fc' },
                                  { label: 'Quality', value: fusionData.quality_score, color: '#facc15' },
                                ].map(s => (
                                  <div key={s.label} className="bg-[#151A22] rounded-lg p-2 border border-white/5">
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
                              {fusionData.cloud_penetration_used && (
                                <div className="text-[11px] font-mono text-cyan-400 flex items-center gap-1.5">
                                  <Radio size={10} /> SAR cloud-penetration active
                                </div>
                              )}
                              <div className="text-[10px] font-mono text-gray-600">
                                Sensors: {fusionData.sensors_fused?.join(', ')}
                              </div>
                              {fusionData.thermal_anomaly !== 0 && (
                                <div className="text-[11px] font-mono text-gray-500">
                                  Thermal: {fusionData.thermal_anomaly > 0 ? '+' : ''}{fusionData.thermal_anomaly?.toFixed(1)}°C anomaly
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
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-rose-300 flex items-center gap-2`}>
                      <Shield size={14} className="text-rose-400" /> Compound Risk
                    </span>
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
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-emerald-300 flex items-center gap-2`}>
                      <DollarSign size={14} className="text-emerald-400" /> Financial Impact
                    </span>
                    <ChevronRight size={14} className={`text-gray-500 transition-transform duration-300 ${showFinancial ? 'rotate-90' : ''}`} />
                  </button>
                  <AnimatePresence>
                    {showFinancial && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-4 pb-4 flex flex-col gap-2.5">
                          {financialLoading && <div className="flex justify-center py-6"><div className="w-7 h-7 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" /></div>}
                          {financialData && !financialLoading && (
                            <>
                              <div className="text-center">
                                <div className="text-[10px] text-gray-500 font-mono">TOTAL EXPOSURE</div>
                                <div className="text-[22px] font-bold font-mono text-emerald-400">
                                  ${financialData.total_impact_usd?.toLocaleString() ?? '0'}
                                </div>
                              </div>
                              <div className="grid grid-cols-3 gap-1.5">
                                {[
                                  { label: 'Direct', value: financialData.direct_damage_usd },
                                  { label: 'Indirect', value: financialData.indirect_costs_usd },
                                  { label: 'Recovery', value: financialData.recovery_cost_usd },
                                ].map(m => (
                                  <div key={m.label} className="bg-[#151A22] rounded-lg p-2 text-center border border-white/5">
                                    <div className="text-[10px] text-gray-500 font-mono">{m.label}</div>
                                    <div className="text-[12px] font-mono text-gray-300">${(m.value / 1000).toFixed(0)}K</div>
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
                    <span className={`text-[13px] uppercase ${textMono} tracking-widest text-gray-400 flex items-center gap-2`}>
                      <ThumbsUp size={14} className="text-gray-500" /> Model Feedback
                    </span>
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

              </div>

              {/* Download Action */}
              <div className="p-6 border-t border-white/10 bg-black/20">
                <a href={selectedRegion ? getReportDownloadUrl(selectedRegion.id) : '#'} target="_blank" rel="noopener noreferrer"
                  className="w-full flex items-center justify-center gap-2 py-3 bg-[#151A22] hover:bg-white/10 text-white font-mono text-[13px] uppercase tracking-widest border border-white/10 hover:border-[#00E5FF]/40 rounded-lg transition-colors group">
                  <Download size={14} className="text-gray-400 group-hover:text-[#00E5FF]" /> Download Structured Report
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
                      <span className={`text-sm uppercase tracking-widest ${textMono} font-bold flex items-center gap-2`} style={{ color: riskColor(adHocData.detection?.detected_risk_level || 'LOW') }}>
                        <AlertTriangle size={16} />
                        {adHocData.detection?.detected_risk_level || 'ANALYZING'} RISK
                      </span>
                    ) : (
                      <span className={`text-sm uppercase tracking-widest ${textMono} font-bold text-cyan-400 flex items-center gap-2`}>
                        <Activity size={16} className="animate-pulse" /> Analyzing...
                      </span>
                    )}
                    <span className="text-white text-[15px] font-semibold mt-1">{adHocLocation.name} — {currentOrb.panelTitle}</span>
                  </div>
                  <button onClick={() => { setAdHocLocation(null); setAdHocData(null); setAdHocExplanation(null); }} className="text-gray-500 hover:text-white"><X size={18} /></button>
                </div>

              </div>

              {/* Content */}
              <div className="flex-grow p-6 flex flex-col gap-5 overflow-y-auto">
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
                      {/* Prediction */}
                      {pred && (
                        <div className="bg-[#151A22]/80 border rounded-xl p-4 flex flex-col gap-2" style={{ borderColor: riskColor(pred.predicted_risk_level) + '30' }}>
                          <span className={`text-[13px] text-gray-500 uppercase ${textMono} flex items-center gap-2`}>
                            <Activity size={12} style={{ color: primaryColor }} /> ML Prediction
                          </span>
                          <div className="flex justify-between items-center text-[14px] font-mono">
                            <span className="text-gray-300">Predicted Risk</span>
                            <span style={{ color: riskColor(pred.predicted_risk_level) }} className="font-bold">{pred.predicted_risk_level}</span>
                          </div>
                          <div className="flex justify-between items-center text-[14px] font-mono">
                            <span className="text-gray-300">Flood Probability</span>
                            <span style={{ color: primaryColor }}>{(pred.flood_probability * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between items-center text-[14px] font-mono">
                            <span className="text-gray-300">Model Confidence</span>
                            <span className="text-gray-400">{(pred.confidence * 100).toFixed(0)}%</span>
                          </div>
                          {pred.model_version && (
                            <div className="text-[12px] text-gray-600 font-mono mt-1">Engine: {pred.model_version}</div>
                          )}
                        </div>
                      )}

                      {/* Live Detection */}
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

                      {/* Validation */}
                      {val && (
                        <div className="bg-[#151A22]/80 border rounded-xl p-4 flex flex-col gap-2" style={{ borderColor: val.agreement ? '#22c55e30' : '#ef444430' }}>
                          <span className={`text-[13px] text-gray-500 uppercase ${textMono} flex items-center gap-2`}>
                            <Satellite size={12} className="text-emerald-400" /> Live Validation — GloFAS
                          </span>
                          <div className="flex items-center gap-2">
                            <span className={`text-[13px] font-bold px-2.5 py-1 rounded ${val.agreement ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                              {val.agreement ? '✓ ALIGNED' : '⚠ DIVERGENT'}
                            </span>
                            <span className="text-[13px] text-gray-500 font-mono">Score: {(val.agreement_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[13px] font-mono mt-1">
                            <span className="text-gray-500">Our Prediction</span>
                            <span style={{ color: riskColor(val.our_prediction) }} className="text-right font-bold">{val.our_prediction}</span>
                            <span className="text-gray-500">GloFAS Risk</span>
                            <span style={{ color: riskColor(val.glofas_risk_level) }} className="text-right font-bold">{val.glofas_risk_level}</span>
                            <span className="text-gray-500">River Discharge</span>
                            <span className="text-right text-cyan-400">{val.glofas_discharge_m3s?.toFixed(1)} m³/s</span>
                          </div>
                        </div>
                      )}

                      {/* Explainability Button */}
                      <button
                        onClick={async () => {
                          if (!adHocLocation || explainLoading) return;
                          setExplainLoading(true);
                          const data = await explainLocation(adHocLocation.lat, adHocLocation.lon);
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
                        {explainLoading ? 'Analyzing live data...' : adHocExplanation ? '↻ Re-Analyze ML vs GloFAS' : 'Analyze: ML Prediction vs GloFAS'}
                      </button>

                      {/* Explanation Panels */}
                      {adHocExplanation && (() => {
                        const ml = adHocExplanation.ml_prediction;
                        const gf = adHocExplanation.glofas_assessment;
                        const comp = adHocExplanation.comparison;
                        return (
                          <>
                            <div className="bg-[#0A1628]/90 border border-white/10 rounded-xl p-4 flex flex-col gap-3">
                              <span className={`text-[13px] text-gray-500 uppercase ${textMono} tracking-widest`}>ML Prediction vs GloFAS Ground Truth</span>
                              <div className="grid grid-cols-2 gap-2">
                                <div className="bg-[#151A22] border border-violet-500/20 rounded-lg p-3 text-center">
                                  <div className={`text-[12px] text-gray-500 uppercase ${textMono} mb-1`}>Our ML Model</div>
                                  <span className="text-sm font-bold px-2 py-0.5 rounded" style={{ color: riskColor(ml.risk_level), backgroundColor: riskColor(ml.risk_level) + '20', border: `1px solid ${riskColor(ml.risk_level)}40` }}>
                                    {ml.risk_level}
                                  </span>
                                  <div className={`text-[13px] text-gray-400 ${textMono} mt-1.5`}>Prob: <b className="text-white">{(ml.probability * 100).toFixed(0)}%</b></div>
                                </div>
                                <div className="bg-[#151A22] border border-cyan-500/20 rounded-lg p-3 text-center">
                                  <div className={`text-[12px] text-gray-500 uppercase ${textMono} mb-1`}>GloFAS Ground Truth</div>
                                  <span className="text-sm font-bold px-2 py-0.5 rounded" style={{ color: riskColor(gf.risk_level), backgroundColor: riskColor(gf.risk_level) + '20', border: `1px solid ${riskColor(gf.risk_level)}40` }}>
                                    {gf.risk_level}
                                  </span>
                                  <div className={`text-[13px] text-gray-400 ${textMono} mt-1.5`}>Discharge: <b className="text-cyan-400">{gf.discharge_m3s} m³/s</b></div>
                                </div>
                              </div>
                              <div className="flex items-center justify-center gap-2 py-1">
                                <span className={`text-[13px] font-bold font-mono px-3 py-1 rounded-full border ${comp.agreement && comp.agreement_score >= 0.9 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                  : comp.agreement ? 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
                                    : 'bg-red-500/15 text-red-400 border-red-500/30'
                                  }`}>
                                  {comp.agreement && comp.agreement_score >= 0.9 ? 'MATCH' : comp.agreement ? 'CLOSE' : 'DIFFERS'}
                                  <span className="ml-1 text-[12px] opacity-70">{(comp.agreement_score * 100).toFixed(0)}%</span>
                                </span>
                              </div>
                            </div>

                            {/* Why agree/differ */}
                            <div className="bg-[#0A1628]/90 border border-white/10 rounded-xl p-4 flex flex-col gap-2">
                              <span className={`text-[13px] uppercase ${textMono} tracking-widest ${comp.agreement && comp.agreement_score >= 0.9 ? 'text-emerald-400' : 'text-amber-400'}`}>
                                {comp.agreement && comp.agreement_score >= 0.9 ? 'Why Both Sources Agree' : 'Why Predictions Differ'}
                              </span>
                              <p className={`text-sm text-gray-300 ${textMono} leading-relaxed`}>{comp.summary}</p>
                              {comp.difference_reasons.map((reason: string, i: number) => (
                                <div key={i} className={`text-[13px] text-gray-400 leading-relaxed p-2.5 rounded-lg border-l-2 bg-[#151A22]/80 ${comp.agreement && comp.agreement_score >= 0.9 ? 'border-emerald-500/40' : 'border-amber-500/40'
                                  }`}>
                                  {reason}
                                </div>
                              ))}
                            </div>

                            {/* Top factors */}
                            <div className="bg-[#151A22]/80 border border-white/5 rounded-xl p-4 flex flex-col gap-2.5">
                              <span className={`text-[13px] text-gray-500 uppercase ${textMono} tracking-widest`}>Top Contributing Factors</span>
                              {ml.top_drivers.slice(0, 5).map((d: { feature: string; importance: number; influence: string }) => (
                                <div key={d.feature} className="flex flex-col gap-0.5">
                                  <div className="flex items-center justify-between">
                                    <span className={`text-[13px] text-gray-400 ${textMono}`}>{d.feature.replace(/_/g, ' ')}</span>
                                    <span className={`text-[13px] text-gray-300 ${textMono} font-bold`}>{(d.importance * 100).toFixed(1)}%</span>
                                  </div>
                                  <div className="w-full h-2 bg-[#0B0E11] rounded-full overflow-hidden">
                                    <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(d.importance * 400, 100)}%`, backgroundColor: d.importance > 0.15 ? '#f59e0b' : d.importance > 0.08 ? primaryColor : '#4b5563' }} />
                                  </div>
                                  <span className={`text-[12px] text-gray-500 ${textMono}`}>{d.influence}</span>
                                </div>
                              ))}
                            </div>
                          </>
                        );
                      })()}
                    </>
                  );
                })()}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ D. TIME-SERIES (BOTTOM CENTER) ═══ */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`absolute left-[390px] right-[430px] bottom-6 h-[180px] z-20 ${glassClass} flex flex-col overflow-hidden`}>
        <div className="px-5 py-3 border-b border-white/10 flex justify-between items-center bg-black/40">
          <h3 className={`text-[12px] uppercase text-gray-400 ${textMono} tracking-widest`}>
            {selectedRegion ? `${selectedRegion.name} — ${currentOrb.panelTitle}` : adHocLocation ? `${adHocLocation.name} — ${currentOrb.panelTitle}` : 'Time-Series & Change Detection'}
          </h3>
          <div className="flex items-center gap-3">
            <span className={`text-[12px] text-gray-500 flex items-center gap-2 ${textMono}`}>
              <SlidersHorizontal size={12} className="text-gray-400" /> Comparison
            </span>
            <div onClick={() => setComparisonMode(!comparisonMode)}
              className="w-8 h-4 border border-white/10 rounded-full cursor-pointer relative transition-colors"
              style={{ backgroundColor: comparisonMode ? primaryColor + '30' : '#151A22', borderColor: comparisonMode ? primaryColor : 'rgba(255,255,255,0.1)' }}
            >
              <div className="absolute top-0.5 w-2.5 h-2.5 rounded-full transition-all"
                style={{ backgroundColor: comparisonMode ? primaryColor : '#9ca3af', left: comparisonMode ? '14px' : '2px' }} />
            </div>
          </div>
        </div>
        <div className="flex-grow flex flex-col relative p-4 min-h-0">
          <div className="flex-grow w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData.length > 0 ? chartData : [{ date: '--', flood: 0, confidence: 0, water_change: 0 }]}>
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
                {comparisonMode && <Area type="monotone" dataKey="confidence" stroke="#f97316" fill="url(#chartColor2)" strokeWidth={1.5} strokeDasharray="4 4" name="Confidence %" />}
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

    </main>
  );
}
