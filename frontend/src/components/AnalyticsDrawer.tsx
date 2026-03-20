"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { X, SlidersHorizontal, Download, Play, Pause } from "lucide-react";
import { useState } from "react";

interface AnalyticsDrawerProps {
    region: any;
    changes: any[];
    onClose: () => void;
    activeOrb: string;
}

export function AnalyticsDrawer({ region, changes, onClose, activeOrb }: AnalyticsDrawerProps) {
    const [isPlaying, setIsPlaying] = useState(false);
    const color = activeOrb === "flood" ? "var(--color-cyan)" : "var(--color-primary)";

    const sortedChanges = changes
        .filter(c => c.region_id === region?.id)
        .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
        .map(c => ({
            date: new Date(c.timestamp).toLocaleDateString(),
            value: c.water_change_pct * 100,
            area: c.affected_area_km2
        }));

    // Derive real metrics from change events
    const absChanges = sortedChanges.map(c => Math.abs(c.value));
    const severityIndex = absChanges.length > 0
        ? Math.min(10, Math.round((Math.max(...absChanges) / 10) * 10) / 10)
        : null;
    const riskDelta = sortedChanges.length >= 2
        ? ((sortedChanges[sortedChanges.length - 1].value - sortedChanges[0].value)).toFixed(1)
        : null;

    return (
        <AnimatePresence>
            {region && (
                <motion.div
                    initial={{ y: 600, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: 600, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    className="absolute bottom-6 right-6 left-[320px] h-72 glass-panel rounded-2xl z-20 flex flex-col overflow-hidden"
                >
                    {/* Header */}
                    <div className="h-14 border-b border-white/10 bg-black/20 flex items-center justify-between px-6">
                        <div className="flex items-center gap-4">
                            <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">{region.name} — Time-Series Analysis</h3>
                            <div className="px-2 py-0.5 rounded bg-white/10 text-[10px] text-white/70 font-mono border border-white/5">
                                {activeOrb === "flood" ? "WATER COVERAGE %" : "DEFORESTATION INDEX"}
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            <button className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-white/60 hover:text-white px-3 py-1.5 rounded-md hover:bg-white/10 transition-colors">
                                <Download size={14} /> Export Report
                            </button>
                            <div className="w-px h-4 bg-white/20" />
                            <button onClick={onClose} className="text-white/40 hover:text-white transition-colors"><X size={18} /></button>
                        </div>
                    </div>

                    <div className="flex flex-grow h-full overflow-hidden">
                        {/* Left: Chart */}
                        <div className="flex-grow p-6 flex flex-col relative h-[230px]">

                            <div className="flex-grow w-full h-full min-h-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={sortedChanges.length ? sortedChanges : [{ date: 'Jan', value: 0 }, { date: 'Feb', value: 0 }]}>
                                        <defs>
                                            <linearGradient id="colorBrand" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={color} stopOpacity={0.5} />
                                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                        <XAxis dataKey="date" stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} tickMargin={10} axisLine={false} />
                                        <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} tickMargin={10} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: "rgba(11, 14, 17, 0.95)", backdropFilter: "blur(10px)", borderColor: "rgba(255,255,255,0.1)", borderRadius: "8px", fontFamily: 'var(--font-mono)' }}
                                            itemStyle={{ color: color, fontWeight: 'bold' }}
                                        />
                                        <Area type="monotone" dataKey="value" stroke={color} strokeWidth={2} fillOpacity={1} fill="url(#colorBrand)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Time Scrubber (Overlay at bottom of chart area) */}
                            <div className="h-10 mt-2 flex items-center gap-4 px-2">
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center text-white hover:bg-white/20 hover:text-[var(--color-primary)] transition-colors"
                                >
                                    {isPlaying ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
                                </button>
                                <div className="flex-grow h-1.5 bg-white/10 rounded-full relative cursor-pointer group">
                                    <div className="absolute top-0 left-0 bottom-0 bg-[var(--color-primary)] rounded-full w-3/4 group-hover:bg-[var(--color-primary-light)] transition-colors" style={{ backgroundColor: color }} />
                                    <div className="absolute top-1/2 -mt-1.5 -ml-1.5 left-3/4 w-3 h-3 bg-white rounded-full shadow-[0_0_10px_white]" />
                                </div>
                                <span className="text-[11px] font-mono text-white/50">2010 - 2026</span>
                            </div>
                        </div>

                        {/* Right: Metrics Gauge Grid */}
                        <div className="w-80 border-l border-white/10 bg-black/10 p-6 flex flex-col gap-6 overflow-y-auto">

                            <div>
                                <h4 className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-[var(--color-text-muted)] font-mono mb-3">
                                    <SlidersHorizontal size={12} /> Comparison Mode
                                </h4>
                                <div className="glass-panel p-1 rounded-lg flex text-xs font-mono font-medium">
                                    <div className="flex-1 py-1.5 text-center bg-white/10 rounded-md text-white shadow-sm cursor-default">Current</div>
                                    <div className="flex-1 py-1.5 text-center text-white/50 hover:text-white cursor-pointer transition-colors">+10 Yr Projection</div>
                                </div>
                            </div>

                            <div className="flex flex-col gap-4">
                                <div className="flex justify-between items-end pb-2 border-b border-white/5">
                                    <span className="text-sm text-white/70">Severity Index</span>
                                    {severityIndex !== null
                                        ? <span className="font-mono text-xl font-bold text-white">{severityIndex}<span className="text-[10px] text-white/40 ml-1">/10</span></span>
                                        : <span className="font-mono text-sm text-white/40">No data</span>
                                    }
                                </div>
                                <div className="flex justify-between items-end pb-2 border-b border-white/5">
                                    <span className="text-sm text-white/70">Risk Delta (period)</span>
                                    {riskDelta !== null
                                        ? <span className="font-mono text-xl font-bold" style={{ color }}>{Number(riskDelta) >= 0 ? '+' : ''}{riskDelta}%</span>
                                        : <span className="font-mono text-sm text-white/40">No data</span>
                                    }
                                </div>
                                <div className="flex justify-between items-end pb-2 border-b border-white/5">
                                    <span className="text-sm text-white/70">Change Events</span>
                                    <span className="font-mono text-xl font-bold text-white">{sortedChanges.length}</span>
                                </div>
                            </div>

                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
