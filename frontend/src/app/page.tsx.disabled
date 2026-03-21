"use client";

import { motion, Variants } from "framer-motion";
import { ChevronRight, ShieldCheck, Activity, Map as MapIcon, Layers, BarChart3, Database, Globe, ArrowRight, Zap, Target } from "lucide-react";
import Link from "next/link";
import Image from "next/image";

// ─── Animation Variants ───
const fadeInUp: Variants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
};

const staggerContainer: Variants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.2 }
    }
};

const floatAnim: Variants = {
    hidden: { y: 0 },
    visible: {
        y: [-10, 10, -10],
        transition: { duration: 6, repeat: Infinity, ease: "easeInOut" }
    }
};

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-[#030303] text-white font-sans selection:bg-[#10B981]/30 overflow-x-hidden relative">

            {/* Global Background Glows */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[-10%] w-[600px] h-[600px] bg-[#10B981]/10 rounded-full blur-[150px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[800px] h-[800px] bg-[#0EA5E9]/10 rounded-full blur-[180px]" />
            </div>

            {/* ═══ A. CLEAN MINIMALIST NAVIGATION (Glassmorphic) ═══ */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-[#050505]/60 backdrop-blur-xl border-b border-white/5 transition-all shadow-[0_4px_30px_rgba(0,0,0,0.1)]">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-3 group">
                        <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-[#10B981] to-[#0EA5E9] shadow-[0_0_15px_rgba(16,185,129,0.8)] group-hover:scale-110 transition-transform" />
                        <span className="text-xl font-black tracking-widest text-white">COSMEON</span>
                    </Link>

                    {/* Center Links */}
                    <div className="hidden md:flex items-center gap-8 text-sm font-semibold text-gray-300">
                        <a href="#platform" className="hover:text-white transition-colors relative group">
                            Platform
                            <span className="absolute -bottom-2 left-0 w-0 h-0.5 bg-gradient-to-r from-[#10B981] to-[#0EA5E9] transition-all group-hover:w-full"></span>
                        </a>
                        <a href="#solutions" className="hover:text-white transition-colors relative group">
                            Solutions
                            <span className="absolute -bottom-2 left-0 w-0 h-0.5 bg-gradient-to-r from-[#10B981] to-[#0EA5E9] transition-all group-hover:w-full"></span>
                        </a>
                        <a href="#pipeline" className="hover:text-white transition-colors relative group">
                            Technology
                            <span className="absolute -bottom-2 left-0 w-0 h-0.5 bg-gradient-to-r from-[#10B981] to-[#0EA5E9] transition-all group-hover:w-full"></span>
                        </a>
                    </div>

                    {/* CTA */}
                    <Link
                        href="https://cosmeon.onrender.com/engine"
                        className="px-6 py-2.5 bg-gradient-to-r from-[#10B981] to-[#0EA5E9] hover:from-[#059669] hover:to-[#0284C7] text-white font-bold rounded-full text-sm transition-all shadow-[0_0_20px_rgba(16,185,129,0.4)] hover:shadow-[0_0_30px_rgba(16,185,129,0.7)]"
                    >
                        Launch Engine
                    </Link>
                </div>
            </nav>

            {/* ═══ B. HERO SECTION ═══ */}
            <section id="platform" className="pt-40 pb-20 px-6 max-w-7xl mx-auto min-h-[90vh] flex flex-col md:flex-row items-center gap-16 relative z-10">

                {/* Left: Typography */}
                <motion.div
                    initial="hidden" animate="visible" variants={staggerContainer}
                    className="flex-1 flex flex-col items-start"
                >
                    <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-8 text-sm text-[#10B981] font-bold tracking-wide shadow-[0_0_20px_rgba(16,185,129,0.2)]">
                        <Activity size={16} className="animate-pulse" />
                        <span>Intelligence Engine v2.0 Live</span>
                    </motion.div>

                    <motion.h1 variants={fadeInUp} className="text-5xl md:text-7xl font-extrabold tracking-tight leading-[1.1] mb-6 text-white">
                        Clarity from Space for{" "}
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#10B981] via-[#34D399] to-[#0EA5E9]">
                            Climate Insights
                        </span>{" "}
                        on Earth.
                    </motion.h1>

                    <motion.p variants={fadeInUp} className="text-lg md:text-xl text-gray-400 max-w-xl mb-10 leading-relaxed font-medium">
                        Transforming raw satellite data into actionable intelligence. Automated detection of flood events, environmental risk, and financial exposure.
                    </motion.p>

                    <motion.div variants={fadeInUp} className="flex flex-wrap gap-4">
                        <Link
                            href="https://cosmeon.onrender.com/engine"
                            className="group inline-flex items-center gap-3 px-8 py-4 bg-white text-black text-lg font-bold rounded-full transition-all hover:bg-gray-200 shadow-[0_0_30px_rgba(255,255,255,0.2)]"
                        >
                            Launch Engine
                            <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
                        </Link>
                        <a
                            href="#solutions"
                            className="inline-flex items-center gap-3 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 text-white text-lg font-bold rounded-full transition-all backdrop-blur-md"
                        >
                            Explore Platform
                        </a>
                    </motion.div>
                </motion.div>

                {/* Right: Floating Imagery */}
                <motion.div
                    initial="hidden" animate="visible" variants={fadeInUp}
                    className="flex-1 w-full relative h-[450px] md:h-[600px] rounded-[40px] overflow-hidden shadow-[0_0_80px_rgba(16,185,129,0.15)] bg-[#0A0A0A] border border-white/10 hidden md:block"
                >
                    <motion.div
                        variants={floatAnim}
                        className="absolute inset-0 flex items-center justify-center p-8 h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#10B981]/10 via-transparent to-transparent"
                    >
                        {/* Abstract map/data visualization mockup */}
                        <div className="w-full h-full rounded-2xl bg-[#050505] border border-white/5 relative overflow-hidden flex flex-col shadow-2xl">
                            <div className="h-12 bg-white/5 border-b border-white/5 flex items-center px-4 justify-between backdrop-blur-sm">
                                <div className="flex gap-2">
                                    <div className="w-3 h-3 rounded-full bg-red-500/80" />
                                    <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                                    <div className="w-3 h-3 rounded-full bg-[#10B981]/80" />
                                </div>
                                <div className="text-xs font-mono text-gray-500">LIVE FEED // COSMEON_01</div>
                            </div>
                            <div className="flex-1 relative overflow-hidden">
                                {/* Grid background */}
                                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:30px_30px]" />

                                <Globe size={400} strokeWidth={0.5} className="absolute -right-24 -bottom-24 text-[#0EA5E9]/20 animate-[spin_60s_linear_infinite]" />

                                {/* Overlay cards */}
                                <div className="absolute top-8 left-8 bg-[#0F0F0F]/80 backdrop-blur-xl p-5 rounded-2xl border border-white/10 flex flex-col gap-3 w-56 z-10 shadow-[0_10px_30px_rgba(0,0,0,0.5)]">
                                    <div className="flex items-center justify-between">
                                        <div className="text-[10px] font-bold text-[#0EA5E9] uppercase tracking-widest">Detection</div>
                                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                    </div>
                                    <div className="text-base font-bold text-white">Flood Risk Anomaly</div>
                                    <div className="flex items-center gap-2">
                                        <div className="h-1.5 flex-1 bg-gray-800 rounded-full overflow-hidden">
                                            <div className="h-full bg-gradient-to-r from-red-500 to-[#10B981] w-[80%]" />
                                        </div>
                                        <span className="text-xs font-mono text-white">HI</span>
                                    </div>
                                </div>

                                <div className="absolute bottom-8 right-8 bg-[#0F0F0F]/80 backdrop-blur-xl p-5 rounded-2xl border border-white/10 flex flex-col gap-2 w-56 z-10 text-right shadow-[0_10px_30px_rgba(0,0,0,0.5)]">
                                    <div className="text-[10px] font-bold text-[#10B981] uppercase tracking-widest flex items-center justify-end gap-2">
                                        <Target size={12} /> Model Confidence
                                    </div>
                                    <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-l from-white to-gray-400">94.2%</div>
                                    <div className="text-xs text-gray-400 font-medium tracking-wide">Sentinel-1 Validated</div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>

            </section>

            {/* ═══ C. POP-UP BENTO BOX GRID ═══ */}
            <section id="solutions" className="py-32 px-6 max-w-7xl mx-auto relative z-10">
                <motion.div
                    initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                    className="mb-20 text-center"
                >
                    <h2 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 text-white">
                        Powerful earth observation,<br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#0EA5E9] to-[#10B981]">beautifully simplified.</span>
                    </h2>
                    <p className="text-xl text-gray-400 max-w-2xl mx-auto font-medium">Purpose-built intelligence models to assess risk instantly, turning complex telemetry into decisive action.</p>
                </motion.div>

                <motion.div
                    initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={staggerContainer}
                    className="grid md:grid-cols-3 gap-8"
                >
                    {/* Card 1 */}
                    <motion.div
                        variants={fadeInUp}
                        whileHover={{ scale: 1.02, y: -5 }}
                        className="group relative h-[420px] bg-[#0A0A0A]/60 backdrop-blur-xl rounded-[32px] p-8 shadow-[0_0_0_1px_rgba(255,255,255,0.05),0_10px_40px_rgba(0,0,0,0.5)] hover:shadow-[0_0_0_1px_rgba(16,185,129,0.3),0_20px_50px_rgba(16,185,129,0.15)] transition-all duration-500 flex flex-col overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 w-32 h-32 bg-[#10B981]/10 rounded-full blur-[50px] group-hover:bg-[#10B981]/20 transition-colors" />

                        <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-8 shrink-0 relative z-10 transition-transform group-hover:scale-110 group-hover:rotate-3 duration-500">
                            <MapIcon className="text-[#10B981]" size={28} />
                        </div>
                        <h3 className="text-2xl font-bold mb-4 text-white relative z-10 tracking-tight">Automated Flood Detection</h3>
                        <p className="text-gray-400 font-medium leading-relaxed relative z-10 text-lg">
                            Anticipate climate shifts with predictive satellite intelligence. COSMEON detects exact risk zones in near real-time.
                        </p>

                        {/* Hidden Pop-up layer on hover */}
                        <motion.div
                            className="absolute bottom-0 left-0 right-0 h-[55%] bg-gradient-to-t from-[#050505] via-[#050505]/95 to-transparent p-8 flex flex-col justify-end translate-y-[110%] group-hover:translate-y-0 transition-transform duration-500 ease-out z-20 border-t border-white/5"
                        >
                            <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 flex items-center justify-between">
                                <span className="text-sm font-bold tracking-widest text-[#10B981] uppercase">SAR Analysis Active</span>
                                <Activity size={20} className="text-[#10B981] animate-pulse" />
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Card 2 */}
                    <motion.div
                        variants={fadeInUp}
                        whileHover={{ scale: 1.02, y: -5 }}
                        className="group relative h-[420px] bg-[#0A0A0A]/60 backdrop-blur-xl rounded-[32px] p-8 shadow-[0_0_0_1px_rgba(255,255,255,0.05),0_10px_40px_rgba(0,0,0,0.5)] hover:shadow-[0_0_0_1px_rgba(14,165,233,0.3),0_20px_50px_rgba(14,165,233,0.15)] transition-all duration-500 flex flex-col overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 w-32 h-32 bg-[#0EA5E9]/10 rounded-full blur-[50px] group-hover:bg-[#0EA5E9]/20 transition-colors" />

                        <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-8 shrink-0 relative z-10 transition-transform group-hover:scale-110 group-hover:-rotate-3 duration-500">
                            <BarChart3 className="text-[#0EA5E9]" size={28} />
                        </div>
                        <h3 className="text-2xl font-bold mb-4 text-white relative z-10 tracking-tight">Infrastructure Exposure</h3>
                        <p className="text-gray-400 font-medium leading-relaxed relative z-10 text-lg">
                            Accelerate resilience testing. Assess land readiness and urban vulnerability using proprietary machine learning models.
                        </p>

                        {/* Hidden Pop-up layer on hover */}
                        <motion.div
                            className="absolute bottom-0 left-0 right-0 h-[55%] bg-gradient-to-t from-[#050505] via-[#050505]/95 to-transparent p-8 flex flex-col justify-end translate-y-[110%] group-hover:translate-y-0 transition-transform duration-500 ease-out z-20 border-t border-white/5"
                        >
                            <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 flex items-center justify-between">
                                <span className="text-sm font-bold tracking-widest text-[#0EA5E9] uppercase">Asset Valuation API</span>
                                <Database size={20} className="text-[#0EA5E9]" />
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Card 3 */}
                    <motion.div
                        variants={fadeInUp}
                        whileHover={{ scale: 1.02, y: -5 }}
                        className="group relative h-[420px] bg-[#0A0A0A]/60 backdrop-blur-xl rounded-[32px] p-8 shadow-[0_0_0_1px_rgba(255,255,255,0.05),0_10px_40px_rgba(0,0,0,0.5)] hover:shadow-[0_0_0_1px_rgba(167,139,250,0.3),0_20px_50px_rgba(167,139,250,0.15)] transition-all duration-500 flex flex-col overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-[50px] group-hover:bg-purple-500/20 transition-colors" />

                        <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-8 shrink-0 relative z-10 transition-transform group-hover:scale-110 group-hover:rotate-3 duration-500">
                            <Layers className="text-purple-400" size={28} />
                        </div>
                        <h3 className="text-2xl font-bold mb-4 text-white relative z-10 tracking-tight">Change Detection</h3>
                        <p className="text-gray-400 font-medium leading-relaxed relative z-10 text-lg">
                            Track environmental evolution over time. Conduct deep historical comparisons with API-ready outputs.
                        </p>

                        {/* Hidden Pop-up layer on hover */}
                        <motion.div
                            className="absolute bottom-0 left-0 right-0 h-[55%] bg-gradient-to-t from-[#050505] via-[#050505]/95 to-transparent p-8 flex flex-col justify-end translate-y-[110%] group-hover:translate-y-0 transition-transform duration-500 ease-out z-20 border-t border-white/5"
                        >
                            <div className="bg-white/5 backdrop-blur-md rounded-xl p-4 border border-white/10 flex items-center justify-between">
                                <span className="text-sm font-bold tracking-widest text-purple-400 uppercase">Time-Series Data</span>
                                <Layers size={20} className="text-purple-400" />
                            </div>
                        </motion.div>
                    </motion.div>
                </motion.div>
            </section>

            {/* ═══ D. FULL-WIDTH DARK MODE TRANSITION (PIPELINE) ═══ */}
            <section id="pipeline" className="py-32 bg-[#050505] text-white relative border-y border-white/5 z-10">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-white/[0.03] to-transparent pointer-events-none" />

                <div className="max-w-7xl mx-auto px-6 relative z-10">
                    <motion.div
                        initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                        className="text-center mb-24"
                    >
                        <h2 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 text-white">The Engine Architecture</h2>
                        <p className="text-xl text-transparent bg-clip-text bg-gradient-to-r from-[#10B981] to-[#0EA5E9] max-w-2xl mx-auto font-bold tracking-wide uppercase">From raw orbit telemetry to boardroom intelligence.</p>
                    </motion.div>

                    <div className="relative pt-10">
                        {/* Connecting Line */}
                        <motion.div
                            initial={{ height: 0 }} whileInView={{ height: "100%" }} viewport={{ once: true, margin: "-100px" }} transition={{ duration: 1.5, ease: "easeInOut" }}
                            className="absolute left-[39px] md:left-[50%] top-0 w-1 bg-gradient-to-b from-[#10B981] via-[#0EA5E9] to-transparent -translate-x-1/2 hidden md:block rounded-full shadow-[0_0_15px_rgba(16,185,129,0.5)]"
                        />

                        <motion.div
                            initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={staggerContainer}
                            className="flex flex-col gap-16 pb-10"
                        >
                            {/* Step 1: Ingest */}
                            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center gap-12 relative z-10 w-full max-w-5xl mx-auto">
                                <div className="md:w-1/2 flex justify-end md:pr-16 w-full">
                                    <div className="bg-[#0A0A0A] p-10 rounded-[32px] border border-white/10 w-full text-left shadow-[0_20px_50px_rgba(0,0,0,0.5)] hover:border-[#10B981]/50 transition-colors group relative overflow-hidden">
                                        <div className="absolute top-0 right-0 w-24 h-24 bg-[#10B981]/10 rounded-full blur-[40px]" />
                                        <div className="w-12 h-12 rounded-xl bg-[#10B981]/20 flex items-center justify-center mb-6 text-[#10B981]">01</div>
                                        <h3 className="text-3xl font-bold mb-4 text-white">Ingest</h3>
                                        <p className="text-gray-400 text-base font-medium leading-relaxed">Satellites capture multi-spectral and SAR data across the globe. We instantly pull new captures into the pipeline via direct API feeds.</p>
                                    </div>
                                </div>
                                <div className="absolute left-[31px] md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-[#10B981] shadow-[0_0_30px_#10B981] hidden md:flex items-center justify-center border-4 border-[#050505]" />

                                {/* Imagery Right for Step 1 */}
                                <div className="md:w-1/2 w-full hidden md:block">
                                    <div className="bg-[#0A0A0A] border border-white/10 rounded-[32px] h-[250px] w-full p-6 flex flex-col gap-4 shadow-2xl relative overflow-hidden group">
                                        <div className="absolute inset-0 bg-gradient-to-br from-[#10B981]/5 to-transparent z-0" />
                                        <div className="relative z-10 flex items-center justify-between border-b border-white/10 pb-4">
                                            <span className="font-mono text-sm text-[#10B981]">API FEED STATUS</span>
                                            <span className="flex h-3 w-3 relative"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span><span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span></span>
                                        </div>
                                        <div className="relative z-10 flex-1 flex flex-col gap-3 font-mono text-xs text-gray-500 pt-2">
                                            <div className="flex gap-4"><span>&gt;_ CONNECTING:</span> <span className="text-white">SENTINEL-1 SAR</span></div>
                                            <div className="flex gap-4"><span>&gt;_ DOWNLOAD_RATE:</span> <span className="text-[#0EA5E9]">4.2 GB/s</span></div>
                                            <div className="flex gap-4"><span>&gt;_ SYNCING REGION:</span> <span className="text-white">OR-WEST-2</span></div>
                                            <div className="flex gap-4"><span>&gt;_ PACKETS RCVD:</span> <span className="text-[#10B981]">98,432.00</span></div>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Step 2: Process */}
                            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center gap-12 relative z-10 w-full max-w-5xl mx-auto">
                                {/* Imagery Left for Step 2 */}
                                <div className="md:w-1/2 w-full hidden md:block">
                                    <div className="bg-[#0A0A0A] border border-white/10 rounded-[32px] h-[250px] w-full p-6 flex items-center justify-center shadow-2xl relative overflow-hidden group">
                                        <div className="absolute inset-0 bg-gradient-to-bl from-[#0EA5E9]/5 to-transparent z-0" />
                                        <Zap size={80} className="text-[#0EA5E9]/20 group-hover:text-[#0EA5E9]/50 transition-colors animate-pulse relative z-10" />
                                        <div className="absolute bottom-6 font-mono text-sm text-[#0EA5E9] tracking-widest relative z-10">Neural Net Processing</div>
                                    </div>
                                </div>
                                <div className="absolute left-[31px] md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-[#0EA5E9] shadow-[0_0_30px_#0EA5E9] hidden md:flex items-center justify-center border-4 border-[#050505]" />
                                <div className="md:w-1/2 flex justify-start md:pl-16 w-full">
                                    <div className="bg-[#0A0A0A] p-10 rounded-[32px] border border-white/10 w-full text-left shadow-[0_20px_50px_rgba(0,0,0,0.5)] hover:border-[#0EA5E9]/50 transition-colors group relative overflow-hidden">
                                        <div className="absolute top-0 right-0 w-24 h-24 bg-[#0EA5E9]/10 rounded-full blur-[40px]" />
                                        <div className="w-12 h-12 rounded-xl bg-[#0EA5E9]/20 flex items-center justify-center mb-6 text-[#0EA5E9]">02</div>
                                        <h3 className="text-3xl font-bold mb-4 text-white">Process</h3>
                                        <p className="text-gray-400 text-base font-medium leading-relaxed">Deep learning models denoise, classify, and extract geospatial anomalies from raw pixels in milliseconds across scalable cloud architecture.</p>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Step 3: Decide */}
                            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center gap-12 relative z-10 w-full max-w-5xl mx-auto">
                                <div className="md:w-1/2 flex justify-end md:pr-16 w-full">
                                    <div className="bg-[#0A0A0A] p-10 rounded-[32px] border border-white/10 w-full text-left shadow-[0_20px_50px_rgba(0,0,0,0.5)] hover:border-purple-500/50 transition-colors group relative overflow-hidden">
                                        <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/10 rounded-full blur-[40px]" />
                                        <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center mb-6 text-purple-400">03</div>
                                        <h3 className="text-3xl font-bold mb-4 text-white">Decide</h3>
                                        <p className="text-gray-400 text-base font-medium leading-relaxed">Clear, structured outputs and risk scores are delivered to administrators via the COSMEON dashboard, integrating directly into business ops.</p>
                                    </div>
                                </div>
                                <div className="absolute left-[31px] md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-purple-500 shadow-[0_0_30px_rgba(168,85,247,0.8)] hidden md:flex items-center justify-center border-4 border-[#050505]" />

                                {/* Imagery Right for Step 3 */}
                                <div className="md:w-1/2 w-full hidden md:block">
                                    <div className="bg-[#0A0A0A] border border-white/10 rounded-[32px] h-[250px] w-full p-6 flex flex-col gap-4 shadow-2xl relative overflow-hidden group">
                                        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent z-0" />
                                        <div className="relative z-10 w-full h-full flex items-center justify-center">
                                            <div className="flex -space-x-4">
                                                <div className="w-20 h-20 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md flex items-center justify-center z-30 shadow-2xl"><Layers className="text-white" size={30} /></div>
                                                <div className="w-20 h-20 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md flex items-center justify-center z-20 scale-90 translate-x-4"><Activity className="text-gray-400" size={30} /></div>
                                                <div className="w-20 h-20 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md flex items-center justify-center z-10 scale-75 translate-x-8"><Database className="text-gray-600" size={30} /></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* ═══ E. FINAL CTA ═══ */}
            <section className="py-40 px-6 bg-[#030303] relative z-10">
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#10B981]/10 via-transparent to-transparent pointer-events-none" />

                <motion.div
                    initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                    className="max-w-4xl mx-auto text-center relative z-10"
                >
                    <h2 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 text-white">Ready to see COSMEON in Action?</h2>

                    <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto">Skip the guesswork. Get real-time, high-fidelity climate risk analysis configured for your exact coordinates today.</p>

                    <motion.div
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="inline-block"
                    >
                        <Link
                            href="https://cosmeon.onrender.com/engine"
                            className="inline-flex items-center justify-center px-16 py-6 bg-gradient-to-r from-[#10B981] to-[#0EA5E9] hover:from-[#059669] hover:to-[#0284C7] text-white text-2xl font-black tracking-wide uppercase rounded-full transition-all shadow-[0_0_40px_rgba(16,185,129,0.5)] hover:shadow-[0_0_60px_rgba(14,165,233,0.7)] hover:ring-4 hover:ring-white/20"
                        >
                            Launch Engine
                        </Link>
                    </motion.div>
                </motion.div>
            </section>

            {/* Footer */}
            <footer className="bg-[#050505] border-t border-white/10 py-16 px-6 relative z-10">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
                    <div className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-full bg-gradient-to-tr from-[#10B981] to-[#0EA5E9]" />
                        <span className="font-black tracking-widest text-xl text-white">COSMEON</span>
                    </div>

                    <div className="flex gap-10 text-sm font-bold tracking-widest uppercase text-gray-500">
                        <a href="#platform" className="hover:text-white transition-colors">Platform</a>
                        <a href="#solutions" className="hover:text-white transition-colors">Solutions</a>
                        <a href="#" className="hover:text-white transition-colors">Legal</a>
                    </div>

                    <p className="text-sm text-gray-600 font-medium tracking-wide">
                        &copy; {new Date().getFullYear()} COSMEON Earth Intelligence.
                    </p>
                </div>
            </footer>
        </div>
    );
}
