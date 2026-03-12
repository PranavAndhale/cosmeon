"use client";

import { motion, Variants } from "framer-motion";
import { ChevronRight, ShieldCheck, Activity, Map as MapIcon, Layers, BarChart3, Database, Globe, ArrowRight } from "lucide-react";
import Link from "next/link";

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
        <div className="min-h-screen bg-[#F7F7F5] text-[#1A1A1A] font-sans selection:bg-[#10B981]/20 overflow-x-hidden">

            {/* ═══ A. CLEAN MINIMALIST NAVIGATION ═══ */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 transition-all">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-[#1A1A1A]" />
                        <span className="text-xl font-bold tracking-tight text-[#1A1A1A]">COSMEON</span>
                    </Link>

                    {/* Center Links */}
                    <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-600">
                        <a href="#platform" className="hover:text-[#10B981] transition-colors">Platform</a>
                        <a href="#solutions" className="hover:text-[#10B981] transition-colors">Solutions</a>
                        <a href="#pipeline" className="hover:text-[#10B981] transition-colors">Technology</a>
                    </div>

                    {/* CTA */}
                    <Link
                        href="https://cosmeon.onrender.com/engine"
                        className="px-6 py-2.5 bg-[#10B981] hover:bg-[#0EA5E9] text-white font-semibold rounded-full text-sm transition-colors shadow-sm"
                    >
                        Launch Engine
                    </Link>
                </div>
            </nav>

            {/* ═══ B. HERO SECTION ═══ */}
            <section className="pt-40 pb-20 px-6 max-w-7xl mx-auto min-h-[90vh] flex flex-col md:flex-row items-center gap-16">

                {/* Left: Typography */}
                <motion.div
                    initial="hidden" animate="visible" variants={staggerContainer}
                    className="flex-1 flex flex-col items-start"
                >
                    <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#10B981]/10 border border-[#10B981]/20 mb-8 text-sm text-[#10B981] font-medium">
                        <Activity size={14} />
                        <span>Intelligence Engine v2.0 Live</span>
                    </motion.div>

                    <motion.h1 variants={fadeInUp} className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.05] mb-6 text-[#1A1A1A]">
                        Clarity from Space for Climate Insights on Earth.
                    </motion.h1>

                    <motion.p variants={fadeInUp} className="text-lg md:text-xl text-gray-500 max-w-xl mb-10 leading-relaxed font-medium">
                        Transforming raw satellite data into actionable intelligence. Automated detection of flood events and environmental risk.
                    </motion.p>

                    <motion.div variants={fadeInUp}>
                        <Link
                            href="https://cosmeon.onrender.com/engine"
                            className="inline-flex items-center gap-3 px-8 py-4 bg-[#10B981] hover:bg-[#0EA5E9] text-white text-lg font-bold rounded-full transition-transform hover:scale-105 shadow-md flex-shrink-0"
                        >
                            Launch Engine
                            <ArrowRight size={20} />
                        </Link>
                    </motion.div>
                </motion.div>

                {/* Right: Floating Imagery */}
                <motion.div
                    initial="hidden" animate="visible" variants={fadeInUp}
                    className="flex-1 w-full relative h-[400px] md:h-[600px] rounded-3xl overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.08)] bg-white border border-gray-100 hidden md:block"
                >
                    <motion.div
                        variants={floatAnim}
                        className="absolute inset-0 flex items-center justify-center p-8 h-full"
                    >
                        {/* Abstract map/data visualization mockup */}
                        <div className="w-full h-full rounded-2xl bg-[#0B3B24]/5 border border-[#0B3B24]/10 relative overflow-hidden flex flex-col">
                            <div className="h-10 border-b border-[#0B3B24]/10 flex items-center px-4 gap-2">
                                <div className="w-3 h-3 rounded-full bg-red-400" />
                                <div className="w-3 h-3 rounded-full bg-amber-400" />
                                <div className="w-3 h-3 rounded-full bg-[#10B981]" />
                            </div>
                            <div className="flex-1 relative overflow-hidden bg-gray-50">
                                <Globe size={300} strokeWidth={0.5} className="absolute -right-20 -bottom-20 text-[#10B981]/20 animate-pulse" />
                                {/* Overlay cards */}
                                <div className="absolute top-6 left-6 bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex flex-col gap-2 w-48 z-10">
                                    <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Detection</div>
                                    <div className="text-sm font-bold text-[#1A1A1A]">Flood Risk Anomaly</div>
                                    <div className="h-2 w-full bg-gray-100 rounded-full mt-2 overflow-hidden">
                                        <div className="h-full bg-[#10B981] w-[80%]" />
                                    </div>
                                </div>
                                <div className="absolute bottom-6 right-6 bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex flex-col gap-2 w-48 z-10 text-right">
                                    <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Confidence</div>
                                    <div className="text-2xl font-bold text-[#1A1A1A]">94.2%</div>
                                    <div className="text-xs text-gray-500 font-medium tracking-wide">Sentinel-1 Validated</div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>

            </section>

            {/* ═══ C. POP-UP BENTO BOX GRID ═══ */}
            <section id="solutions" className="py-32 px-6 max-w-7xl mx-auto">
                <motion.div
                    initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                    className="mb-16 text-center"
                >
                    <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4 text-[#1A1A1A]">Powerful earth observation,<br />beautifully simplified.</h2>
                    <p className="text-xl text-gray-500 max-w-2xl mx-auto font-medium">Purpose-built intelligence models to assess risk instantly.</p>
                </motion.div>

                <motion.div
                    initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={staggerContainer}
                    className="grid md:grid-cols-3 gap-6"
                >
                    {/* Card 1 */}
                    <motion.div
                        variants={fadeInUp}
                        whileHover={{ scale: 1.02, y: -5 }}
                        className="group relative h-[400px] bg-white rounded-3xl p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] hover:shadow-[0_20px_50px_rgb(0,0,0,0.1)] transition-all duration-300 border border-gray-100 flex flex-col overflow-hidden"
                    >
                        <div className="w-12 h-12 rounded-2xl bg-[#10B981]/10 flex items-center justify-center mb-6 shrink-0 relative z-10 transition-transform group-hover:scale-110 duration-300">
                            <MapIcon className="text-[#10B981]" size={24} />
                        </div>
                        <h3 className="text-2xl font-bold mb-3 text-[#1A1A1A] relative z-10">Automated Flood Detection</h3>
                        <p className="text-gray-500 font-medium leading-relaxed relative z-10">
                            Anticipate climate shifts with predictive satellite intelligence. COSMEON detects exact risk zones in near real-time.
                        </p>

                        {/* Hidden Pop-up layer on hover */}
                        <motion.div
                            className="absolute bottom-0 left-0 right-0 h-1/2 bg-gradient-to-t from-white via-white to-white/90 p-8 flex flex-col justify-end translate-y-[110%] group-hover:translate-y-0 transition-transform duration-500 ease-out z-20 shadow-[0_-20px_40px_rgba(0,0,0,0.03)]"
                        >
                            <div className="border-t border-gray-100 pt-4 flex items-center justify-between">
                                <span className="text-sm font-bold tracking-wide text-[#10B981] uppercase">SAR Analysis Active</span>
                                <Activity size={18} className="text-[#10B981] animate-pulse" />
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Card 2 */}
                    <motion.div
                        variants={fadeInUp}
                        whileHover={{ scale: 1.02, y: -5 }}
                        className="group relative h-[400px] bg-white rounded-3xl p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] hover:shadow-[0_20px_50px_rgb(0,0,0,0.1)] transition-all duration-300 border border-gray-100 flex flex-col overflow-hidden"
                    >
                        <div className="w-12 h-12 rounded-2xl bg-[#0B3B24]/10 flex items-center justify-center mb-6 shrink-0 relative z-10 transition-transform group-hover:scale-110 duration-300">
                            <BarChart3 className="text-[#0B3B24]" size={24} />
                        </div>
                        <h3 className="text-2xl font-bold mb-3 text-[#1A1A1A] relative z-10">Infrastructure Exposure</h3>
                        <p className="text-gray-500 font-medium leading-relaxed relative z-10">
                            Accelerate resilience testing. Assess land readiness and urban vulnerability using proprietary machine learning models.
                        </p>

                        {/* Hidden Pop-up layer on hover */}
                        <motion.div
                            className="absolute bottom-0 left-0 right-0 h-1/2 bg-gradient-to-t from-white via-white to-white/90 p-8 flex flex-col justify-end translate-y-[110%] group-hover:translate-y-0 transition-transform duration-500 ease-out z-20 shadow-[0_-20px_40px_rgba(0,0,0,0.03)]"
                        >
                            <div className="border-t border-gray-100 pt-4 flex items-center justify-between">
                                <span className="text-sm font-bold tracking-wide text-[#0B3B24] uppercase">Asset Valuation API</span>
                                <Database size={18} className="text-[#0B3B24]" />
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* Card 3 */}
                    <motion.div
                        variants={fadeInUp}
                        whileHover={{ scale: 1.02, y: -5 }}
                        className="group relative h-[400px] bg-white rounded-3xl p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] hover:shadow-[0_20px_50px_rgb(0,0,0,0.1)] transition-all duration-300 border border-gray-100 flex flex-col overflow-hidden"
                    >
                        <div className="w-12 h-12 rounded-2xl bg-[#1A1A1A]/10 flex items-center justify-center mb-6 shrink-0 relative z-10 transition-transform group-hover:scale-110 duration-300">
                            <Layers className="text-[#1A1A1A]" size={24} />
                        </div>
                        <h3 className="text-2xl font-bold mb-3 text-[#1A1A1A] relative z-10">Change Detection</h3>
                        <p className="text-gray-500 font-medium leading-relaxed relative z-10">
                            Track environmental evolution over time. Conduct deep historical comparisons with API-ready outputs.
                        </p>

                        {/* Hidden Pop-up layer on hover */}
                        <motion.div
                            className="absolute bottom-0 left-0 right-0 h-1/2 bg-gradient-to-t from-white via-white to-white/90 p-8 flex flex-col justify-end translate-y-[110%] group-hover:translate-y-0 transition-transform duration-500 ease-out z-20 shadow-[0_-20px_40px_rgba(0,0,0,0.03)]"
                        >
                            <div className="border-t border-gray-100 pt-4 flex items-center justify-between">
                                <span className="text-sm font-bold tracking-wide text-[#1A1A1A] uppercase">Time-Series Data</span>
                                <Layers size={18} className="text-[#1A1A1A]" />
                            </div>
                        </motion.div>
                    </motion.div>
                </motion.div>
            </section>

            {/* ═══ D. FULL-WIDTH DARK MODE TRANSITION (PIPELINE) ═══ */}
            <section id="pipeline" className="py-32 bg-[#0B3B24] text-white relative border-none">
                <div className="max-w-7xl mx-auto px-6">
                    <motion.div
                        initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                        className="text-center mb-20"
                    >
                        <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4 text-white">The Engine Architecture</h2>
                        <p className="text-xl text-[#10B981] max-w-2xl mx-auto font-medium">From raw orbit telemetry to boardroom intelligence.</p>
                    </motion.div>

                    <div className="relative pt-10">
                        {/* Connecting Line */}
                        <motion.div
                            initial={{ height: 0 }} whileInView={{ height: "100%" }} viewport={{ once: true, margin: "-100px" }} transition={{ duration: 1.5, ease: "easeInOut" }}
                            className="absolute left-[39px] md:left-[50%] top-0 w-0.5 bg-gradient-to-b from-[#10B981] via-[#10B981]/50 to-transparent -translate-x-1/2 hidden md:block"
                        />

                        <motion.div
                            initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={staggerContainer}
                            className="flex flex-col gap-12 pb-10"
                        >
                            {/* Step 1: Ingest */}
                            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center gap-8 relative z-10 w-full max-w-4xl mx-auto">
                                <div className="md:w-1/2 flex justify-end md:pr-12 w-full">
                                    <div className="bg-[#082E1C] p-8 rounded-3xl border border-[#10B981]/20 w-full text-left shadow-xl shadow-black/20">
                                        <h3 className="text-2xl font-bold mb-2">Ingest</h3>
                                        <p className="text-[#10B981]/80 text-sm font-medium leading-relaxed">Satellites capture multi-spectral and SAR data across the globe. We instantly pull new captures into the pipeline.</p>
                                    </div>
                                </div>
                                <div className="absolute left-[31px] md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-[#10B981] shadow-[0_0_20px_#10B981] hidden md:block" />
                                <div className="md:w-1/2 w-full hidden md:block" />
                            </motion.div>

                            {/* Step 2: Process */}
                            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center gap-8 relative z-10 w-full max-w-4xl mx-auto">
                                <div className="md:w-1/2 w-full hidden md:block" />
                                <div className="absolute left-[31px] md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-white shadow-[0_0_20px_white] hidden md:block" />
                                <div className="md:w-1/2 flex justify-start md:pl-12 w-full">
                                    <div className="bg-[#082E1C] p-8 rounded-3xl border border-white/20 w-full text-left shadow-xl shadow-black/20">
                                        <h3 className="text-2xl font-bold mb-2 text-white">Process</h3>
                                        <p className="text-white/70 text-sm font-medium leading-relaxed">Deep learning models denoise, classify, and extract geospatial anomalies from raw pixels in milliseconds.</p>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Step 3: Decide */}
                            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center gap-8 relative z-10 w-full max-w-4xl mx-auto">
                                <div className="md:w-1/2 flex justify-end md:pr-12 w-full">
                                    <div className="bg-[#082E1C] p-8 rounded-3xl border border-[#10B981]/20 w-full text-left border-l-8 border-l-[#10B981] shadow-xl shadow-black/20">
                                        <h3 className="text-2xl font-bold mb-2">Decide</h3>
                                        <p className="text-[#10B981]/80 text-sm font-medium leading-relaxed">Clear, structured outputs and risk scores are delivered to administrators via the COSMEON dashboard.</p>
                                    </div>
                                </div>
                                <div className="absolute left-[31px] md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-[#10B981] shadow-[0_0_20px_#10B981] hidden md:block" />
                                <div className="md:w-1/2 w-full hidden md:block" />
                            </motion.div>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* ═══ E. FINAL CTA ═══ */}
            <section className="py-40 px-6 bg-[#F7F7F5]">
                <motion.div
                    initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                    className="max-w-4xl mx-auto text-center"
                >
                    <h2 className="text-5xl md:text-7xl font-bold tracking-tight mb-8 text-[#1A1A1A]">Ready to see COSMEON in Action?</h2>

                    <motion.div
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="inline-block mt-8"
                    >
                        <Link
                            href="https://cosmeon.onrender.com/engine"
                            className="inline-flex items-center justify-center px-16 py-6 bg-[#10B981] hover:bg-[#0EA5E9] text-white text-2xl font-bold rounded-full transition-colors shadow-[0_10px_40px_rgba(16,185,129,0.3)] animate-[pulse_3s_ease-in-out_infinite]"
                        >
                            Launch Engine
                        </Link>
                    </motion.div>
                </motion.div>
            </section>

            {/* Footer */}
            <footer className="bg-white border-t border-gray-100 py-12 px-6">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-[#1A1A1A]" />
                        <span className="font-bold tracking-tight text-xl text-[#1A1A1A]">COSMEON</span>
                    </div>

                    <div className="flex gap-8 text-sm font-bold tracking-wide uppercase text-gray-400">
                        <a href="#platform" className="hover:text-[#10B981] transition-colors">Platform</a>
                        <a href="#solutions" className="hover:text-[#10B981] transition-colors">Solutions</a>
                        <a href="#" className="hover:text-[#10B981] transition-colors">Legal</a>
                    </div>

                    <p className="text-sm text-gray-400 font-medium">
                        &copy; {new Date().getFullYear()} COSMEON Earth Intelligence.
                    </p>
                </div>
            </footer>
        </div>
    );
}
