"use client";

import { motion, Variants } from "framer-motion";
import { ChevronRight, Satellite, ShieldCheck, Activity, Globe, Map as MapIcon, Layers, BarChart3, Database } from "lucide-react";
import Link from "next/link";

// ─── Animation Variants ───
const fadeInUp: Variants = {
    hidden: { opacity: 0, y: 40 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
};

const staggerContainer: Variants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.2 }
    }
};

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-[#0B0E11] text-white font-sans selection:bg-[#20E251]/30 overflow-x-hidden">

            {/* ═══ A. STICKY NAVIGATION BAR ═══ */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0B0E11]/80 backdrop-blur-md border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    {/* Logo */}
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-[#20E251] shadow-[0_0_12px_#20E251]" />
                        <span className="text-xl font-bold tracking-tight text-white">COSMEON</span>
                    </div>

                    {/* Center Links (Desktop) */}
                    <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-400">
                        <a href="#platform" className="hover:text-white transition-colors">Platform</a>
                        <a href="#solutions" className="hover:text-white transition-colors">Solutions</a>
                        <a href="#api" className="hover:text-white transition-colors">API</a>
                        <a href="#docs" className="hover:text-white transition-colors">Docs</a>
                    </div>

                    {/* CTA */}
                    <Link
                        href="https://cosmeon.onrender.com"
                        className="group px-6 py-2.5 bg-[#20E251] hover:bg-[#34f063] text-black font-semibold rounded-full text-sm transition-all flex items-center gap-2"
                    >
                        Launch Engine
                        <ChevronRight size={16} className="group-hover:translate-x-1 transition-transform" />
                    </Link>
                </div>
            </nav>

            {/* ═══ B. HERO SECTION ═══ */}
            <section className="relative pt-40 pb-20 px-6 flex flex-col items-center justify-center min-h-[90vh] text-center overflow-hidden">
                {/* Subtle background glow */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-[#20E251]/5 rounded-full blur-[120px] pointer-events-none" />

                <motion.div
                    initial="hidden" animate="visible" variants={staggerContainer}
                    className="relative z-10 max-w-4xl mx-auto flex flex-col items-center"
                >
                    <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 mb-8 text-sm text-gray-300">
                        <Activity size={14} className="text-[#20E251]" />
                        <span>Earth Intelligence Engine v2.0 Live</span>
                    </motion.div>

                    <motion.h1 variants={fadeInUp} className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1] mb-6">
                        Clarity from Space<br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-500">for Climate Insights on Earth.</span>
                    </motion.h1>

                    <motion.p variants={fadeInUp} className="text-lg md:text-xl text-gray-400 max-w-2xl mb-10 leading-relaxed">
                        COSMEON transforms raw satellite data into actionable intelligence. From detecting flood events to monitoring critical infrastructure in near real-time, we provide the ground-truth intelligence you need to act.
                    </motion.p>

                    <motion.div variants={fadeInUp}>
                        <Link
                            href="https://cosmeon.onrender.com"
                            className="inline-flex items-center gap-3 px-8 py-4 bg-[#20E251] hover:bg-[#34f063] text-black text-lg font-semibold rounded-full transition-transform hover:scale-105"
                        >
                            Launch Engine
                            <ChevronRight size={20} />
                        </Link>
                    </motion.div>
                </motion.div>

                {/* Visual Placeholder (Globe/Video) */}
                <motion.div
                    initial={{ opacity: 0, y: 60 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6, duration: 1 }}
                    className="relative z-10 w-full max-w-5xl mt-20 h-[400px] md:h-[500px] rounded-2xl border border-white/10 bg-gradient-to-b from-white/5 to-transparent overflow-hidden group shadow-2xl"
                >
                    {/* Inner green glow */}
                    <div className="absolute inset-0 shadow-[inset_0_0_100px_rgba(32,226,81,0.05)] pointer-events-none" />

                    {/* Abstract representation of the interface */}
                    <div className="absolute inset-0 flex items-center justify-center opacity-40 group-hover:opacity-70 transition-opacity duration-700">
                        <Globe size={180} strokeWidth={0.5} className="text-[#20E251] animate-[spin_60s_linear_infinite]" />
                    </div>
                    <div className="absolute inset-0 bg-[url('https://transparenttextures.com/patterns/cubes.png')] opacity-10 mix-blend-overlay" />

                    <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end">
                        <div className="flex gap-2">
                            <div className="w-2 h-2 rounded-full bg-[#20E251] animate-pulse" />
                            <span className="text-xs font-mono text-gray-400">LIVE FEED: SENTINEL-1 / SENTINEL-2 / LANDSAT</span>
                        </div>
                        <div className="text-xs font-mono text-gray-500 hidden md:block">LAT 26.0 / LON 85.5</div>
                    </div>
                </motion.div>
            </section>

            {/* ═══ C. THE "BEFORE & AFTER" SECTION ═══ */}
            <section id="platform" className="py-32 px-6 bg-[#0B0E11]">
                <div className="max-w-5xl mx-auto">
                    <motion.div
                        initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
                        className="text-center mb-20"
                    >
                        <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4">From manual guesswork to<br />automated clarity.</h2>
                        <p className="text-xl text-gray-400">Stop relying on fragmented data and delayed reporting.</p>
                    </motion.div>

                    <div className="grid md:grid-cols-2 gap-8 relative">
                        {/* Divider line for desktop */}
                        <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-px bg-white/10 -translate-x-1/2" />

                        {/* Left Column: Before */}
                        <motion.div
                            initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer}
                            className="flex flex-col gap-6 p-8 rounded-2xl bg-white/[0.02] border border-white/5"
                        >
                            <h3 className="text-xl font-semibold text-gray-300 mb-2">Before COSMEON</h3>

                            {[
                                "Fragmented data from multiple specialized sources.",
                                "Expensive mistakes caused by delayed information.",
                                "Manual image processing that can't keep pace.",
                                "Uncertainty in fast-moving climate events."
                            ].map((text, i) => (
                                <motion.div key={i} variants={fadeInUp} className="flex items-start gap-4">
                                    <div className="w-6 h-6 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center shrink-0 mt-0.5">
                                        <span className="text-red-400 text-sm">⨯</span>
                                    </div>
                                    <p className="text-gray-400 leading-relaxed">{text}</p>
                                </motion.div>
                            ))}
                        </motion.div>

                        {/* Right Column: After */}
                        <motion.div
                            initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer}
                            className="flex flex-col gap-6 p-8 rounded-2xl bg-[#20E251]/5 border border-[#20E251]/20 shadow-[0_0_30px_rgba(32,226,81,0.03)]"
                        >
                            <h3 className="text-xl font-semibold text-white mb-2 flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-[#20E251] shadow-[0_0_8px_#20E251]" />
                                After COSMEON
                            </h3>

                            {[
                                "Unified intelligence platform, all in one place.",
                                "Automated analysis operating at market speed.",
                                "Clear signals across flood and infrastructure risks.",
                                "Absolute confidence to act on verified ground truth."
                            ].map((text, i) => (
                                <motion.div key={i} variants={fadeInUp} className="flex items-start gap-4">
                                    <div className="w-6 h-6 rounded-full bg-[#20E251]/20 border border-[#20E251]/40 flex items-center justify-center shrink-0 mt-0.5">
                                        <ShieldCheck size={14} className="text-[#20E251]" />
                                    </div>
                                    <p className="text-gray-200 leading-relaxed">{text}</p>
                                </motion.div>
                            ))}
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* ═══ D. SOLUTIONS GRID ═══ */}
            <section id="solutions" className="py-32 px-6 border-t border-white/5 relative overflow-hidden">
                {/* Subtle background element */}
                <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-500/5 rounded-full blur-[100px] pointer-events-none" />

                <div className="max-w-7xl mx-auto">
                    <motion.div
                        initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeInUp}
                        className="mb-16"
                    >
                        <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4">One platform, multiple ways<br />to bring clarity.</h2>
                        <p className="text-xl text-gray-400 max-w-2xl">Purpose-built models for the most critical earth observation challenges.</p>
                    </motion.div>

                    <motion.div
                        initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-50px" }} variants={staggerContainer}
                        className="grid md:grid-cols-3 gap-6"
                    >
                        {/* Card 1 */}
                        <motion.div variants={fadeInUp} className="group p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:-translate-y-1 hover:border-white/30 transition-all duration-300">
                            <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-6">
                                <MapIcon className="text-blue-400" size={24} />
                            </div>
                            <h3 className="text-xl font-semibold mb-3">Automated Flood Detection</h3>
                            <p className="text-gray-400 leading-relaxed">
                                Anticipate climate shifts with predictive satellite intelligence. COSMEON detects anomalies and exact risk zones in near real-time, giving you a crucial head start.
                            </p>
                        </motion.div>

                        {/* Card 2 */}
                        <motion.div variants={fadeInUp} className="group p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:-translate-y-1 hover:border-[#20E251]/30 transition-all duration-300">
                            <div className="w-12 h-12 rounded-xl bg-[#20E251]/10 flex items-center justify-center mb-6">
                                <BarChart3 className="text-[#20E251]" size={24} />
                            </div>
                            <h3 className="text-xl font-semibold mb-3">Infrastructure Exposure</h3>
                            <p className="text-gray-400 leading-relaxed">
                                Accelerate resilience testing. Seamlessly assess land readiness and urban infrastructure vulnerability using our proprietary machine learning models.
                            </p>
                        </motion.div>

                        {/* Card 3 */}
                        <motion.div variants={fadeInUp} className="group p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:-translate-y-1 hover:border-purple-500/30 transition-all duration-300">
                            <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center mb-6">
                                <Layers className="text-purple-400" size={24} />
                            </div>
                            <h3 className="text-xl font-semibold mb-3">Change Detection</h3>
                            <p className="text-gray-400 leading-relaxed">
                                Track environmental evolution over time. Conduct deep historical satellite data comparisons with meticulously structured, API-ready outputs.
                            </p>
                        </motion.div>
                    </motion.div>
                </div>
            </section>

            {/* ═══ E. PROOF / CASE STUDY SECTION ═══ */}
            <section className="py-24 px-6 border-t border-white/5">
                <div className="max-w-5xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.6 }}
                        className="p-10 md:p-14 rounded-3xl bg-gradient-to-br from-[#151A22] to-[#0B0E11] border border-white/10 relative overflow-hidden group"
                    >
                        {/* Background pattern */}
                        <div className="absolute top-0 right-0 w-[300px] h-full opacity-10 mix-blend-overlay bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-white to-transparent pointer-events-none" />

                        <div className="flex items-center gap-3 mb-8">
                            <span className="px-3 py-1 text-xs font-semibold uppercase tracking-wider text-[#20E251] bg-[#20E251]/10 rounded-full">Case Study</span>
                            <span className="text-gray-500 font-mono text-sm">Flood Risk Validation</span>
                        </div>

                        <h3 className="text-3xl md:text-4xl font-bold mb-10 max-w-2xl">Where actionable intelligence makes the difference.</h3>

                        <div className="grid md:grid-cols-3 gap-8 md:gap-12">
                            <div className="flex flex-col gap-2">
                                <h4 className="text-sm font-bold uppercase tracking-widest text-gray-500">The Challenge</h4>
                                <p className="text-gray-300 leading-relaxed text-sm">Urban planners needed to assess infrastructure risk in rapidly developing zones. Without public data, they had no visibility into actual flood exposure.</p>
                            </div>
                            <div className="flex flex-col gap-2">
                                <h4 className="text-sm font-bold uppercase tracking-widest text-[#20E251]">The Solution</h4>
                                <p className="text-gray-300 leading-relaxed text-sm">Using Sentinel-1 and Sentinel-2 imagery, COSMEON&apos;s analysis revealed a <span className="text-white font-semibold">70% discrepancy</span> between projected safety and physical reality.</p>
                            </div>
                            <div className="flex flex-col gap-2">
                                <h4 className="text-sm font-bold uppercase tracking-widest text-white">The Outcome</h4>
                                <p className="text-gray-300 leading-relaxed text-sm">By identifying the risk beforehand, planners avoided massive over-valuation and corrected their models based on verified ground truth.</p>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* ═══ F. FINAL CTA & FOOTER ═══ */}
            <section className="pt-32 pb-10 px-6 border-t border-white/5 relative">
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-2xl h-px bg-gradient-to-r from-transparent via-[#20E251]/50 to-transparent" />

                <motion.div
                    initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                    className="max-w-4xl mx-auto text-center mb-32"
                >
                    <Satellite size={48} className="mx-auto text-gray-500 mb-8 opacity-50" />
                    <h2 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">Ready to see COSMEON in Action?</h2>
                    <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
                        Start processing raw imagery into structured insights today. No enterprise contracts. No multi-month implementations.
                    </p>
                    <Link
                        href="https://cosmeon.onrender.com"
                        className="inline-flex items-center justify-center px-10 py-5 bg-[#20E251] hover:bg-[#34f063] text-black text-xl font-bold rounded-full transition-transform hover:scale-105 shadow-[0_0_40px_rgba(32,226,81,0.3)]"
                    >
                        Launch Engine
                    </Link>
                </motion.div>

                {/* Minimalist Footer */}
                <footer className="max-w-7xl mx-auto border-t border-white/10 pt-10 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-gray-500" />
                        <span className="font-bold tracking-tight text-xl">COSMEON</span>
                    </div>

                    <div className="flex gap-8 text-sm text-gray-400">
                        <a href="#" className="hover:text-white transition-colors">Product</a>
                        <a href="#" className="hover:text-white transition-colors">Company</a>
                        <a href="#" className="hover:text-white transition-colors">Legal</a>
                        <a href="#api" className="hover:text-white transition-colors">API</a>
                    </div>

                    <p className="text-sm text-gray-600">
                        &copy; {new Date().getFullYear()} COSMEON Earth Intelligence.
                    </p>
                </footer>
            </section>

        </div>
    );
}
