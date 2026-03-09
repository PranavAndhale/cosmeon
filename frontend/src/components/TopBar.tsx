"use client";

import { Activity, Bell, Search, LayoutTemplate, Layers } from "lucide-react";

export function TopBar() {
    return (
        <div className="absolute top-4 left-6 right-20 h-14 z-20 flex items-center justify-between pointer-events-none">

            {/* Brand & Search */}
            <div className="flex items-center gap-6 pointer-events-auto">
                <div className="flex flex-col">
                    <h1 className="text-xl font-mono font-bold tracking-tight text-white flex items-center gap-2">
                        <LayoutTemplate size={20} className="text-[var(--color-primary)]" />
                        COSMEON<span className="text-[var(--color-primary)]">.io</span>
                    </h1>
                    <span className="text-[9px] uppercase tracking-[0.2em] text-[var(--color-text-muted)]">Earth Intel Engine</span>
                </div>

                <div className="glass-panel h-11 px-4 rounded-xl flex items-center gap-3 w-80 ml-6">
                    <Search size={16} className="text-white/40" />
                    <input
                        type="text"
                        placeholder="Search coordinates or assets..."
                        className="bg-transparent border-none outline-none text-sm text-white placeholder:text-white/30 w-full font-mono"
                    />
                </div>
            </div>

            {/* Global Actions */}
            <div className="flex items-center gap-3 pointer-events-auto">
                <div className="glass-panel h-11 px-4 rounded-xl flex items-center gap-4 text-sm font-mono border-white/10">
                    <div className="flex items-center gap-2 text-white/70 hover:text-white cursor-pointer transition-colors">
                        <Layers size={16} className="text-[var(--color-cyan)]" /> API Access
                    </div>
                    <div className="w-px h-4 bg-white/10" />
                    <div className="flex items-center gap-2 cursor-pointer transition-colors">
                        <div className="w-2 h-2 rounded-full bg-[var(--color-primary)] shadow-[0_0_8px_var(--color-primary)]" />
                        <span>System Live</span>
                    </div>
                </div>

                <button className="glass-panel w-11 h-11 rounded-xl flex items-center justify-center text-white/70 hover:text-white hover:border-white/30 transition-all">
                    <Bell size={18} />
                </button>

                <button className="glass-panel w-11 h-11 rounded-xl flex items-center justify-center font-mono font-bold text-white border-[var(--color-primary)]/30 hover:border-[var(--color-primary)] transition-all overflow-hidden relative group">
                    <div className="absolute inset-0 bg-[var(--color-primary)]/10 group-hover:bg-[var(--color-primary)]/20 transition-all" />
                    PA
                </button>
            </div>

        </div>
    );
}
