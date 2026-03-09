"use client";

import { motion } from "framer-motion";
import { FolderGit2, Globe2, Droplets, LeafyGreen, ShieldAlert, ChevronRight } from "lucide-react";

interface SidebarProps {
    activeOrb: string;
    setActiveOrb: (orb: string) => void;
    regions: any[];
    selectedRegion: number | null;
    setSelectedRegion: (id: number) => void;
}

export function Sidebar({ activeOrb, setActiveOrb, regions, selectedRegion, setSelectedRegion }: SidebarProps) {

    const orbs = [
        { id: "flood", name: "Flood Risk Orb", icon: Droplets, color: "var(--color-cyan)" },
        { id: "carbon", name: "Carbon Orb", icon: LeafyGreen, color: "var(--color-primary)" },
    ];

    return (
        <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute top-20 left-6 bottom-6 w-72 glass-panel rounded-2xl z-20 flex flex-col overflow-hidden select-none"
        >
            {/* Orb Selector Header */}
            <div className="p-5 border-b border-white/10 bg-black/20">
                <h2 className="text-[11px] uppercase tracking-widest text-[var(--color-text-muted)] mb-3 font-mono">Workspace Orbs</h2>
                <div className="flex flex-col gap-2">
                    {orbs.map(orb => (
                        <div
                            key={orb.id}
                            onClick={() => setActiveOrb(orb.id)}
                            className={`
                flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all duration-300 border
                ${activeOrb === orb.id ? 'bg-white/5 border-white/20 shadow-inner' : 'border-transparent hover:bg-white/5 opacity-60 hover:opacity-100'}
              `}
                        >
                            <div className="w-8 h-8 rounded-md flex items-center justify-center bg-black/40 border border-white/5" style={{ color: orb.color }}>
                                <orb.icon size={16} />
                            </div>
                            <span className="font-mono text-sm font-medium">{orb.name}</span>
                            {activeOrb === orb.id && <div className="ml-auto w-1.5 h-1.5 rounded-full" style={{ backgroundColor: orb.color, boxShadow: `0 0 8px ${orb.color}` }} />}
                        </div>
                    ))}
                </div>
            </div>

            {/* Project Navigator (Regions) */}
            <div className="p-5 flex-grow overflow-y-auto">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-[11px] uppercase tracking-widest text-[var(--color-text-muted)] font-mono">Project Navigator</h2>
                    <FolderGit2 size={14} className="text-white/40" />
                </div>

                <div className="flex flex-col gap-1">
                    {regions.map(region => (
                        <div
                            key={region.id}
                            onClick={() => setSelectedRegion(region.id)}
                            className={`
                 group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors border-l-2
                 ${selectedRegion === region.id ? 'bg-[var(--color-primary)]/10 border-[var(--color-primary)] text-white' : 'border-transparent hover:bg-white/5 text-white/60 hover:text-white'}
               `}
                        >
                            <div className="flex items-center gap-3">
                                <Globe2 size={16} className={selectedRegion === region.id ? 'text-[var(--color-primary)]' : 'text-white/40'} />
                                <span className="text-sm font-medium">{region.name}</span>
                            </div>
                            <ChevronRight size={14} className={`transition-transform ${selectedRegion === region.id ? 'text-[var(--color-primary)] translate-x-1' : 'opacity-0 group-hover:opacity-100'}`} />
                        </div>
                    ))}
                </div>
            </div>

            {/* MCA Tool Integration */}
            <div className="p-5 border-t border-white/10 bg-black/20">
                <div className="glass-panel p-4 rounded-xl flex items-start gap-4 cursor-pointer hover:border-[var(--color-primary)]/50 transition-colors">
                    <div className="mt-0.5 text-[var(--color-primary)]"><ShieldAlert size={18} /></div>
                    <div>
                        <h4 className="text-sm font-bold font-mono text-white mb-1">Multi-Criteria Analysis</h4>
                        <p className="text-[11px] text-[var(--color-text-muted)] leading-tight">Run composite risk overlays on active region.</p>
                    </div>
                </div>
            </div>

        </motion.div>
    );
}
