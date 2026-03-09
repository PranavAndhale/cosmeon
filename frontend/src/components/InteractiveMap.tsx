"use client";

import { useEffect, useRef, useState } from "react";
import Map, { Marker, NavigationControl, Popup, useMap, MapRef } from "react-map-gl/maplibre";
import 'maplibre-gl/dist/maplibre-gl.css';
import { Target, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Region {
    id: number;
    name: string;
    bbox?: number[];
    riskScore?: number;
}

interface InteractiveMapProps {
    regions: Region[];
    selectedRegionId: number | null;
    onRegionSelect: (id: number) => void;
}

// Subcomponent to handle 3D camera sweeps
function CameraController({ regions, selectedRegionId }: { regions: Region[], selectedRegionId: number | null }) {
    const { current: map } = useMap();

    useEffect(() => {
        if (!map) return;

        if (selectedRegionId) {
            const region = regions.find(r => r.id === selectedRegionId);
            if (region && region.bbox && region.bbox.length === 4) {
                const lon = (region.bbox[0] + region.bbox[2]) / 2;
                const lat = (region.bbox[1] + region.bbox[3]) / 2;
                map.flyTo({
                    center: [lon, lat],
                    zoom: 7,
                    pitch: 45, // 3D tilt
                    bearing: 15,
                    duration: 2000,
                    essential: true
                });
            }
        } else {
            map.flyTo({
                center: [0, 20],
                zoom: 2,
                pitch: 0,
                bearing: 0,
                duration: 2500,
                essential: true
            });
        }
    }, [selectedRegionId, regions, map]);

    return null;
}

export default function InteractiveMap({ regions, selectedRegionId, onRegionSelect }: InteractiveMapProps) {
    const [hoverInfo, setHoverInfo] = useState<{ region: Region, x: number, y: number } | null>(null);

    return (
        <div className="absolute inset-0 z-0">
            <Map
                initialViewState={{
                    longitude: 0,
                    latitude: 20,
                    zoom: 2,
                    pitch: 0
                }}
                mapStyle={{
                    version: 8,
                    sources: {
                        'satellite': {
                            type: 'raster',
                            tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
                            tileSize: 256
                        },
                        'dark-labels': {
                            type: 'raster',
                            tiles: ['https://a.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png'],
                            tileSize: 256
                        }
                    },
                    layers: [
                        { id: 'satellite-layer', type: 'raster', source: 'satellite', paint: { 'raster-opacity': 0.8 } },
                        { id: 'labels-layer', type: 'raster', source: 'dark-labels', paint: { 'raster-opacity': 1 } }
                    ]
                }}
                interactiveLayerIds={['regions']}
                cursor={hoverInfo ? 'pointer' : 'auto'}
            >
                <CameraController regions={regions} selectedRegionId={selectedRegionId} />

                {/* Navigation Controls */}
                <div className="absolute top-4 right-4 z-10 glass-panel rounded-lg overflow-hidden border border-white/10">
                    <NavigationControl showCompass={true} showZoom={true} visualizePitch={true} />
                </div>

                {regions.map((region, idx) => {
                    let lon = idx * 10;
                    let lat = idx * 5;
                    if (region.bbox && region.bbox.length === 4) {
                        lon = (region.bbox[0] + region.bbox[2]) / 2;
                        lat = (region.bbox[1] + region.bbox[3]) / 2;
                    }

                    const isSelected = selectedRegionId === region.id;
                    const score = region.riskScore || Math.floor(Math.random() * 100);

                    return (
                        <Marker
                            key={region.id}
                            longitude={lon}
                            latitude={lat}
                            anchor="center"
                            onClick={e => {
                                e.originalEvent.stopPropagation();
                                onRegionSelect(region.id);
                            }}
                        >
                            <div
                                className="relative group cursor-pointer"
                                onMouseEnter={() => setHoverInfo({ region: { ...region, riskScore: score }, x: lon, y: lat })}
                                onMouseLeave={() => setHoverInfo(null)}
                            >
                                {/* Outer Pulse Ring */}
                                <div className="absolute -inset-4 rounded-full border border-[var(--color-primary)] opacity-30 animate-[ping_2.5s_cubic-bezier(0,0,0.2,1)_infinite]" />

                                {/* Core Target Icon */}
                                <div className={`
                    w-8 h-8 rounded-full flex items-center justify-center
                    transition-all duration-300
                    ${isSelected ? 'bg-[var(--color-primary)] text-black shadow-[0_0_20px_var(--color-primary)]' : 'glass-panel text-[var(--color-primary)] hover:border-[var(--color-primary)]'}
                `}>
                                    <Target size={16} className={isSelected ? 'animate-[spin_4s_linear_infinite]' : ''} />
                                </div>
                            </div>
                        </Marker>
                    );
                })}

                {/* Dynamic Tooltip on Hover */}
                <AnimatePresence>
                    {hoverInfo && (
                        <Popup
                            longitude={hoverInfo.x}
                            latitude={hoverInfo.y}
                            closeButton={false}
                            closeOnClick={false}
                            anchor="bottom"
                            offset={24}
                            className="z-50"
                        >
                            <motion.div
                                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className="flex flex-col gap-2 min-w-[200px]"
                            >
                                <div className="text-[10px] uppercase tracking-widest text-[var(--color-text-muted)] flex items-center gap-1"><Activity size={10} color="var(--color-primary)" /> Live Intelligence</div>
                                <div className="font-mono text-lg font-bold text-white tracking-tight">{hoverInfo.region.name}</div>

                                <div className="mt-2 pt-2 border-t border-white/10 flex justify-between items-end">
                                    <div className="flex flex-col">
                                        <span className="text-[10px] text-white/50 uppercase">Composite Risk</span>
                                        <span className="font-mono font-bold text-[var(--color-primary)] text-xl">{hoverInfo.region.riskScore}</span>
                                    </div>
                                    <div className="w-16 h-1 rounded-full bg-white/10 overflow-hidden mb-1.5">
                                        <div className="h-full bg-[var(--color-primary)]" style={{ width: `${hoverInfo.region.riskScore}%` }} />
                                    </div>
                                </div>
                            </motion.div>
                        </Popup>
                    )}
                </AnimatePresence>

            </Map>
        </div>
    );
}
