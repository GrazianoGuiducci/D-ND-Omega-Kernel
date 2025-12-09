import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
    HeaderChatModel,
    loadHeaderChatModels,
    saveHeaderChatModels,
} from '../services/openrouter-model-config';
import { openRouterService } from '../services/openRouterService';
import { Resizer } from './Resizer';

type Props = {
    open: boolean;
    onClose: () => void;
};

// Tipo per risposta backend
type OpenRouterModelDto = {
    id: string;
    name?: string;
    provider?: string;
    pricing?: {
        input?: string;   // es. "$2.5/M"
        output?: string;
    };
    context_length?: number;
    tags?: string[];
};

const SNAP_THRESHOLD = 250;

const ModelCatalogModal: React.FC<Props> = ({ open, onClose }) => {
    // Modal & Resizing State
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
    const [modalSize, setModalSize] = useState({ width: 1100, height: 750 });
    const [sidebarWidth, setSidebarWidth] = useState(300);
    const [isResizing, setIsResizing] = useState(false);

    const isResizingSidebarRef = useRef(false);
    const isResizingModalRef = useRef(false);
    const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
    const rafRef = useRef<number | null>(null);

    // Data State
    const [byokKey, setByokKey] = useState<string>('');
    const [byokSaved, setByokSaved] = useState<boolean>(false);
    const [models, setModels] = useState<HeaderChatModel[]>([]);
    const [pinnedIds, setPinnedIds] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [sortBy, setSortBy] = useState<'name' | 'context' | 'price'>('name');
    const [filterText, setFilterText] = useState('');

    // --- RESIZING LOGIC ---
    useEffect(() => {
        const handleResize = () => setIsMobile(window.innerWidth < 768);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const startResizingSidebar = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        isResizingSidebarRef.current = true;
        dragStartRef.current = { x: e.clientX, y: e.clientY, w: sidebarWidth || 300, h: 0 };
        setIsResizing(true);
    }, [sidebarWidth]);

    const startResizingModal = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        isResizingModalRef.current = true;
        dragStartRef.current = { x: e.clientX, y: e.clientY, w: modalSize.width, h: modalSize.height };
        setIsResizing(true);
    }, [modalSize]);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizingSidebarRef.current && !isResizingModalRef.current) return;
            if (rafRef.current) cancelAnimationFrame(rafRef.current);

            rafRef.current = requestAnimationFrame(() => {
                const deltaX = e.clientX - dragStartRef.current.x;
                const deltaY = e.clientY - dragStartRef.current.y;

                if (isResizingSidebarRef.current) {
                    const rawWidth = dragStartRef.current.w + deltaX;
                    if (rawWidth < SNAP_THRESHOLD) {
                        setSidebarWidth(0);
                    } else {
                        const newWidth = Math.max(SNAP_THRESHOLD, Math.min(500, rawWidth));
                        setSidebarWidth(newWidth);
                    }
                }

                if (isResizingModalRef.current) {
                    const newWidth = Math.max(700, Math.min(window.innerWidth - 20, dragStartRef.current.w + deltaX));
                    const newHeight = Math.max(500, Math.min(window.innerHeight - 20, dragStartRef.current.h + deltaY));
                    setModalSize({ width: newWidth, height: newHeight });
                }
            });
        };

        const handleMouseUp = () => {
            if (isResizingSidebarRef.current || isResizingModalRef.current) {
                isResizingSidebarRef.current = false;
                isResizingModalRef.current = false;
                setIsResizing(false);
                if (rafRef.current) cancelAnimationFrame(rafRef.current);
            }
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, []);

    // --- DATA LOGIC ---

    // Carica BYOK da localStorage all'apertura
    useEffect(() => {
        if (!open) return;
        if (typeof window === 'undefined') return;
        try {
            const raw = window.localStorage.getItem('openrouter_api_key') || '';
            setByokKey(raw);
            setByokSaved(false);
        } catch { }
    }, [open]);

    // Carica catalogo modelli + stati pinned
    useEffect(() => {
        if (!open) return;

        const fetchModels = async () => {
            setIsLoading(true);
            try {
                const userKey = openRouterService.getUserKey();
                const headers: HeadersInit = { 'Content-Type': 'application/json' };
                if (userKey) headers['X-OpenRouter-Key'] = userKey;

                const res = await fetch('/api/v1/openrouter/models', { headers });
                if (!res.ok) throw new Error('Failed to fetch models');

                const body: OpenRouterModelDto[] = await res.json();
                const transformed: HeaderChatModel[] = body.map((m) => ({
                    id: m.id,
                    name: m.name || m.id,
                    provider: m.provider || 'Unknown',
                    ctx: m.context_length,
                    priceIn: m.pricing?.input ? `$${m.pricing.input}/M` : null,
                    priceOut: m.pricing?.output ? `$${m.pricing.output}/M` : null,
                    note: m.tags && m.tags.length ? m.tags.join(', ') : undefined,
                }));
                setModels(transformed);
            } catch (e) {
                console.warn("Failed to fetch models from backend, using fallback/pinned", e);
                const currentPinned = loadHeaderChatModels();
                setModels(currentPinned);
            } finally {
                setIsLoading(false);
            }

            const currentPinned = loadHeaderChatModels();
            setPinnedIds(new Set(currentPinned.map((m) => m.id)));
        };

        fetchModels();
    }, [open]);

    const handleSaveByok = () => {
        if (typeof window === 'undefined') return;
        try {
            const key = byokKey.trim();
            if (key.length === 0) {
                window.localStorage.removeItem('openrouter_api_key');
            } else {
                window.localStorage.setItem('openrouter_api_key', key);
            }
            setByokSaved(true);
            setTimeout(() => setByokSaved(false), 2000);
        } catch { }
    };

    const togglePin = (model: HeaderChatModel) => {
        const nextPinned = new Set(pinnedIds);
        if (nextPinned.has(model.id)) {
            nextPinned.delete(model.id);
        } else {
            nextPinned.add(model.id);
        }
        setPinnedIds(nextPinned);
        const pinnedModels = models.filter((m) => nextPinned.has(m.id));
        saveHeaderChatModels(pinnedModels);
    };

    const isPinned = (id: string) => pinnedIds.has(id);

    // Sorting and Filtering
    const getSortedAndFilteredModels = () => {
        let result = [...models];

        if (filterText) {
            const lowerFilter = filterText.toLowerCase();
            result = result.filter(m =>
                m.name.toLowerCase().includes(lowerFilter) ||
                m.provider?.toLowerCase().includes(lowerFilter) ||
                m.id.toLowerCase().includes(lowerFilter)
            );
        }

        result.sort((a, b) => {
            if (sortBy === 'name') return a.name.localeCompare(b.name);
            if (sortBy === 'context') return (b.ctx || 0) - (a.ctx || 0);
            // Rough price sort (using input price string parsing)
            if (sortBy === 'price') {
                const getPrice = (s?: string | null) => s ? parseFloat(s.replace(/[^0-9.]/g, '')) : 0;
                return getPrice(a.priceIn) - getPrice(b.priceIn);
            }
            return 0;
        });

        return result;
    };

    const displayedModels = getSortedAndFilteredModels();

    if (!open) return null;

    return (
        <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-[250] flex items-center justify-center p-0 md:p-6">
            <div
                className={`bg-[#0a0a0c] shadow-2xl border border-white/10 flex flex-col overflow-hidden relative
                    ${isMobile ? 'w-full h-full rounded-none' : 'md:rounded-2xl'}
                    ${isResizing ? 'transition-none' : 'transition-all duration-200 ease-out'}`}
                style={{
                    width: isMobile ? '100%' : `${modalSize.width}px`,
                    height: isMobile ? '100%' : `${modalSize.height}px`,
                }}
            >
                {/* Header */}
                <div className="h-12 bg-white/5 border-b border-white/10 flex items-center justify-between px-4 shrink-0">
                    <div className="flex items-center gap-2">
                        <span className="text-neon-cyan font-bold text-lg">Ω</span>
                        <span className="text-sm font-bold text-white uppercase tracking-widest">Model Catalog</span>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-white p-1 hover:bg-white/10 rounded transition-colors">
                        ✕
                    </button>
                </div>

                {/* Content */}
                <div className="flex flex-1 min-h-0">

                    {/* Sidebar (Filters & BYOK) */}
                    <div
                        className="flex-shrink-0 h-full transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] relative z-10 bg-black/40"
                        style={{ width: isMobile ? '100%' : (sidebarWidth === 0 ? '48px' : `${sidebarWidth}px`) }}
                    >
                        {sidebarWidth === 0 && !isMobile ? (
                            <div className="w-full h-full">
                                <button
                                    onClick={() => setSidebarWidth(300)}
                                    className="h-full w-full bg-black/40 border-r border-white/10 flex flex-col items-center justify-between py-8 hover:bg-white/5 hover:border-neon-cyan/50 transition-all duration-300 group"
                                >
                                    <div className="p-2 rounded-lg bg-white/5 text-neon-cyan group-hover:bg-neon-cyan group-hover:text-black transition-all mb-4">
                                        <span className="block w-5 h-5 rotate-[-90deg]">›</span>
                                    </div>
                                    <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                                        <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-slate-500 group-hover:text-neon-cyan transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                                            <span>Settings</span>
                                            <span className="w-px h-8 bg-white/10 group-hover:bg-neon-cyan/50 transition-colors"></span>
                                        </div>
                                    </div>
                                </button>
                            </div>
                        ) : (
                            <div className="h-full border-r border-white/10 flex flex-col w-full overflow-hidden">
                                <div className="p-4 border-b border-white/10">
                                    <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Configuration</h3>

                                    {/* BYOK Section */}
                                    <div className="space-y-3 mb-6">
                                        <label className="text-xs text-white font-bold">OpenRouter API Key</label>
                                        <input
                                            type="password"
                                            value={byokKey}
                                            onChange={(e) => setByokKey(e.target.value)}
                                            placeholder="sk-or-..."
                                            className="w-full bg-black border border-white/20 rounded px-3 py-2 text-xs text-white font-mono focus:border-neon-cyan focus:outline-none"
                                        />
                                        <button
                                            onClick={handleSaveByok}
                                            className="w-full bg-neon-cyan/10 hover:bg-neon-cyan/20 text-neon-cyan text-xs px-4 py-2 rounded border border-neon-cyan/30 transition-colors font-bold"
                                        >
                                            {byokSaved ? 'SAVED!' : 'SAVE KEY'}
                                        </button>
                                        <div className="flex justify-between text-[10px] text-slate-500">
                                            <a href="https://openrouter.ai/" target="_blank" rel="noreferrer" className="hover:text-neon-cyan">Get Key</a>
                                            <a href="https://openrouter.ai/docs/faq" target="_blank" rel="noreferrer" className="hover:text-neon-cyan">FAQ</a>
                                        </div>
                                    </div>

                                    <div className="border-t border-white/10 my-4"></div>

                                    {/* Filters */}
                                    <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Filters</h3>
                                    <div className="space-y-3">
                                        <div>
                                            <label className="text-[10px] text-slate-500 block mb-1">Search</label>
                                            <input
                                                type="text"
                                                value={filterText}
                                                onChange={(e) => setFilterText(e.target.value)}
                                                placeholder="Filter models..."
                                                className="w-full bg-black border border-white/20 rounded px-2 py-1.5 text-xs text-white focus:border-neon-cyan focus:outline-none"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-[10px] text-slate-500 block mb-1">Sort By</label>
                                            <select
                                                value={sortBy}
                                                onChange={(e) => setSortBy(e.target.value as any)}
                                                className="w-full bg-black border border-white/20 rounded px-2 py-1.5 text-xs text-white focus:border-neon-cyan focus:outline-none"
                                            >
                                                <option value="name">Name</option>
                                                <option value="context">Context Length</option>
                                                <option value="price">Price</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex-1 p-4 overflow-y-auto">
                                    <div className="text-[10px] text-slate-500 leading-relaxed">
                                        Pin models to make them available in the quick selector. Context length and pricing are fetched directly from OpenRouter.
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Gutter Resizer */}
                    {!isMobile && sidebarWidth > 0 && (
                        <Resizer onMouseDown={startResizingSidebar} isVisible={true} />
                    )}

                    {/* Main Area (Model List) */}
                    <div className="flex-1 flex flex-col bg-black/20 min-w-0">
                        <div className="p-4 border-b border-white/10 flex justify-between items-center bg-black/40">
                            <div className="text-xs text-slate-400">
                                Showing <span className="text-white font-bold">{displayedModels.length}</span> models
                            </div>
                            {isLoading && <span className="text-xs text-neon-cyan animate-pulse">Fetching from OpenRouter...</span>}
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 space-y-2">
                            {displayedModels.map((m) => (
                                <div key={m.id} className={`p-3 rounded border ${isPinned(m.id) ? 'border-neon-cyan/30 bg-neon-cyan/5' : 'border-white/5 bg-white/5'} flex justify-between items-center hover:border-white/20 transition-colors group`}>
                                    <div className="min-w-0 flex-1 mr-4">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="text-sm font-bold text-white truncate">{m.name}</span>
                                            {m.id.includes('free') && <span className="text-[10px] bg-green-900/50 text-green-400 px-1.5 py-0.5 rounded border border-green-500/30">FREE</span>}
                                        </div>
                                        <div className="text-xs text-slate-400 flex gap-3 items-center flex-wrap">
                                            <span className="text-slate-300">{m.provider}</span>
                                            {m.ctx && <span className="text-slate-600">•</span>}
                                            {m.ctx && <span className="text-neon-purple">{Math.round(m.ctx / 1024)}k ctx</span>}
                                            {m.priceIn && <span className="text-slate-600">•</span>}
                                            {m.priceIn && <span>{m.priceIn} in</span>}
                                            {m.priceOut && <span>{m.priceOut} out</span>}
                                        </div>
                                        <div className="text-[10px] text-slate-600 mt-1 truncate font-mono">{m.id}</div>
                                    </div>

                                    <button
                                        onClick={() => togglePin(m)}
                                        className={`p-2 rounded-full transition-colors ${isPinned(m.id) ? 'text-neon-cyan bg-neon-cyan/10' : 'text-slate-600 hover:text-white hover:bg-white/10'}`}
                                        title={isPinned(m.id) ? "Remove from selector" : "Pin to selector"}
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                                            <path fillRule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clipRule="evenodd" />
                                        </svg>
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>

                </div>

                {/* Corner resizer */}
                {!isMobile && (
                    <div
                        className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 flex items-end justify-end p-1 group"
                        onMouseDown={startResizingModal}
                    >
                        <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-neon-cyan transition-colors" />
                    </div>
                )}
            </div>
        </div>
    );
};

export default ModelCatalogModal;
