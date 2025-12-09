import React, { useState, useEffect } from 'react';
import { SparklesIcon } from './icons/SparklesIcon';
import { BookOpenIcon } from './icons/BookOpenIcon';
import { SwatchIcon } from './icons/SwatchIcon';
import SmartTooltip from './SmartTooltip';
import { Theme } from '../themes';
import ModelCatalogModal from './ModelCatalogModal';
import {
    HeaderChatModel,
    loadHeaderChatModels,
    loadSelectedModel,
    saveSelectedModel
} from '../services/openrouter-model-config';

interface HeaderProps {
    onAskAiClick: () => void;
    onOpenDocs: () => void;
    onToggleTheme: () => void;
    currentTheme: Theme;
    onToggleKernel: () => void;
    showTooltips: boolean;
    onToggleTooltips: () => void;
}

const Header: React.FC<HeaderProps> = ({
    onAskAiClick,
    onOpenDocs,
    onToggleTheme,
    currentTheme,
    onToggleKernel,
    showTooltips,
    onToggleTooltips
}) => {
    // Model Selector State
    const [models, setModels] = useState<HeaderChatModel[]>(() => loadHeaderChatModels());
    const [selectedModelId, setSelectedModelId] = useState<string>(() => loadSelectedModel(models));
    const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
    const [isCatalogOpen, setIsCatalogOpen] = useState(false);

    // Listen for model updates
    useEffect(() => {
        const handler = (ev: Event) => {
            const custom = ev as CustomEvent<HeaderChatModel[]>;
            if (Array.isArray(custom.detail)) {
                setModels(custom.detail);
                // If selected model is gone, fallback
                if (!custom.detail.find((m) => m.id === selectedModelId)) {
                    const fallback = custom.detail[0]?.id || loadSelectedModel(custom.detail);
                    setSelectedModelId(fallback);
                    saveSelectedModel(fallback);
                }
            }
        };
        window.addEventListener('or:header-models-updated', handler as EventListener);
        return () => window.removeEventListener('or:header-models-updated', handler as EventListener);
    }, [selectedModelId]);

    const handleModelSelect = (id: string) => {
        setSelectedModelId(id);
        saveSelectedModel(id);
        setIsModelMenuOpen(false);
    };

    const selectedModelName = models.find(m => m.id === selectedModelId)?.name || 'No Model active';

    return (
        <>
            <header className="h-14 border-b border-white/10 bg-[var(--bg-base)] flex items-center justify-between px-4 md:px-6 relative z-50 shrink-0">
                {/* LEFT: BRANDING */}
                <div className="flex items-center gap-4">
                    <SmartTooltip
                        id="header-brand"
                        content={
                            <div className="space-y-2">
                                <div className="text-xs text-slate-300">
                                    <span className="text-neon-cyan font-bold">OMEGA KERNEL</span> is the central nervous system of the D-ND architecture.
                                </div>
                                <div className="text-[10px] text-slate-500 border-t border-white/10 pt-2 mt-2">
                                    STATUS: <span className="text-green-400">ONLINE</span><br />
                                    VERSION: 2.4.0 (React Migration)
                                </div>
                            </div>
                        }
                        source="SYSTEM"
                        logic="Identity & Status"
                        outcome="User Awareness"
                        position="bottom"
                        alignment="start"
                        isVisible={showTooltips}
                    >
                        <div className="flex items-center gap-2 cursor-help group">
                            <div className="w-8 h-8 bg-neon-cyan/10 rounded flex items-center justify-center border border-neon-cyan/30 group-hover:bg-neon-cyan/20 transition-all">
                                <span className="text-neon-cyan font-bold text-lg">Ω</span>
                            </div>
                            <div className="hidden md:block">
                                <h1 className="text-sm font-bold text-white tracking-wider">D-ND OMEGA</h1>
                                <div className="text-[10px] text-slate-500 font-mono">COGNITIVE KERNEL</div>
                            </div>
                        </div>
                    </SmartTooltip>
                </div>

                {/* CENTER: ACTIONS */}
                <div className="flex items-center gap-2">
                    <SmartTooltip
                        source="VIEW_CONTROLLER"
                        logic="Switch to Kernel/Cockpit View"
                        outcome="Physics Engine Visualization"
                        position="bottom"
                        isVisible={showTooltips}
                    >
                        <button
                            onClick={onToggleKernel}
                            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-xs text-slate-300 hover:text-white transition-all flex items-center gap-2"
                        >
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span className="hidden md:inline">KERNEL VIEW</span>
                        </button>
                    </SmartTooltip>

                    {/* MODEL SELECTOR DROPDOWN */}
                    <SmartTooltip
                        source="OPENROUTER_SERVICE"
                        logic="Select Active LLM Model"
                        outcome="Inference Configuration"
                        position="bottom"
                        isVisible={showTooltips}
                    >
                        <div className="relative">
                            <button
                                onClick={() => setIsModelMenuOpen(!isModelMenuOpen)}
                                className="px-3 py-1.5 bg-neon-cyan/10 hover:bg-neon-cyan/20 border border-neon-cyan/30 rounded text-xs text-neon-cyan transition-all flex items-center gap-2 min-w-[40px] md:min-w-[160px] justify-between"
                            >
                                <div className="flex items-center gap-2">
                                    <SparklesIcon className="w-4 h-4" />
                                    <span className="truncate max-w-[120px] hidden md:inline">{selectedModelName}</span>
                                </div>
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 opacity-70">
                                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                                </svg>
                            </button>

                            {isModelMenuOpen && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setIsModelMenuOpen(false)}></div>
                                    <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 w-64 bg-[#0a0a0c] border border-white/20 rounded shadow-xl z-50 overflow-hidden flex flex-col">
                                        <button
                                            onClick={() => { setIsCatalogOpen(true); setIsModelMenuOpen(false); }}
                                            className="px-4 py-3 text-left text-xs font-bold text-neon-cyan hover:bg-white/5 border-b border-white/10 flex items-center gap-2"
                                        >
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                                            </svg>
                                            OPEN MODEL CATALOG...
                                        </button>

                                        <div className="max-h-60 overflow-y-auto py-1">
                                            <div className="px-2 py-1 text-[10px] text-slate-500 uppercase tracking-wider font-bold">Pinned Models</div>
                                            {models.map(m => (
                                                <button
                                                    key={m.id}
                                                    onClick={() => handleModelSelect(m.id)}
                                                    className={`w-full px-4 py-2 text-left text-xs hover:bg-white/5 flex items-center justify-between group ${m.id === selectedModelId ? 'text-white bg-white/5' : 'text-slate-400'}`}
                                                >
                                                    <span className="truncate">{m.name}</span>
                                                    {m.id === selectedModelId && <span className="w-1.5 h-1.5 rounded-full bg-neon-cyan"></span>}
                                                </button>
                                            ))}
                                            {models.length === 0 && (
                                                <div className="px-4 py-2 text-xs text-slate-600 italic">No pinned models</div>
                                            )}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </SmartTooltip>
                </div>

                {/* RIGHT: UTILS */}
                <div className="flex items-center gap-3">
                    <SmartTooltip
                        source="UI_CONTROLLER"
                        logic="Toggle System Tooltips"
                        outcome={showTooltips ? "Tooltips Enabled" : "Tooltips Disabled"}
                        position="bottom"
                        alignment="end"
                        isVisible={true} // Always visible for this button
                    >
                        <button
                            onClick={onToggleTooltips}
                            className={`p-2 rounded-full transition-all ${showTooltips ? 'text-neon-cyan bg-neon-cyan/10' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
                            </svg>
                        </button>
                    </SmartTooltip>

                    <SmartTooltip
                        source="KNOWLEDGE_BASE"
                        logic="Open Documentation Modal"
                        outcome="System Guide"
                        position="bottom"
                        alignment="end"
                        isVisible={showTooltips}
                    >
                        <button
                            onClick={onOpenDocs}
                            className="text-slate-400 hover:text-white transition-colors text-xs font-mono flex items-center gap-1"
                        >
                            <BookOpenIcon className="w-4 h-4" />
                            <span className="hidden md:inline">[DOCS]</span>
                        </button>
                    </SmartTooltip>

                    <SmartTooltip
                        source="THEME_ENGINE"
                        logic={`Switch Theme (Current: ${currentTheme})`}
                        outcome="Visual Adaptation"
                        position="bottom"
                        alignment="end"
                        isVisible={showTooltips}
                    >
                        <button
                            onClick={onToggleTheme}
                            className="w-6 h-6 rounded-full border border-white/20 bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 hover:scale-110 transition-transform flex items-center justify-center"
                        >
                            <SwatchIcon className="w-3 h-3 text-white opacity-50" />
                        </button>
                    </SmartTooltip>
                </div>
            </header>

            <ModelCatalogModal
                open={isCatalogOpen}
                onClose={() => setIsCatalogOpen(false)}
            />
        </>
    );
};

export default Header;
