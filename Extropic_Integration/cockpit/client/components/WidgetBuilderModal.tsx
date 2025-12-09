
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { WidgetConfig, WidgetType, DataSourceType, WidgetColorTheme, UploadedFile } from '../types';
import { PlusIcon } from './icons/PlusIcon';
import { SparklesIcon } from './icons/SparklesIcon';
import { TrashIcon } from './icons/TrashIcon';
import { MaximizeIcon } from './icons/MaximizeIcon';
import { SquaresPlusIcon } from './icons/SquaresPlusIcon';
import { generateWidgetConfig } from '../services/kernelBridge.service';
import FileUploader from './FileUploader';
import DynamicWidgetCard from './DynamicWidgetCard';
import { Resizer } from './Resizer';
import { SEMANTIC_CONFIG, getSemanticColor, HOLOGRAM_CONFIG } from '../themes';

interface WidgetBuilderModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (config: WidgetConfig) => void;
    modules: WidgetConfig[];
    onUpdateModule: (config: WidgetConfig) => void;
    onDeleteModule: (id: string) => void;
    initialTab?: 'registry' | 'forge';
}

const TEMPLATES = [
    { label: 'Profit Trend Analysis', type: 'area', source: 'management', keys: ['profit'], color: 'green' },
    { label: 'Revenue vs Cost', type: 'bar', source: 'management', keys: ['revenue', 'costs'], color: 'cyan' },
    { label: 'Mixed Performance', type: 'composed', source: 'management', keys: ['revenue', 'profit'], color: 'purple' },
    { label: 'Cost Breakdown Ring', type: 'pie', source: 'management', keys: ['costs', 'profit'], color: 'orange' },
    { label: 'Revenue Volatility', type: 'line', source: 'management', keys: ['revenue'], color: 'purple' },
    { label: 'Liquidity Burn Rate', type: 'line', source: 'cashflow', keys: ['cash'], color: 'orange' },
    { label: 'Cash Safety Margin', type: 'area', source: 'cashflow', keys: ['cash'], color: 'cyan' },
    { label: 'Bankability Radar', type: 'radar', source: 'rating', keys: ['score'], color: 'cyan' },
    { label: 'Payroll Impact', type: 'bar', source: 'hr', keys: ['payroll', 'costs'], color: 'purple' },
    { label: 'Headcount Efficiency', type: 'radial', source: 'hr', keys: ['headcount'], color: 'green' },
];

const DEFAULT_SUGGESTIONS = [
    { label: 'Opex Analyzer', description: 'Compare Revenue vs Costs', prompt: 'Create a cyan bar chart comparing Revenue and Costs.', icon: 'bar' },
    { label: 'Liquidity Horizon', description: '12-Month Cash Forecast', prompt: 'Generate a purple line chart showing the Cash Flow trend.', icon: 'line' },
    { label: 'Profit Core', description: 'Net Profit Visualization', prompt: 'Visualize the Net Profit evolution using a green area chart.', icon: 'area' },
    { label: 'Structural Balance', description: 'Revenue/Cost Donut', prompt: 'Create a donut chart (pie) for Revenue vs Total Costs.', icon: 'pie' },
    { label: 'Risk Metric Web', description: 'Rating Radar Scan', prompt: 'Generate a radar chart using Rating data.', icon: 'radar' }
];

type BuilderMode = 'manual' | 'ai';
type Tab = 'registry' | 'forge';
interface Suggestion { label: string; description: string; prompt: string; icon?: string; isDetected?: boolean; }
const PREVIEW_DATA_MNGT = Array.from({ length: 8 }, (_, i) => ({
    month: `M-${i + 1}`,
    revenue: 100 + Math.random() * 50,
    costs: 60 + Math.random() * 30,
    profit: 40 + Math.random() * 20,
    payroll: 30 + Math.random() * 10,
    headcount: 10,
    volume: Math.floor(Math.random() * 1000) + 500
}));

// Directions for resizing
type ResizeDirection = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

const WidgetBuilderModal: React.FC<WidgetBuilderModalProps> = ({ isOpen, onClose, onSave, modules, onUpdateModule, onDeleteModule, initialTab = 'registry' }) => {
    // --- DRAG & RESIZE STATE ---
    const modalRef = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ x: 100, y: 100 });
    const [size, setSize] = useState({ w: 1100, h: 700 }); // Slightly larger default
    const [isMaximized, setIsMaximized] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);

    const [activeTab, setActiveTab] = useState<Tab>(initialTab);
    const [mode, setMode] = useState<BuilderMode>('manual');

    // Forge Split Panel State
    const [configPanelWidth, setConfigPanelWidth] = useState(380);

    // Manual State
    const [title, setTitle] = useState('New Widget Construct');
    const [type, setType] = useState<WidgetType>('bar');
    const [source, setSource] = useState<DataSourceType>('management');
    const [color, setColor] = useState<WidgetColorTheme>('cyan');
    const [selectedKeys, setSelectedKeys] = useState<string[]>(['revenue']);

    // AI State
    const [aiPrompt, setAiPrompt] = useState('');
    const [attachedFile, setAttachedFile] = useState<UploadedFile | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [aiFeedback, setAiFeedback] = useState('');
    const [activeSuggestions, setActiveSuggestions] = useState<Suggestion[]>(DEFAULT_SUGGESTIONS);

    useEffect(() => {
        if (isOpen) {
            setActiveTab(initialTab);
            // Center modal
            const initialX = Math.max(0, (window.innerWidth - 1100) / 2);
            const initialY = Math.max(0, (window.innerHeight - 700) / 2);
            setPosition({ x: initialX, y: initialY });
        }
    }, [isOpen, initialTab]);

    useEffect(() => {
        if (!attachedFile) { setActiveSuggestions(DEFAULT_SUGGESTIONS); return; }
        setActiveSuggestions([{
            label: 'Deep Structure Scan', description: 'Let AI decipher the file',
            prompt: 'Analyze the structure of this file deeply, identify the most important numerical metrics and visualize them.',
            isDetected: true, icon: 'sparkle'
        }]);
    }, [attachedFile]);

    // Stable configuration for the preview to avoid re-renders during resize
    // MOVED UP to prevent Hook Error #310 (Hooks must be called unconditionally)
    const previewConfig: WidgetConfig = useMemo(() => ({
        id: 'preview',
        title: title || 'Untitled Widget',
        type, dataSource: source, dataKeys: selectedKeys.length > 0 ? selectedKeys : ['revenue'],
        colorTheme: color, isSystem: false, isVisible: true, colSpan: 1
    }), [title, type, source, selectedKeys, color]);

    // Stable data reference
    const previewData = PREVIEW_DATA_MNGT;

    // Stable no-op action
    const handlePreviewAction = useCallback(() => { }, []);
    const handleNoOpDelete = useCallback(() => { }, []);

    if (!isOpen) return null;

    // --- DRAG LOGIC (Header) ---
    const handleHeaderMouseDown = (e: React.MouseEvent) => {
        if (isMaximized) return;
        e.preventDefault();
        setIsDragging(true);
        const startX = e.clientX - position.x;
        const startY = e.clientY - position.y;

        const onMouseMove = (ev: MouseEvent) => {
            setPosition({
                x: ev.clientX - startX,
                y: ev.clientY - startY
            });
        };
        const onMouseUp = () => {
            setIsDragging(false);
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    // --- PANEL RESIZE LOGIC (Forge) ---
    const startConfigResize = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsDragging(true);
        const startX = e.clientX;
        const startWidth = configPanelWidth;

        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        let animationFrameId: number;

        const onMouseMove = (ev: MouseEvent) => {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            animationFrameId = requestAnimationFrame(() => {
                const delta = ev.clientX - startX;
                const newWidth = startWidth + delta;

                // Snap logic consistent with home
                if (newWidth < 100) {
                    setConfigPanelWidth(0); // Snap close
                } else if (newWidth >= 100 && newWidth < 280) {
                    setConfigPanelWidth(280); // Snap min width
                } else {
                    setConfigPanelWidth(Math.min(newWidth, 600)); // Max width
                }
            });
        };

        const onMouseUp = () => {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            setIsDragging(false);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    // --- MODAL RESIZE LOGIC ---
    const handleResizeStart = (direction: ResizeDirection) => (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (isMaximized) return;

        setIsResizing(true);
        const startX = e.clientX;
        const startY = e.clientY;
        const startW = size.w;
        const startH = size.h;
        const startPosX = position.x;
        const startPosY = position.y;

        const onMouseMove = (ev: MouseEvent) => {
            const deltaX = ev.clientX - startX;
            const deltaY = ev.clientY - startY;
            let newW = startW;
            let newH = startH;
            let newX = startPosX;
            let newY = startPosY;

            if (direction.includes('e')) newW = Math.max(800, startW + deltaX);
            else if (direction.includes('w')) { newW = Math.max(800, startW - deltaX); newX = startPosX + (startW - newW); }
            if (direction.includes('s')) newH = Math.max(500, startH + deltaY);
            else if (direction.includes('n')) { newH = Math.max(500, startH - deltaY); newY = startPosY + (startH - newH); }

            setSize({ w: newW, h: newH });
            setPosition({ x: newX, y: newY });
        };

        const onMouseUp = () => {
            setIsResizing(false);
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    const handleSaveManual = () => {
        if (!title) return;
        onSave({ id: Date.now().toString(), title, type, dataSource: source, dataKeys: selectedKeys, colorTheme: color, isSystem: false, isVisible: true, colSpan: 1 });
        setActiveTab('registry');
    };

    const handleGenerateAi = async () => {
        if (!aiPrompt.trim() && !attachedFile) return;
        setIsGenerating(true);
        setAiFeedback('');
        try {
            const response = await generateWidgetConfig(aiPrompt, attachedFile || undefined);
            onSave(response.config);
            setAiFeedback(response.explanation);
            setAiPrompt(''); setAttachedFile(null); setActiveTab('registry');
        } catch (e) { console.error(e); setAiFeedback("Gen Error: " + (e as any).message); }
        finally { setIsGenerating(false); }
    };

    const loadTemplate = (tpl: any) => {
        setTitle(tpl.label); setType(tpl.type as WidgetType); setSource(tpl.source as DataSourceType);
        setSelectedKeys(tpl.keys); setColor(tpl.color); setMode('manual');
    };
    const toggleKey = (key: string) => setSelectedKeys(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]);
    const getAvailableKeys = () => {
        if (source === 'management') return ['revenue', 'costs', 'profit', 'volume'];
        if (source === 'cashflow') return ['cash'];
        if (source === 'rating') return ['score'];
        if (source === 'hr') return ['payroll', 'headcount'];
        return [];
    };

    const containerStyle = isMaximized
        ? { position: 'fixed' as const, top: 0, left: 0, width: '100%', height: '100%', borderRadius: 0 }
        : { position: 'fixed' as const, top: position.y, left: position.x, width: size.w, height: size.h };

    const activeModules = modules.filter(m => m.isVisible);
    const hiddenSystemModules = modules.filter(m => !m.isVisible && m.isSystem);

    const renderSelect = (label: string, value: string, onChange: (val: string) => void, options: { val: string, label: string }[]) => (
        <div className="flex flex-col gap-1">
            <span className="text-[9px] uppercase tracking-wider text-slate-500 font-mono">{label}</span>
            <div className="relative group">
                <select
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    className="w-full border border-white/10 rounded p-2 outline-none appearance-none hover:border-neon-cyan focus:border-neon-cyan font-mono text-xs cursor-pointer bg-black text-white shadow-lg transition-all"
                >
                    {options.map(opt => (
                        <option key={opt.val} value={opt.val} className="bg-black text-white py-2">{opt.label}</option>
                    ))}
                </select>
                <div className="absolute right-2 top-2.5 pointer-events-none text-neon-cyan text-[10px]">▼</div>
            </div>
        </div>
    );

    return createPortal(
        <div className="fixed inset-0 z-[9999] pointer-events-none">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-md pointer-events-auto transition-opacity duration-300" onClick={onClose} aria-hidden="true"></div>

            <div
                ref={modalRef}
                style={containerStyle}
                className="pointer-events-auto border border-white/10 shadow-deep flex flex-col overflow-hidden bg-[#050b14] rounded-xl transition-all duration-75"
            >
                {!isMaximized && (
                    <>
                        <div onMouseDown={handleResizeStart('nw')} className="absolute top-0 left-0 w-4 h-4 cursor-nw-resize z-50"></div>
                        <div onMouseDown={handleResizeStart('ne')} className="absolute top-0 right-0 w-4 h-4 cursor-ne-resize z-50"></div>
                        <div onMouseDown={handleResizeStart('sw')} className="absolute bottom-0 left-0 w-4 h-4 cursor-sw-resize z-50"></div>
                        <div onMouseDown={handleResizeStart('se')} className="absolute bottom-0 right-0 w-6 h-6 cursor-se-resize z-50 flex items-end justify-end p-1 group">
                            <div className="w-2 h-2 border-r-2 border-b-2 border-slate-500 group-hover:border-white"></div>
                        </div>
                        <div onMouseDown={handleResizeStart('n')} className="absolute top-0 left-4 right-4 h-2 cursor-n-resize z-40"></div>
                        <div onMouseDown={handleResizeStart('s')} className="absolute bottom-0 left-4 right-4 h-2 cursor-s-resize z-40"></div>
                        <div onMouseDown={handleResizeStart('w')} className="absolute left-0 top-4 bottom-4 w-2 cursor-w-resize z-40"></div>
                        <div onMouseDown={handleResizeStart('e')} className="absolute right-0 top-4 bottom-4 w-2 cursor-e-resize z-40"></div>
                    </>
                )}

                {/* HEADER with TABS */}
                <div
                    onMouseDown={handleHeaderMouseDown}
                    className={`h-16 pl-4 pr-4 border-b border-white/10 flex justify-between items-center relative shrink-0 bg-black select-none ${isMaximized ? '' : 'cursor-move'}`}
                >
                    <div className="flex items-center gap-6">
                        <div className="flex items-center gap-3">
                            <div className="p-1.5 rounded border border-white/10 bg-black/50"><PlusIcon className="h-4 w-4 text-neon-cyan" /></div>
                            <div>
                                <h2 className="text-sm font-bold uppercase tracking-widest text-white">Module Configurator</h2>
                            </div>
                        </div>

                        {/* TABS IN HEADER */}
                        <div className="flex items-center bg-black/50 rounded-lg p-1 border border-white/10" onMouseDown={e => e.stopPropagation()}>
                            <button
                                onClick={() => setActiveTab('registry')}
                                className={`px-4 py-1.5 text-[10px] font-bold uppercase rounded-md transition-all flex items-center gap-2 ${activeTab === 'registry' ? 'bg-[#00f3ff] text-slate-900 shadow-[0_0_15px_rgba(0,243,255,0.4)]' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                            >
                                <SquaresPlusIcon className="w-3 h-3" /> Registry
                            </button>
                            <button
                                onClick={() => setActiveTab('forge')}
                                className={`px-4 py-1.5 text-[10px] font-bold uppercase rounded-md transition-all flex items-center gap-2 ${activeTab === 'forge' ? 'bg-neon-purple text-white shadow-[0_0_15px_rgba(188,19,254,0.4)]' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                            >
                                <SparklesIcon className="w-3 h-3" /> Forge
                            </button>
                        </div>
                    </div>

                    <div className="flex items-center gap-2" onMouseDown={e => e.stopPropagation()}>
                        <button onClick={() => setIsMaximized(!isMaximized)} className="hover:bg-white/10 p-1.5 rounded text-slate-400 hover:text-white"><MaximizeIcon className="w-4 h-4" /></button>
                        <button onClick={onClose} className="hover:bg-white/10 p-1.5 rounded text-slate-400 hover:text-white">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                    </div>
                </div>

                {/* BODY */}
                <div className="flex flex-grow overflow-hidden relative z-10 bg-black/20">

                    {/* TAB: REGISTRY (Full Width) */}
                    {activeTab === 'registry' && (
                        <div className="w-full h-full p-6 overflow-y-auto custom-scrollbar">
                            <div className="max-w-4xl mx-auto space-y-8">
                                <div className="space-y-4">
                                    <h3 className="text-xs font-bold uppercase text-neon-cyan tracking-widest border-b border-white/10 pb-2">Active Matrix</h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {activeModules.map(mod => (
                                            <div key={mod.id} className="p-4 rounded border border-white/10 bg-white/5 flex justify-between items-center hover:border-neon-cyan/30 transition-colors group">
                                                <div className="flex items-center gap-3">
                                                    <div className={`w-2 h-2 rounded-full bg-${mod.colorTheme === 'cyan' ? 'cyan-400' : mod.colorTheme === 'purple' ? 'purple-500' : mod.colorTheme === 'green' ? 'green-400' : 'orange-400'} shadow-[0_0_5px]`}></div>
                                                    <span className="text-sm text-slate-200 font-mono">{mod.title}</span>
                                                </div>
                                                <button onClick={() => onDeleteModule(mod.id)} className="p-2 text-slate-500 hover:text-red-500 hover:bg-red-500/10 rounded transition-colors"><TrashIcon className="w-4 h-4" /></button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                {hiddenSystemModules.length > 0 && (
                                    <div className="space-y-4 opacity-60 hover:opacity-100 transition-opacity">
                                        <h3 className="text-xs font-bold uppercase text-slate-500 tracking-widest border-b border-slate-800 pb-2">Offline Modules</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {hiddenSystemModules.map(mod => (
                                                <div key={mod.id} className="p-4 rounded border border-dashed border-slate-700 flex justify-between items-center bg-black/20">
                                                    <span className="text-sm text-slate-500 font-mono">{mod.title}</span>
                                                    <button onClick={() => onUpdateModule({ ...mod, isVisible: true })} className="text-[10px] bg-neon-cyan/10 text-neon-cyan px-3 py-1.5 rounded uppercase border border-neon-cyan/20 hover:bg-neon-cyan/20 hover:border-neon-cyan">Re-Initialize</button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* TAB: FORGE (Split Pane) */}
                    {activeTab === 'forge' && (
                        <div className="flex w-full h-full">

                            {/* LEFT: CONFIGURATION PANEL (Resizable) */}
                            <div
                                style={{ width: configPanelWidth > 0 ? configPanelWidth : '48px' }}
                                className="shrink-0 flex flex-col border-r border-white/10 bg-black/20 relative transition-[width] duration-75 ease-linear"
                            >
                                {configPanelWidth < 100 ? (
                                    // Collapsed State
                                    <div className="h-full w-full flex flex-col items-center py-6 gap-4">
                                        <button onClick={() => setConfigPanelWidth(380)} className="p-2 text-slate-500 hover:text-neon-cyan"><PlusIcon className="w-5 h-5" /></button>
                                    </div>
                                ) : (
                                    // Expanded State
                                    <div className="flex flex-col h-full">
                                        {/* Mode Toggles */}
                                        <div className="p-4 border-b border-white/10 shrink-0 flex gap-2">
                                            <button
                                                onClick={() => setMode('manual')}
                                                className={`flex-1 py-2 text-[10px] font-bold uppercase rounded transition-all border ${mode === 'manual' ? 'bg-neon-cyan/10 text-neon-cyan border-neon-cyan shadow-[0_0_10px_rgba(0,243,255,0.2)]' : 'bg-transparent text-slate-400 border-white/10 hover:border-white/30'}`}
                                            >
                                                Manual
                                            </button>
                                            <button
                                                onClick={() => setMode('ai')}
                                                className={`flex-1 py-2 text-[10px] font-bold uppercase rounded transition-all border flex items-center justify-center gap-2 ${mode === 'ai' ? 'bg-neon-purple/10 text-neon-purple border-neon-purple shadow-[0_0_10px_rgba(188,19,254,0.2)]' : 'bg-transparent text-slate-400 border-white/10 hover:border-white/30'}`}
                                            >
                                                <SparklesIcon className="w-3 h-3" /> AI Auto
                                            </button>
                                        </div>

                                        {/* Scrollable Config Area */}
                                        <div className="flex-grow overflow-y-auto custom-scrollbar p-4">
                                            {mode === 'manual' ? (
                                                <div className="space-y-8">
                                                    {/* 1. Templates - Visual Grid */}
                                                    <div className="space-y-3">
                                                        <h4 className="text-[9px] font-bold uppercase text-slate-500 tracking-widest flex items-center gap-2">
                                                            <span className="w-1 h-1 bg-neon-cyan rounded-full"></span>
                                                            Blueprint Selection
                                                        </h4>
                                                        <div className="grid grid-cols-2 gap-2">
                                                            {TEMPLATES.map((tpl, i) => (
                                                                <div
                                                                    key={i}
                                                                    onClick={() => loadTemplate(tpl)}
                                                                    className={`cursor-pointer border p-2.5 rounded group transition-all duration-200 relative overflow-hidden ${title === tpl.label ? 'border-neon-cyan bg-neon-cyan/5' : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'}`}
                                                                >
                                                                    <div className="relative z-10">
                                                                        <div className="text-[10px] font-bold text-slate-300 group-hover:text-white truncate">{tpl.label}</div>
                                                                        <div className="text-[9px] text-slate-500 font-mono mt-1 capitalize">{tpl.type} • {tpl.color}</div>
                                                                    </div>
                                                                    {title === tpl.label && <div className="absolute top-0 right-0 w-2 h-2 bg-neon-cyan shadow-[0_0_5px_#00f3ff]"></div>}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>

                                                    {/* Divider */}
                                                    <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

                                                    {/* 2. Inputs - More spaced out */}
                                                    <div className="space-y-5">
                                                        <h4 className="text-[9px] font-bold uppercase text-slate-500 tracking-widest flex items-center gap-2">
                                                            <span className="w-1 h-1 bg-neon-purple rounded-full"></span>
                                                            Construct Parameters
                                                        </h4>

                                                        <div className="flex flex-col gap-1">
                                                            <span className="text-[9px] uppercase tracking-wider text-slate-500 font-mono">Widget Title</span>
                                                            <input
                                                                type="text"
                                                                value={title}
                                                                onChange={(e) => setTitle(e.target.value)}
                                                                className="w-full bg-black border border-white/10 rounded p-2 text-sm text-white focus:border-neon-cyan outline-none font-mono placeholder-slate-600 transition-colors focus:shadow-[0_0_10px_rgba(0,243,255,0.2)]"
                                                                placeholder="Enter Identifier..."
                                                            />
                                                        </div>

                                                        <div className="grid grid-cols-2 gap-4">
                                                            {renderSelect('Input Stream', source, (v) => setSource(v as DataSourceType), [{ val: 'management', label: 'P&L Data' }, { val: 'cashflow', label: 'Cash Flow' }, { val: 'rating', label: 'Rating' }, { val: 'hr', label: 'HR Analytics' }])}
                                                            {renderSelect('Visualizer', type, (v) => setType(v as WidgetType), [{ val: 'bar', label: 'Bar Chart' }, { val: 'line', label: 'Line Chart' }, { val: 'area', label: 'Area Graph' }, { val: 'pie', label: 'Donut/Pie' }, { val: 'composed', label: 'Composed' }, { val: 'radar', label: 'Radar Scan' }, { val: 'radial', label: 'Radial Gauge' }])}
                                                        </div>

                                                        <div className="flex flex-col gap-2">
                                                            <span className="text-[9px] uppercase tracking-wider text-slate-500 font-mono">Active Metrics</span>
                                                            <div className="flex flex-wrap gap-2 p-2 bg-black/30 rounded border border-white/5">
                                                                {getAvailableKeys().map(key => (
                                                                    <button
                                                                        key={key}
                                                                        onClick={() => toggleKey(key)}
                                                                        className={`px-3 py-1.5 rounded-sm text-[10px] uppercase font-bold tracking-wider border transition-all duration-200 ${selectedKeys.includes(key) ? 'bg-neon-cyan/20 border-neon-cyan text-neon-cyan shadow-[0_0_10px_rgba(0,243,255,0.2)]' : 'bg-black border-white/10 text-slate-500 hover:border-white/30 hover:text-slate-300'}`}
                                                                    >
                                                                        {key}
                                                                    </button>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        <div className="flex flex-col gap-2">
                                                            <span className="text-[9px] uppercase tracking-wider text-slate-500 font-mono">Hologram Theme</span>
                                                            <div className="flex gap-3 items-center">
                                                                {(['cyan', 'purple', 'green', 'orange'] as WidgetColorTheme[]).map(c => (
                                                                    <button
                                                                        key={c}
                                                                        onClick={() => setColor(c)}
                                                                        className={`w-8 h-8 rounded-md border-2 transition-all flex items-center justify-center relative overflow-hidden group ${color === c ? 'border-white scale-110 shadow-lg' : 'border-transparent opacity-60 hover:opacity-100 hover:scale-105'}`}
                                                                        style={{ backgroundColor: 'rgba(0,0,0,0.5)', borderColor: color === c ? '#fff' : 'rgba(255,255,255,0.1)' }}
                                                                    >
                                                                        <div className="absolute inset-0 opacity-50" style={{ backgroundColor: c === 'cyan' ? '#00f3ff' : c === 'purple' ? '#bc13fe' : c === 'green' ? '#22c55e' : '#f97316' }}></div>
                                                                        {color === c && <div className="w-1.5 h-1.5 bg-white rounded-full relative z-10 shadow-[0_0_5px_white]"></div>}
                                                                    </button>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ) : (
                                                /* AI MODE CONTENT */
                                                <div className="flex flex-col gap-6">
                                                    <FileUploader selectedFile={attachedFile} onFileSelect={setAttachedFile} />
                                                    <div className="flex flex-col gap-2">
                                                        <span className="text-[9px] uppercase tracking-wider text-slate-500 font-mono">Command Prompt</span>
                                                        <textarea value={aiPrompt} onChange={(e) => setAiPrompt(e.target.value)} placeholder="Describe the chart you want to build..." className="w-full h-32 bg-black border border-white/10 rounded p-3 text-sm text-white font-mono outline-none focus:border-neon-purple transition-colors focus:shadow-[0_0_10px_rgba(188,19,254,0.2)]" />
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-2">
                                                        {activeSuggestions.slice(0, 4).map((s, i) => <button key={i} onClick={() => setAiPrompt(s.prompt)} className="text-left p-2 border border-white/5 hover:border-neon-purple/50 rounded text-[10px] text-slate-400 hover:text-white bg-white/5 transition-colors">{s.label}</button>)}
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        {/* Footer Actions */}
                                        <div className="p-4 border-t border-white/10 shrink-0 bg-black/40 backdrop-blur">
                                            {mode === 'manual' ? (
                                                <button
                                                    onClick={handleSaveManual}
                                                    disabled={!title}
                                                    className="w-full py-3 border border-neon-cyan text-neon-cyan font-bold uppercase tracking-widest rounded hover:bg-neon-cyan hover:text-black hover:shadow-[0_0_20px_rgba(0,243,255,0.6)] transition-all disabled:opacity-50 disabled:shadow-none disabled:border-slate-700 disabled:text-slate-500 disabled:bg-transparent bg-transparent relative overflow-hidden group"
                                                >
                                                    <span className="relative z-10">Initialize System</span>
                                                    <div className="absolute inset-0 bg-neon-cyan/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                                                </button>
                                            ) : (
                                                <button onClick={handleGenerateAi} disabled={isGenerating} className="w-full py-3 bg-transparent border border-neon-purple text-neon-purple font-bold uppercase tracking-widest rounded hover:bg-neon-purple hover:text-white hover:shadow-[0_0_20px_rgba(188,19,254,0.6)] disabled:opacity-50 transition-all">
                                                    {isGenerating ? 'Architecting...' : 'Generate Construct'}
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* RESIZER HANDLER */}
                            <Resizer onMouseDown={startConfigResize} isVisible={true} />

                            {/* RIGHT: PREVIEW PANEL */}
                            <div className="flex-1 flex flex-col h-full overflow-hidden bg-black/40 relative group">
                                {/* Preview Background FX */}
                                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[length:40px_40px] pointer-events-none"></div>
                                <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/20 pointer-events-none"></div>

                                <div className="absolute top-4 right-4 z-20">
                                    <div className="text-[9px] font-mono text-neon-cyan uppercase tracking-widest bg-black/60 px-2 py-1 rounded border border-neon-cyan/20 shadow-neon/20 backdrop-blur">
                                        Live Preview
                                    </div>
                                </div>

                                <div className="flex-grow p-8 flex items-center justify-center relative z-10 w-full h-full overflow-auto">
                                    <div className="w-full max-w-3xl aspect-video min-h-[300px] relative">
                                        {/* Holographic Container */}
                                        <div className="absolute -inset-1 bg-gradient-to-br from-neon-cyan/20 to-neon-purple/20 rounded-lg blur-sm opacity-50"></div>
                                        <div className="relative w-full h-full bg-obsidian border border-white/10 rounded-lg shadow-2xl overflow-hidden flex flex-col">
                                            <DynamicWidgetCard
                                                config={previewConfig}
                                                data={previewData}
                                                onDelete={handleNoOpDelete}
                                                currentTheme="cyberpunk"
                                                onAction={handlePreviewAction}
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div className="p-2 text-center text-[9px] text-slate-600 font-mono bg-black/40 border-t border-white/5">
                                    Visual approximation based on synthetic data pattern // RENDER_ENGINE_v2.8
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>,
        document.body
    );
};

export default WidgetBuilderModal;
