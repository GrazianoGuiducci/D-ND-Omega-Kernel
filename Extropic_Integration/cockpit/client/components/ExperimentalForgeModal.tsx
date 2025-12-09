import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { PlusIcon } from './icons/PlusIcon';
import { SparklesIcon } from './icons/SparklesIcon';
import { MaximizeIcon } from './icons/MaximizeIcon';
import { Resizer } from './Resizer';
import { UploadedFile } from '../types';
import FileUploader from './FileUploader';
import SmartTooltip from './SmartTooltip';

// Forge Service Integration
const generateExperimentCode = async (prompt: string, model: string = "google/gemini-2.0-flash-exp:free") => {
    const response = await fetch('/api/forge/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model })
    });
    if (!response.ok) throw new Error('Generation failed');
    return await response.json();
};

const injectExperimentCode = async (code: string, filename: string) => {
    const response = await fetch('/api/forge/inject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, filename })
    });
    if (!response.ok) throw new Error('Injection failed');
    return await response.json();
};

interface ExperimentalForgeModalProps {
    isOpen: boolean;
    onClose: () => void;
    onInject: (code: string, filename: string) => void;
}

type BuilderMode = 'manual' | 'ai';
type Tab = 'architect' | 'blueprint';
type ResizeDirection = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

const TEMPLATES = [
    { label: 'Fibonacci Sequence', description: 'Mathematical growth pattern', icon: '🔢' },
    { label: 'Ising Grid 10x10', description: 'Standard thermodynamic lattice', icon: '🕸️' },
    { label: 'Market Volatility', description: 'Financial entropy simulation', icon: '📉' },
    { label: 'Quantum Walk', description: 'Probabilistic exploration', icon: '⚛️' },
];

const ExperimentalForgeModal: React.FC<ExperimentalForgeModalProps> = ({ isOpen, onClose, onInject }) => {
    // --- DRAG & RESIZE STATE ---
    const modalRef = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ x: 100, y: 100 });
    const [size, setSize] = useState({ w: 1200, h: 800 });
    const [isMaximized, setIsMaximized] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);

    const [activeTab, setActiveTab] = useState<Tab>('architect');
    const [mode, setMode] = useState<BuilderMode>('ai'); // Default to AI for Forge

    // Split Panel State
    const [configPanelWidth, setConfigPanelWidth] = useState(450);

    // Forge State
    const [prompt, setPrompt] = useState('');
    const [attachedFile, setAttachedFile] = useState<UploadedFile | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedCode, setGeneratedCode] = useState('');
    const [explanation, setExplanation] = useState('');
    const [filename, setFilename] = useState('experiment_01.py');

    useEffect(() => {
        if (isOpen) {
            // Center modal
            const initialX = Math.max(0, (window.innerWidth - 1200) / 2);
            const initialY = Math.max(0, (window.innerHeight - 800) / 2);
            setPosition({ x: initialX, y: initialY });
        }
    }, [isOpen]);

    if (!isOpen) return null;

    // --- DRAG LOGIC ---
    const handleHeaderMouseDown = (e: React.MouseEvent) => {
        if (isMaximized) return;
        e.preventDefault();
        setIsDragging(true);
        const startX = e.clientX - position.x;
        const startY = e.clientY - position.y;

        const onMouseMove = (ev: MouseEvent) => {
            setPosition({ x: ev.clientX - startX, y: ev.clientY - startY });
        };
        const onMouseUp = () => {
            setIsDragging(false);
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    // --- RESIZE LOGIC ---
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

    // --- PANEL RESIZE ---
    const startConfigResize = (e: React.MouseEvent) => {
        e.preventDefault();
        const startX = e.clientX;
        const startWidth = configPanelWidth;
        const onMouseMove = (ev: MouseEvent) => {
            setConfigPanelWidth(Math.max(300, Math.min(800, startWidth + (ev.clientX - startX))));
        };
        const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    const handleGenerate = async () => {
        if (!prompt) return;
        setIsGenerating(true);
        try {
            // TODO: Add model selector in UI, currently hardcoded or using default
            const result = await generateExperimentCode(prompt);
            setGeneratedCode(result.code);
            setExplanation(result.explanation);
            if (result.filename) setFilename(result.filename);
            setActiveTab('blueprint'); // Switch to preview
        } catch (e) {
            console.error(e);
            setExplanation("Error generating code. Please try again.");
        } finally {
            setIsGenerating(false);
        }
    };

    const handleInject = async () => {
        try {
            await injectExperimentCode(generatedCode, filename);
            onInject(generatedCode, filename);
        } catch (e) {
            console.error("Injection failed", e);
            alert("Failed to inject experiment. Check console.");
        }
    };

    const containerStyle = isMaximized
        ? { position: 'fixed' as const, top: 0, left: 0, width: '100%', height: '100%', borderRadius: 0 }
        : { position: 'fixed' as const, top: position.y, left: position.x, width: size.w, height: size.h };

    return createPortal(
        <div className="fixed inset-0 z-[9999] pointer-events-none">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-md pointer-events-auto transition-opacity duration-300" onClick={onClose} aria-hidden="true"></div>

            <div
                ref={modalRef}
                style={containerStyle}
                className="pointer-events-auto border border-white/10 shadow-deep flex flex-col overflow-hidden bg-[#050b14] rounded-xl transition-all duration-75"
            >
                {/* RESIZE HANDLES - Ensure high z-index */}
                {!isMaximized && (
                    <>
                        <div onMouseDown={handleResizeStart('nw')} className="absolute top-0 left-0 w-4 h-4 cursor-nw-resize z-[60]"></div>
                        <div onMouseDown={handleResizeStart('ne')} className="absolute top-0 right-0 w-4 h-4 cursor-ne-resize z-[60]"></div>
                        <div onMouseDown={handleResizeStart('sw')} className="absolute bottom-0 left-0 w-4 h-4 cursor-sw-resize z-[60]"></div>
                        <div onMouseDown={handleResizeStart('se')} className="absolute bottom-0 right-0 w-6 h-6 cursor-se-resize z-[60]"></div>
                        <div onMouseDown={handleResizeStart('n')} className="absolute top-0 left-4 right-4 h-2 cursor-n-resize z-[55]"></div>
                        <div onMouseDown={handleResizeStart('s')} className="absolute bottom-0 left-4 right-4 h-2 cursor-s-resize z-[55]"></div>
                        <div onMouseDown={handleResizeStart('w')} className="absolute left-0 top-4 bottom-4 w-2 cursor-w-resize z-[55]"></div>
                        <div onMouseDown={handleResizeStart('e')} className="absolute right-0 top-4 bottom-4 w-2 cursor-e-resize z-[55]"></div>
                    </>
                )}

                {/* HEADER */}
                <div
                    onMouseDown={handleHeaderMouseDown}
                    className={`h-16 pl-4 pr-4 border-b border-white/10 flex justify-between items-center relative shrink-0 bg-black select-none ${isMaximized ? '' : 'cursor-move'}`}
                >
                    <div className="flex items-center gap-6">
                        <div className="flex items-center gap-3">
                            <div className="p-1.5 rounded border border-white/10 bg-black/50"><SparklesIcon className="h-4 w-4 text-neon-purple" /></div>
                            <div>
                                <h2 className="text-sm font-bold uppercase tracking-widest text-white">Experimental Forge</h2>
                            </div>
                        </div>

                        {/* TABS */}
                        <div className="flex items-center bg-black/50 rounded-lg p-1 border border-white/10" onMouseDown={e => e.stopPropagation()}>
                            <SmartTooltip source="FORGE_UI" logic="Switch to Architect View" outcome="Configure experiment parameters">
                                <button
                                    onClick={() => setActiveTab('architect')}
                                    className={`px-4 py-1.5 text-[10px] font-bold uppercase rounded-md transition-all flex items-center gap-2 ${activeTab === 'architect' ? 'bg-neon-purple text-white shadow-[0_0_15px_rgba(188,19,254,0.4)]' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                                >
                                    <PlusIcon className="w-3 h-3" /> Architect
                                </button>
                            </SmartTooltip>
                            <SmartTooltip source="FORGE_UI" logic="Switch to Blueprint View" outcome="Preview generated Python code">
                                <button
                                    onClick={() => setActiveTab('blueprint')}
                                    className={`px-4 py-1.5 text-[10px] font-bold uppercase rounded-md transition-all flex items-center gap-2 ${activeTab === 'blueprint' ? 'bg-neon-cyan text-black shadow-[0_0_15px_rgba(0,243,255,0.4)]' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                                >
                                    <span className="font-mono">{'{}'}</span> Blueprint
                                </button>
                            </SmartTooltip>
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

                    {/* LEFT: ARCHITECT PANEL */}
                    <div
                        style={{ width: configPanelWidth }}
                        className="shrink-0 flex flex-col border-r border-white/10 bg-black/20 relative transition-[width] duration-75 ease-linear"
                    >
                        <div className="flex flex-col h-full">
                            {/* Mode Toggles */}
                            <div className="p-4 border-b border-white/10 shrink-0 flex gap-2">
                                <SmartTooltip source="FORGE_AI" logic="AI Generation Mode" outcome="Use LLM to generate code from prompt">
                                    <button
                                        onClick={() => setMode('ai')}
                                        className={`flex-1 py-2 text-[10px] font-bold uppercase rounded transition-all border flex items-center justify-center gap-2 ${mode === 'ai' ? 'bg-neon-purple/10 text-neon-purple border-neon-purple shadow-[0_0_10px_rgba(188,19,254,0.2)]' : 'bg-transparent text-slate-400 border-white/10 hover:border-white/30'}`}
                                    >
                                        <SparklesIcon className="w-3 h-3" /> AI Architect
                                    </button>
                                </SmartTooltip>
                                <SmartTooltip source="FORGE_TEMPLATE" logic="Manual Template Mode" outcome="Select from pre-defined protocols">
                                    <button
                                        onClick={() => setMode('manual')}
                                        className={`flex-1 py-2 text-[10px] font-bold uppercase rounded transition-all border ${mode === 'manual' ? 'bg-neon-cyan/10 text-neon-cyan border-neon-cyan shadow-[0_0_10px_rgba(0,243,255,0.2)]' : 'bg-transparent text-slate-400 border-white/10 hover:border-white/30'}`}
                                    >
                                        Manual Template
                                    </button>
                                </SmartTooltip>
                            </div>

                            {/* Config Area */}
                            <div className="flex-grow overflow-y-auto custom-scrollbar p-6 space-y-6">
                                {mode === 'ai' ? (
                                    <>
                                        <div className="space-y-2">
                                            <h3 className="text-[10px] font-bold uppercase text-slate-500 tracking-widest">Data Ingestion</h3>
                                            <FileUploader selectedFile={attachedFile} onFileSelect={setAttachedFile} />
                                        </div>
                                        <div className="space-y-2">
                                            <h3 className="text-[10px] font-bold uppercase text-slate-500 tracking-widest">Natural Language Prompt</h3>
                                            <textarea
                                                value={prompt}
                                                onChange={(e) => setPrompt(e.target.value)}
                                                placeholder="Describe the experiment you want to create (e.g., 'Simulate a 3-body problem with high entropy')..."
                                                className="w-full h-48 bg-black border border-white/10 rounded p-4 text-sm text-white font-mono outline-none focus:border-neon-purple transition-colors focus:shadow-[0_0_10px_rgba(188,19,254,0.2)] resize-none"
                                            />
                                        </div>
                                    </>
                                ) : (
                                    <div className="space-y-4">
                                        <h3 className="text-[10px] font-bold uppercase text-slate-500 tracking-widest">Standard Protocols</h3>
                                        <div className="grid grid-cols-1 gap-3">
                                            {TEMPLATES.map((tpl, i) => (
                                                <SmartTooltip key={i} source="TEMPLATE_LIB" logic={`Select ${tpl.label}`} outcome="Load template into prompt">
                                                    <div
                                                        onClick={() => { setPrompt(`Create a ${tpl.label} experiment.`); setMode('ai'); }}
                                                        className="cursor-pointer border border-white/10 bg-white/5 p-4 rounded hover:bg-white/10 hover:border-neon-cyan transition-all group"
                                                    >
                                                        <div className="flex items-center gap-3">
                                                            <span className="text-2xl">{tpl.icon}</span>
                                                            <div>
                                                                <div className="text-sm font-bold text-slate-200 group-hover:text-neon-cyan">{tpl.label}</div>
                                                                <div className="text-xs text-slate-500">{tpl.description}</div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </SmartTooltip>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Footer */}
                            <div className="p-4 border-t border-white/10 shrink-0 bg-black/40 backdrop-blur">
                                <SmartTooltip source="FORGE_ACTION" logic="Generate Protocol" outcome="Send prompt to LLM">
                                    <button
                                        onClick={handleGenerate}
                                        disabled={isGenerating || (mode === 'ai' && !prompt)}
                                        className="w-full py-3 bg-transparent border border-neon-purple text-neon-purple font-bold uppercase tracking-widest rounded hover:bg-neon-purple hover:text-white hover:shadow-[0_0_20px_rgba(188,19,254,0.6)] disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                                    >
                                        {isGenerating ? (
                                            <>
                                                <span className="animate-spin">⚙️</span> Forging...
                                            </>
                                        ) : (
                                            <>
                                                <SparklesIcon className="w-4 h-4" /> Generate Protocol
                                            </>
                                        )}
                                    </button>
                                </SmartTooltip>
                            </div>
                        </div>
                    </div>

                    {/* RESIZER */}
                    <Resizer onMouseDown={startConfigResize} isVisible={true} />

                    {/* RIGHT: BLUEPRINT PANEL */}
                    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#0a0a0a] relative">
                        {generatedCode ? (
                            <>
                                <div className="flex-grow overflow-auto p-0 relative">
                                    <div className="absolute top-0 left-0 w-full h-8 bg-white/5 border-b border-white/10 flex items-center px-4 justify-between">
                                        <span className="text-[10px] font-mono text-slate-400">{filename}</span>
                                        <span className="text-[10px] font-mono text-neon-green">PYTHON 3.10</span>
                                    </div>
                                    <textarea
                                        value={generatedCode}
                                        onChange={(e) => setGeneratedCode(e.target.value)}
                                        className="w-full h-full bg-[#0a0a0a] text-slate-300 font-mono text-xs p-4 pt-10 outline-none resize-none"
                                        spellCheck={false}
                                    />
                                </div>
                                <div className="p-4 border-t border-white/10 bg-black/40 backdrop-blur flex justify-between items-center">
                                    <div className="text-xs text-slate-500 max-w-md truncate">
                                        {explanation}
                                    </div>
                                    <SmartTooltip source="FORGE_INJECT" logic="Inject Code" outcome="Save to Kernel experiments">
                                        <button
                                            onClick={handleInject}
                                            className="px-6 py-2 bg-neon-cyan/10 border border-neon-cyan text-neon-cyan font-bold uppercase tracking-widest rounded hover:bg-neon-cyan hover:text-black transition-all shadow-[0_0_10px_rgba(0,243,255,0.2)]"
                                        >
                                            Inject into Kernel
                                        </button>
                                    </SmartTooltip>
                                </div>
                            </>
                        ) : (
                            <div className="flex-grow flex flex-col items-center justify-center text-slate-600 opacity-50">
                                <SparklesIcon className="w-16 h-16 mb-4" />
                                <p className="text-sm font-mono uppercase tracking-widest">Awaiting Blueprint Generation</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>,
        document.body
    );
};

export default ExperimentalForgeModal;
