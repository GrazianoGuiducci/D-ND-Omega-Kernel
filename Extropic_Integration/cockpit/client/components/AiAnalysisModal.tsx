
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { SparklesIcon } from './icons/SparklesIcon';
import { ArrowPathIcon } from './icons/ArrowPathIcon';
import { MaximizeIcon } from './icons/MaximizeIcon';
import FileUploader from './FileUploader';
import { UploadedFile } from '../types';

interface AiAnalysisModalProps {
    isOpen: boolean;
    onClose: () => void;
    prompt: string;
    setPrompt: (prompt: string) => void;
    response: string;
    isLoading: boolean;
    error: string | null;
    onAsk: (file: UploadedFile | null) => void;
}

const ALL_PROMPT_EXAMPLES = [
    "Analizza questo Bilancio PDF e dimmi se è coerente con il Cash Flow attuale.",
    "Trova i rischi nascosti nei costi del Q3 e suggerisci tagli.",
    "Crea un piano strategico basato sul mio Rating attuale per arrivare ad 'A'.",
    "Simula uno scenario 'Worst Case' con un calo del fatturato del 20%.",
    "Verifica se ci sono bandi PNRR compatibili con i miei investimenti in R&S.",
    "Analizza la varianza tra costi fissi e variabili negli ultimi 6 mesi.",
    "Calcola il DSCR prospettico basandoti sui dati di forecast."
];

const getRandomPrompts = (count: number) => {
    const shuffled = [...ALL_PROMPT_EXAMPLES].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
};

// Directions for resizing
type ResizeDirection = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

const AiAnalysisModal: React.FC<AiAnalysisModalProps> = ({ isOpen, onClose, prompt, setPrompt, response, isLoading, error, onAsk }) => {
    const [attachedFile, setAttachedFile] = useState<UploadedFile | null>(null);
    const [currentExamples, setCurrentExamples] = useState<string[]>([]);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // --- DRAG & RESIZE STATE ---
    const modalRef = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ x: 100, y: 100 });
    const [size, setSize] = useState({ w: 800, h: 600 });
    const [isMaximized, setIsMaximized] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);

    // Center modal initially
    useEffect(() => {
        if (isOpen) {
            const initialX = Math.max(0, (window.innerWidth - 800) / 2);
            const initialY = Math.max(0, (window.innerHeight - 600) / 2);
            setPosition({ x: initialX, y: initialY });
            setCurrentExamples(getRandomPrompts(3));
        }
    }, [isOpen]);

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

    // --- OMNI-DIRECTIONAL RESIZE LOGIC ---
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

            // Horizontal Resizing
            if (direction.includes('e')) {
                newW = Math.max(400, startW + deltaX);
            } else if (direction.includes('w')) {
                newW = Math.max(400, startW - deltaX);
                newX = startPosX + (startW - newW); // Move X to keep right side anchored
            }

            // Vertical Resizing
            if (direction.includes('s')) {
                newH = Math.max(300, startH + deltaY);
            } else if (direction.includes('n')) {
                newH = Math.max(300, startH - deltaY);
                newY = startPosY + (startH - newH); // Move Y to keep bottom anchored
            }

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

    const handleRegeneratePrompts = () => {
        setCurrentExamples(getRandomPrompts(3));
    };

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [prompt]);

    if (!isOpen) return null;

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAsk();
        }
    };

    const handleAsk = () => {
        if (!prompt.trim() && !attachedFile) return;
        onAsk(attachedFile);
    };

    const containerStyle = isMaximized 
        ? { position: 'fixed' as const, top: 0, left: 0, width: '100%', height: '100%', borderRadius: 0 } 
        : { position: 'fixed' as const, top: position.y, left: position.x, width: size.w, height: size.h };

    return (
        // Outer container: Blocks clicks (backdrop) and sits above everything (Z-Index 100)
        <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm pointer-events-auto"> 
            <div
                ref={modalRef}
                className="pointer-events-auto border border-neon-cyan/50 shadow-[0_0_50px_rgba(0,0,0,0.8)] flex flex-col relative overflow-hidden transition-all duration-75 bg-black/90 backdrop-blur-lg rounded-lg"
                style={containerStyle}
            >
                {/* --- INVISIBLE RESIZE HANDLES --- */}
                {!isMaximized && (
                    <>
                        {/* Corners */}
                        <div onMouseDown={handleResizeStart('nw')} className="absolute top-0 left-0 w-4 h-4 cursor-nw-resize z-50"></div>
                        <div onMouseDown={handleResizeStart('ne')} className="absolute top-0 right-0 w-4 h-4 cursor-ne-resize z-50"></div>
                        <div onMouseDown={handleResizeStart('sw')} className="absolute bottom-0 left-0 w-4 h-4 cursor-sw-resize z-50"></div>
                        <div onMouseDown={handleResizeStart('se')} className="absolute bottom-0 right-0 w-6 h-6 cursor-se-resize z-50 flex items-end justify-end p-1 group">
                            {/* Visual Indicator only on Bottom Right */}
                            <div className="w-2 h-2 border-r-2 border-b-2 border-slate-500 group-hover:border-neon-cyan"></div>
                        </div>

                        {/* Sides */}
                        <div onMouseDown={handleResizeStart('n')} className="absolute top-0 left-4 right-4 h-2 cursor-n-resize z-40"></div>
                        <div onMouseDown={handleResizeStart('s')} className="absolute bottom-0 left-4 right-4 h-2 cursor-s-resize z-40"></div>
                        <div onMouseDown={handleResizeStart('w')} className="absolute left-0 top-4 bottom-4 w-2 cursor-w-resize z-40"></div>
                        <div onMouseDown={handleResizeStart('e')} className="absolute right-0 top-4 bottom-4 w-2 cursor-e-resize z-40"></div>
                    </>
                )}

                {/* Header - Draggable */}
                <div 
                    onMouseDown={handleHeaderMouseDown}
                    className={`p-3 bg-gradient-to-r from-black to-slate-900 border-b border-white/10 flex items-center justify-between relative z-20 shrink-0 select-none ${isMaximized ? '' : 'cursor-move'}`}
                >
                    <h2 className="text-sm font-bold flex items-center gap-2 text-white tracking-widest uppercase font-sans">
                        <SparklesIcon className="h-4 w-4 text-neon-cyan" />
                        Neural Interface v2.9 <span className="text-green-500 animate-pulse text-[10px] tracking-normal">● ONLINE</span>
                    </h2>
                    <div className="flex gap-2 items-center" onMouseDown={e => e.stopPropagation()}>
                         <button onClick={() => setIsMaximized(!isMaximized)} className="text-slate-500 hover:text-white transition-colors p-1">
                            <MaximizeIcon className="w-4 h-4" />
                         </button>
                         <button onClick={onClose} className="text-slate-500 hover:text-red-500 transition-colors p-1">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                         </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-grow p-6 overflow-y-auto custom-scrollbar relative z-10">
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-0 pointer-events-none bg-[length:100%_4px,3px_100%]"></div>

                    <div className="relative z-10">
                        {error && <div className="bg-red-900/20 border border-red-500 text-red-400 p-3 rounded mb-4 font-mono text-xs">{`> ERROR: ${error}`}</div>}
                        
                        {isLoading && !response && (
                            <div className="flex flex-col items-center justify-center text-neon-cyan py-12 font-mono">
                               <div className="relative w-12 h-12 mb-4">
                                   <div className="absolute inset-0 border-t-2 border-neon-cyan rounded-full animate-spin"></div>
                                   <div className="absolute inset-2 border-b-2 border-neon-purple rounded-full animate-spin reverse"></div>
                               </div>
                               <p className="animate-pulse text-xs tracking-wider">NEURAL PROCESSING...</p>
                            </div>
                        )}

                        {response && (
                            <div className="prose prose-invert prose-sm max-w-none font-mono text-slate-300 leading-relaxed">
                               <div className="whitespace-pre-wrap">{response}</div>
                            </div>
                        )}

                        {!isLoading && !response && (
                             <div className="space-y-6">
                                 <p className="text-slate-400 font-mono text-sm">
                                    {`> Terminal Ready. Waiting for input stream...`}
                                 </p>
                                 <div className="bg-white/5 p-4 rounded border border-white/10 text-sm">
                                    <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
                                        <strong className="text-neon-cyan font-mono uppercase tracking-wide text-xs">Suggested Protocols</strong>
                                        <button 
                                            onClick={handleRegeneratePrompts}
                                            className="text-slate-400 hover:text-white transition-colors"
                                        >
                                            <ArrowPathIcon className="h-4 w-4" />
                                        </button>
                                    </div>
                                    <ul className="space-y-2">
                                        {currentExamples.map((example, index) => (
                                            <li 
                                                key={index} 
                                                className="flex items-start gap-3 cursor-pointer group p-2 rounded hover:bg-white/5 transition-all border border-transparent hover:border-white/10"
                                                onClick={() => setPrompt(example)}
                                            >
                                                <span className="text-slate-500 font-mono text-xs mt-1">{`0${index+1}`}</span>
                                                <span className="text-slate-300 font-mono text-xs group-hover:text-neon-cyan transition-colors">{example}</span>
                                            </li>
                                        ))}
                                    </ul>
                                 </div>
                             </div>
                        )}
                    </div>
                </div>

                {/* Footer / Input */}
                <div className="p-4 bg-black/50 border-t border-white/10 relative z-20 shrink-0 backdrop-blur-sm">
                    <FileUploader 
                        selectedFile={attachedFile}
                        onFileSelect={setAttachedFile}
                    />
                    <div className="flex gap-3 mt-2 items-end">
                        <div className="flex-grow relative">
                             <textarea
                                ref={textareaRef}
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Enter command..."
                                rows={1}
                                className="w-full bg-slate-900/80 border border-white/20 rounded p-3 text-white font-mono text-sm focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan focus:outline-none resize-none placeholder-slate-600 overflow-hidden min-h-[46px] max-h-[200px]"
                                disabled={isLoading}
                            />
                        </div>
                        
                        <button
                            onClick={handleAsk}
                            disabled={isLoading || (!prompt.trim() && !attachedFile)}
                            className="px-5 py-3 h-auto bg-neon-cyan/20 hover:bg-neon-cyan/30 text-white font-bold rounded uppercase tracking-wider font-sans disabled:opacity-50 transition-all border border-neon-cyan hover:shadow-[0_0_15px_rgba(0,243,255,0.4)] self-end text-xs"
                        >
                            EXEC
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AiAnalysisModal;
