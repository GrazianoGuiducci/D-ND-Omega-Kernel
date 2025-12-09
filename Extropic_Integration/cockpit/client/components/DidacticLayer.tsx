
import React, { useEffect, useRef } from 'react';
import { DslStep } from '../types';

interface DidacticLayerProps {
    trace: DslStep[];
    onClose: () => void;
}

const DidacticLayer: React.FC<DidacticLayerProps> = ({ trace, onClose }) => {
    // Scroll to bottom when trace updates
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [trace]);

    return (
        <div className="flex flex-col h-full font-sans" style={{ backgroundColor: 'var(--bg-surface)', color: 'var(--text-sub)' }}>
            {/* Header */}
            <div className="p-4 border-b flex justify-between items-center shrink-0" style={{ backgroundColor: 'var(--bg-base)', borderColor: 'var(--col-muted)' }}>
                <div>
                    <h3 className="text-xs font-bold uppercase tracking-widest font-mono" style={{ color: 'var(--col-primary)' }}>
                        OMEGA_DSL_TRACE
                    </h3>
                    <p className="text-[9px] font-mono mt-1" style={{ color: 'var(--text-sub)' }}>
                        Inferential Logic & Semantic Parsing
                    </p>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 hover:bg-white/10 rounded transition-colors"
                    style={{ color: 'var(--text-sub)' }}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            {/* Terminal Output Area */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar font-mono text-xs space-y-4">
                {trace.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center opacity-50" style={{ color: 'var(--text-sub)' }}>
                        <div className="w-12 h-12 border rounded-full flex items-center justify-center mb-4" style={{ borderColor: 'var(--col-muted)' }}>
                            <span className="animate-pulse">_</span>
                        </div>
                        <p className="text-[10px] uppercase tracking-widest">Waiting for Injection</p>
                    </div>
                ) : (
                    trace.map((step) => {
                        const isActive = step.status === 'active';
                        const isComplete = step.status === 'complete';
                        const statusColor = isActive ? 'text-green-400' : isComplete ? 'text-slate-400' : 'text-slate-600';
                        // Keep specific green for active state, but use theme for base
                        const borderColor = isActive ? 'border-green-500/50' : 'border-transparent';
                        const bgColor = isActive ? 'bg-green-900/10' : 'bg-black/20'; // Use semi-transparent black for contrast against surface

                        return (
                            <div
                                key={step.id}
                                className={`border ${borderColor} ${bgColor} p-3 rounded transition-all duration-300 relative overflow-hidden`}
                                style={{ borderColor: !isActive ? 'var(--col-muted)' : undefined }}
                            >
                                {isActive && (
                                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-green-500 animate-pulse"></div>
                                )}

                                <div className="flex justify-between items-start mb-2">
                                    <span className={`text-[10px] uppercase font-bold tracking-wider ${statusColor}`}>
                                        {step.id < 10 ? `0${step.id}` : step.id} // {step.label}
                                    </span>
                                    {isActive && <span className="text-[9px] text-green-500 animate-pulse">PROCESSING</span>}
                                    {isComplete && <span className="text-[9px]" style={{ color: 'var(--text-sub)' }}>OK</span>}
                                </div>

                                <div className="font-bold mb-2 pl-2 border-l" style={{ color: 'var(--text-main)', borderColor: 'var(--col-muted)' }}>
                                    {step.code}
                                </div>

                                <div className="text-[10px] italic p-2 rounded" style={{ backgroundColor: 'var(--bg-base)', color: 'var(--text-sub)' }}>
                                    "{step.rosetta}"
                                </div>
                            </div>
                        );
                    })
                )}
                <div ref={bottomRef} />
            </div>

            {/* Footer */}
            <div className="p-3 border-t flex justify-between text-[9px] font-mono" style={{ backgroundColor: 'var(--bg-base)', borderColor: 'var(--col-muted)', color: 'var(--text-sub)' }}>
                <span>MEM_USAGE: 24MB</span>
                <span>LATENCY: 12ms</span>
            </div>
        </div>
    );
};

export default DidacticLayer;
