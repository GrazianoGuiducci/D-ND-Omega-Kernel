import React, { useState, useEffect, useRef, useCallback } from 'react';
import ControlMatrix from './ControlMatrix';
import VisualCortex from './VisualCortex';
import ExperimentManager from './ExperimentManager';
import MetricTensorVis from './MetricTensorVis';
import DidacticLayer from './DidacticLayer';
import { OmegaPhysicsEngine, EXPERIMENTS, translateIntentToHamiltonian } from '../services/omegaPhysics';
import { generateDslTrace } from '../services/omegaDSL';
import { OmegaState } from '../types';
import { getKernelIntent } from '../services/kernelBridge.service';
import QuantumTooltip from './QuantumTooltip';
import SmartTooltip from './SmartTooltip';
import { Resizer } from './Resizer';
import EnergyGraph from './EnergyGraph';

interface OmegaCockpitProps {
    onClose: () => void;
    onOpenDocs: () => void;
    initialExperimentId?: string;
    showTooltips?: boolean;
}

type ViewMode = 'CORTEX' | 'TENSOR';

const OmegaCockpit: React.FC<OmegaCockpitProps> = ({ onClose, onOpenDocs, initialExperimentId, showTooltips = true }) => {
    // Physics Engine Ref (Local Simulation for 60fps FX)
    const engineRef = useRef<OmegaPhysicsEngine>(new OmegaPhysicsEngine());

    // UI Layout State
    // UI Layout State
    // Default to 33% width for each panel
    const [leftPanelWidth, setLeftPanelWidth] = useState(window.innerWidth / 3);
    const [rightPanelWidth, setRightPanelWidth] = useState(window.innerWidth / 3);
    const [topPanelHeight, setTopPanelHeight] = useState(450); // Increased default height for better fit
    const [isResizingLeft, setIsResizingLeft] = useState(false);
    const [isResizingRight, setIsResizingRight] = useState(false);
    const [isResizingVertical, setIsResizingVertical] = useState(false); // New state for vertical resize

    // UI Logic State
    const [state, setState] = useState<OmegaState & { energyHistory: number[] }>({
        lattice: engineRef.current.nodes,
        metrics: engineRef.current.metrics,
        phase: 'IDLE',
        dslTrace: [],
        activeExperimentId: initialExperimentId || 'flat_space',
        tensorMatrix: [],
        energyHistory: [] // Initialize
    });

    const [viewMode, setViewMode] = useState<ViewMode>('CORTEX');
    const [prompt, setPrompt] = useState('');
    const [result, setResult] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    const activeExperiment = EXPERIMENTS.find(e => e.id === state.activeExperimentId);
    const isFinance = activeExperiment?.category === 'FINANCE';

    // Initialize Engine with Initial Experiment
    useEffect(() => {
        if (initialExperimentId) {
            engineRef.current.loadExperiment(initialExperimentId);
            const exp = EXPERIMENTS.find(e => e.id === initialExperimentId);
            if (exp?.intent) {
                setPrompt(exp.intent);
                // Trigger initial injection after a short delay to allow UI to settle
                setTimeout(() => handleInject(exp.intent), 500);
            }
        }
    }, []); // Run once on mount

    // --- SYNC WITH PYTHON BACKEND ---
    useEffect(() => {
        const syncInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/state');
                if (!response.ok) return;
                const data = await response.json();

                // Update Physics Engine Parameters based on Backend State
                if (data.metrics) {
                    // Map Python metrics to JS Physics
                    // Python 'temperature' -> JS 'currentTemp'
                    if (data.metrics.temperature !== undefined) {
                        // Smooth transition or direct set? Direct for responsiveness.
                        engineRef.current.currentTemp = data.metrics.temperature;
                    }

                    // Python 'gravity' (from didactic) -> JS 'gravity'
                    if (data.didactic && data.didactic.gravity !== undefined) {
                        engineRef.current.gravity = data.didactic.gravity;
                    }
                }

                // If backend sends a specific lattice configuration, we could update nodes here.
                // For now, we let the local physics evolve based on the parameters.

            } catch (e) {
                console.warn("Backend sync failed", e);
            }
        }, 1000); // Poll every second

        return () => clearInterval(syncInterval);
    }, []);

    // Animation Loop (Local Physics)
    useEffect(() => {
        let animationFrameId: number;
        const loop = () => {
            const { nodes, metrics, tensorMap } = engineRef.current.step();
            let currentPhase: any = 'IDLE';
            if (metrics.temperature > 1.5) currentPhase = 'PERTURBATION';
            else if (metrics.temperature > 0.5) currentPhase = 'ANNEALING';
            else if (metrics.coherence > 0.8) currentPhase = 'CRYSTALLIZED';

            setState(prev => {
                // Update Energy History (Keep last 100 points)
                const newHistory = [...prev.energyHistory, metrics.energy];
                if (newHistory.length > 100) newHistory.shift();

                return {
                    ...prev,
                    lattice: [...nodes],
                    metrics: { ...metrics },
                    phase: currentPhase,
                    tensorMatrix: tensorMap,
                    energyHistory: newHistory
                };
            });
            animationFrameId = requestAnimationFrame(loop);
        };
        loop();
        return () => cancelAnimationFrame(animationFrameId);
    }, []);

    const animateDslTrace = (trace: any[]) => {
        let stepIndex = 0;
        const interval = setInterval(() => {
            if (stepIndex >= trace.length) {
                clearInterval(interval);
                return;
            }
            setState(prev => {
                const newTrace = [...trace];
                newTrace.forEach((s, i) => {
                    if (i < stepIndex) s.status = 'complete';
                    else if (i === stepIndex) s.status = 'active';
                    else s.status = 'pending';
                });
                return { ...prev, dslTrace: newTrace };
            });
            stepIndex++;
        }, 800);
    };

    const handleInject = async (intentOverride?: string) => {
        const targetIntent = intentOverride || prompt;
        if (!targetIntent) return;

        setIsProcessing(true);
        setResult(null);

        // 1. SEMANTIC BRIDGE: Translate Intent -> Physics
        const physicsParams = translateIntentToHamiltonian(targetIntent);

        // 2. APPLY TO KERNEL (Runtime Injection)
        engineRef.current.gravity = physicsParams.gravity;
        engineRef.current.currentTemp = physicsParams.temperature;
        engineRef.current.potentialScale = physicsParams.potential;

        // Perturb system to initiate transition
        engineRef.current.perturb(physicsParams.temperature * 0.5);

        // 3. VISUAL FEEDBACK (DSL Trace)
        const trace = generateDslTrace(targetIntent);
        animateDslTrace(trace);

        try {
            // Call Python Backend via Service
            // Note: getAiAnalysis signature might need checking if it supports "OMEGA_KERNEL_MODE" correctly
            // or if we should call a specific endpoint for intents like in vanilla.
            // Vanilla used /api/intent. React uses getAiAnalysis which calls /api/analyze (likely).
            // We should probably add a specific service method for /api/intent or ensure getAiAnalysis handles it.
            // For now, assuming getAiAnalysis is the gateway.

            const aiResponse = await getKernelIntent(
                targetIntent,
                "OMEGA_KERNEL_MODE"
            );
            setResult(aiResponse.analysis);
        } catch (e) {
            setResult("ERROR: KERNEL CONNECTION FAILED.");
        } finally {
            setIsProcessing(false);
        }
    };

    // Resizing Logic for Left Panel (Horizontal)
    useEffect(() => {
        if (!isResizingLeft) return;
        const onMouseMove = (e: MouseEvent) => {
            setLeftPanelWidth(Math.max(250, Math.min(e.clientX, 600)));
        };
        const onMouseUp = () => setIsResizingLeft(false);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        return () => { document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); };
    }, [isResizingLeft]);

    // Resizing Logic for Left Panel (Vertical Split)
    useEffect(() => {
        if (!isResizingVertical) return;
        const onMouseMove = (e: MouseEvent) => {
            // Constrain height between 200px and window height - 200px
            setTopPanelHeight(Math.max(200, Math.min(e.clientY, window.innerHeight - 200)));
        };
        const onMouseUp = () => setIsResizingVertical(false);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        return () => { document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); };
    }, [isResizingVertical]);

    // Resizing Logic for Right Panel
    useEffect(() => {
        if (!isResizingRight) return;
        const onMouseMove = (e: MouseEvent) => {
            setRightPanelWidth(Math.max(250, Math.min(window.innerWidth - e.clientX, 600)));
        };
        const onMouseUp = () => setIsResizingRight(false);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        return () => { document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); };
    }, [isResizingRight]);


    // ... (existing functions)

    return (
        <div className="fixed inset-0 z-[100] font-sans flex overflow-hidden selection:bg-cyan-500/30" style={{ backgroundColor: 'var(--bg-base)', color: 'var(--text-sub)' }}>

            {/* COLUMN 1: CONTROL & EXPERIMENTS */}
            <div
                className="shrink-0 relative z-20 border-r border-white/10 flex flex-col shadow-[10px_0_30px_rgba(0,0,0,0.5)] transition-all duration-75 ease-linear"
                style={{ width: leftPanelWidth, backgroundColor: 'var(--bg-base)', borderColor: 'var(--col-muted)' }}
            >
                {/* TOP: CONTROL MATRIX */}
                <div className="border-b border-white/10 overflow-y-auto" style={{ height: topPanelHeight, borderColor: 'var(--col-muted)' }}>
                    <SmartTooltip
                        source="Thermostat"
                        logic="Noise Injection"
                        outcome="State Perturbation"
                        isVisible={showTooltips}
                        position="right"
                        className="block w-full h-full"
                    >
                        <div className="h-full">
                            <ControlMatrix
                                temp={state.metrics.temperature}
                                setTemp={(t) => engineRef.current.currentTemp = t}
                                prompt={prompt}
                                setPrompt={setPrompt}
                                onInject={handleInject}
                                isProcessing={isProcessing}
                            />
                        </div>
                    </SmartTooltip>
                </div>

                {/* VERTICAL RESIZER */}
                <div
                    className="h-1 w-full cursor-row-resize bg-white/5 hover:bg-neon-cyan transition-colors z-50 flex items-center justify-center group"
                    onMouseDown={(e) => { e.preventDefault(); setIsResizingVertical(true); }}
                >
                    <div className="w-8 h-0.5 bg-slate-600 group-hover:bg-white rounded-full"></div>
                </div>

                {/* BOTTOM: EXPERIMENTS */}
                <div className="flex-1 overflow-hidden" style={{ backgroundColor: 'var(--bg-surface)' }}>
                    <ExperimentManager
                        activeExpId={state.activeExperimentId}
                        onSelectExperiment={(exp) => {
                            engineRef.current.loadExperiment(exp.id);
                            setState(prev => ({ ...prev, activeExperimentId: exp.id }));

                            // AUTO-INJECT IF PROTOCOL HAS INTENT
                            if (exp.intent) {
                                setPrompt(exp.intent);
                                handleInject(exp.intent);
                            }
                        }}
                    />
                </div>
            </div>

            {/* RESIZER LEFT */}
            <div className="relative z-30">
                <Resizer onMouseDown={(e) => { e.preventDefault(); setIsResizingLeft(true); }} isVisible={true} />
            </div>

            {/* COLUMN 2: COMBINATORIAL MECHANICS (Middle) */}
            <div className="flex-1 relative z-10 flex flex-col min-w-0" style={{ backgroundColor: 'var(--bg-base)' }}>
                {/* HUD Header */}
                <div className="h-12 border-b border-white/10 flex justify-between items-center px-6" style={{ backgroundColor: 'var(--bg-base)', borderColor: 'var(--col-muted)' }}>
                    <div className="flex gap-6">
                        <SmartTooltip source="View Selector" logic="Particle Render" outcome="Micro-State Vis" isVisible={showTooltips} position="bottom">
                            <button
                                onClick={() => setViewMode('CORTEX')}
                                className={`text-[10px] font-mono tracking-[0.2em] transition-all h-12 border-b-2 px-2 ${viewMode === 'CORTEX' ? 'text-cyan-400 border-cyan-500 bg-cyan-900/10' : 'text-slate-600 border-transparent hover:text-slate-400'}`}
                            >
                                VISUAL_CORTEX
                            </button>
                        </SmartTooltip>

                        <SmartTooltip source="View Selector" logic="Field Topology" outcome="Macro-State Vis" isVisible={showTooltips} position="bottom">
                            <button
                                onClick={() => setViewMode('TENSOR')}
                                className={`text-[10px] font-mono tracking-[0.2em] transition-all h-12 border-b-2 px-2 ${viewMode === 'TENSOR' ? 'text-purple-400 border-purple-500 bg-purple-900/10' : 'text-slate-600 border-transparent hover:text-slate-400'}`}
                            >
                                METRIC_TENSOR
                            </button>
                        </SmartTooltip>
                    </div>

                    <div className="flex gap-4 text-[10px] font-mono text-slate-600 items-center">
                        <SmartTooltip source="Hamiltonian" logic="Time Integration" outcome="Stability Trend" isVisible={showTooltips} position="bottom">
                            <div className="w-32 h-8">
                                <EnergyGraph data={state.energyHistory || []} />
                            </div>
                        </SmartTooltip>

                        <SmartTooltip source="Metric Tensor" logic="Curvature Calc" outcome="Attractor Strength" isVisible={showTooltips} position="bottom">
                            <span className={`cursor-help border-b border-dashed border-slate-800 ${state.metrics.curvature > 1 ? "text-purple-400 animate-pulse" : ""}`}>
                                {isFinance ? 'CORRELATION:' : 'GRAVITY:'} {state.metrics.curvature.toFixed(2)}
                            </span>
                        </SmartTooltip>

                        <SmartTooltip source="Phase Space" logic="Shannon Entropy" outcome="Disorder Measure" isVisible={showTooltips} position="bottom">
                            <span className="cursor-help border-b border-dashed border-slate-800">
                                {isFinance ? 'VOLATILITY:' : 'ENTROPY:'} {state.metrics.entropy.toFixed(3)}
                            </span>
                        </SmartTooltip>
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={onOpenDocs}
                            className="p-1 hover:text-white transition-colors text-[10px] font-mono tracking-widest border border-transparent hover:border-white/20 px-2 rounded text-slate-500"
                        >
                            [GUIDE.SYS]
                        </button>
                        <button
                            onClick={onClose}
                            className="p-1 hover:text-red-500 transition-colors text-[10px] font-mono tracking-widest border border-transparent hover:border-red-900 px-2 rounded"
                        >
                            [EXIT_SYSTEM]
                        </button>
                    </div>
                </div>

                {/* Main Visualizer Area */}
                <div className="flex-1 relative p-4 overflow-hidden" style={{ backgroundColor: 'var(--bg-base)' }}>
                    {viewMode === 'CORTEX' ? (
                        <VisualCortex nodes={state.lattice} phase={state.phase} />
                    ) : (
                        <MetricTensorVis tensorMap={state.tensorMatrix || []} gravity={state.metrics.curvature} />
                    )}

                    {/* Floating HUD */}
                    <div className="absolute bottom-8 left-8 flex gap-4 pointer-events-none">
                        <div className={`bg-black/60 backdrop-blur border ${isFinance ? 'border-green-500/20' : 'border-white/5'} p-3 rounded min-w-[100px] shadow-lg`}>
                            <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">{isFinance ? 'Bias' : 'Potential'}</div>
                            <div className="text-lg font-mono text-white">{(engineRef.current.potentialScale * 100).toFixed(0)}%</div>
                        </div>
                        <div className={`bg-black/60 backdrop-blur border ${isFinance ? 'border-green-500/20' : 'border-white/5'} p-3 rounded min-w-[100px] shadow-lg`}>
                            <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Coherence</div>
                            <div className={`text-lg font-mono ${state.metrics.coherence > 0.8 ? 'text-cyan-400' : 'text-slate-500'}`}>
                                {(state.metrics.coherence * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                </div>

                {/* Result Output (Bottom) */}
                {result && (
                    <div className="h-48 border-t backdrop-blur-sm p-4 overflow-y-auto font-mono text-xs shadow-inner animate-slide-up-fade relative z-20" style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-muted)', color: 'var(--text-sub)' }}>
                        <div className="mb-2 uppercase tracking-widest text-[9px]" style={{ color: 'var(--col-primary)' }}>{'>>'} MANIFESTATION_OUTPUT</div>
                        <div className="whitespace-pre-wrap">{result}</div>
                    </div>
                )}
            </div>

            {/* RESIZER RIGHT */}
            <div className="relative z-30">
                <Resizer onMouseDown={(e) => { e.preventDefault(); setIsResizingRight(true); }} isVisible={true} />
            </div>

            {/* COLUMN 3: DIDACTIC LAYER */}
            <div
                className="shrink-0 border-l border-white/10 flex flex-col relative z-20 shadow-[-10px_0_30px_rgba(0,0,0,0.5)] transition-all duration-75 ease-linear"
                style={{ width: rightPanelWidth, backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-muted)' }}
            >
                <DidacticLayer trace={state.dslTrace} onClose={onClose} />
            </div>
        </div>
    );
};

export default OmegaCockpit;
