
import React, { useState } from 'react';
import { ExperimentConfig, ExperimentCategory } from '../types';
import { EXPERIMENTS } from '../services/omegaPhysics';
import { generatePhysicsConfig } from '../services/kernelBridge.service';
import { SparklesIcon } from './icons/SparklesIcon';
import { PlusIcon } from './icons/PlusIcon';
import { ArrowPathIcon } from './icons/ArrowPathIcon';
import { CurrencyDollarIcon } from './icons/CurrencyDollarIcon';
import QuantumTooltip from './QuantumTooltip';

interface ExperimentManagerProps {
    activeExpId?: string;
    onSelectExperiment: (exp: ExperimentConfig) => void;
}

const ExperimentManager: React.FC<ExperimentManagerProps> = ({ activeExpId, onSelectExperiment }) => {
    // Local State for Custom Experiments
    const [customExperiments, setCustomExperiments] = useState<ExperimentConfig[]>([]);
    const [isForging, setIsForging] = useState(false);
    const [forgePrompt, setForgePrompt] = useState('');
    const [view, setView] = useState<'LIBRARY' | 'FORGE'>('LIBRARY');
    const [categoryFilter, setCategoryFilter] = useState<ExperimentCategory>('PHYSICS');

    const handleForge = async () => {
        if (!forgePrompt.trim()) return;
        setIsForging(true);
        try {
            // Call AI to generate physics params
            const newExp = await generatePhysicsConfig(forgePrompt);

            // Add to local list and select it
            setCustomExperiments(prev => [newExp, ...prev]);
            onSelectExperiment(newExp);

            setForgePrompt('');
            setView('LIBRARY'); // Go back to list to see result
            setCategoryFilter(newExp.category as ExperimentCategory); // Auto switch tab to show the new item

        } catch (e) {
            console.error(e);
        } finally {
            setIsForging(false);
        }
    };

    const allExperiments = [...customExperiments, ...EXPERIMENTS];
    const filteredExperiments = allExperiments.filter(exp => (exp.category || 'PHYSICS') === categoryFilter);

    return (
        <div className="flex flex-col h-full" style={{ backgroundColor: 'var(--bg-base)' }}>
            {/* Header / Tabs */}
            <div className="p-3 border-b flex flex-col gap-3" style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-muted)' }}>
                <div className="flex justify-between items-center">
                    <div className="text-[9px] font-mono text-slate-600 tracking-widest uppercase">
                        Topology Labs
                    </div>
                    <div className="flex gap-2 bg-black/50 p-1 rounded border border-white/5">
                        <button
                            onClick={() => setView('LIBRARY')}
                            className={`px-3 py-1 text-[9px] font-bold uppercase rounded transition-all ${view === 'LIBRARY' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            Lib
                        </button>
                        <QuantumTooltip
                            title="Theory Forge"
                            desc="Crea esperimenti personalizzati con l'AI."
                            mechanism="L'LLM traduce il linguaggio naturale nei 3 parametri fisici (Gravità, Potenziale, Temperatura)."
                            type="energy"
                        >
                            <button
                                onClick={() => setView('FORGE')}
                                className={`px-3 py-1 text-[9px] font-bold uppercase rounded transition-all flex items-center gap-1 ${view === 'FORGE' ? 'bg-purple-900/40 text-purple-300 border border-purple-500/30' : 'text-slate-500 hover:text-purple-400'}`}
                            >
                                <PlusIcon className="w-3 h-3" /> Forge
                            </button>
                        </QuantumTooltip>
                    </div>
                </div>

                {/* Sub-Tabs for Category (Physics vs Finance) */}
                {view === 'LIBRARY' && (
                    <div className="grid grid-cols-3 w-full border-b gap-px bg-[#1f1f1f]" style={{ borderColor: 'var(--col-muted)', backgroundColor: 'var(--col-muted)' }}>
                        <QuantumTooltip
                            title="Pure Physics Domain"
                            desc="Simulazioni astratte di topologia."
                            mechanism="Reticolo senza asset associati. Focus su transizioni di fase pura."
                            type="info"
                        >
                            <button
                                onClick={() => setCategoryFilter('PHYSICS')}
                                className={`w-full py-2 text-[9px] font-mono uppercase tracking-wider transition-all ${categoryFilter === 'PHYSICS' ? 'bg-opacity-10' : 'bg-[var(--bg-base)] text-slate-600 hover:text-slate-400'}`}
                                style={categoryFilter === 'PHYSICS' ? {
                                    color: 'var(--col-primary)',
                                    backgroundColor: 'rgba(var(--col-primary-rgb), 0.1)'
                                } : {}}
                            >
                                Physics
                            </button>
                        </QuantumTooltip>

                        <QuantumTooltip
                            title="D-ND Concepts"
                            desc="Prototipi concettuali e test teorici."
                            mechanism="Configurazioni sperimentali per validare l'isomorfismo D-ND/Extropic."
                            type="warning"
                        >
                            <button
                                onClick={() => setCategoryFilter('D_ND_CONCEPT')}
                                className={`w-full py-2 text-[9px] font-mono uppercase tracking-wider transition-all ${categoryFilter === 'D_ND_CONCEPT' ? 'bg-opacity-10' : 'bg-[var(--bg-base)] text-slate-600 hover:text-slate-400'}`}
                                style={categoryFilter === 'D_ND_CONCEPT' ? {
                                    color: 'var(--col-cash)',
                                    backgroundColor: 'rgba(250, 204, 21, 0.1)'
                                } : {}}
                            >
                                D-ND Concept
                            </button>
                        </QuantumTooltip>

                        <QuantumTooltip
                            title="Econophysics Domain"
                            desc="Modelli di mercato finanziario."
                            mechanism="I nodi del reticolo sono mappati su Ticker (BTC, AAPL). Lo spin rappresenta Buy/Sell pressure."
                            type="physics"
                        >
                            <button
                                onClick={() => setCategoryFilter('FINANCE')}
                                className={`w-full py-2 text-[9px] font-mono uppercase tracking-wider transition-all flex items-center justify-center gap-1 ${categoryFilter === 'FINANCE' ? 'bg-opacity-10' : 'bg-[var(--bg-base)] text-slate-600 hover:text-slate-400'}`}
                                style={categoryFilter === 'FINANCE' ? {
                                    color: 'var(--col-profit)',
                                    backgroundColor: 'rgba(74, 222, 128, 0.1)'
                                } : {}}
                            >
                                <CurrencyDollarIcon className="w-3 h-3" /> Finance
                            </button>
                        </QuantumTooltip>
                    </div>
                )}
            </div>

            {/* CONTENT AREA */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-3 relative">

                {/* VIEW: FORGE */}
                {view === 'FORGE' && (
                    <div className="space-y-4 animate-slide-up-fade">
                        <div className="bg-purple-900/10 border border-purple-500/30 p-4 rounded-lg">
                            <h4 className="text-[10px] font-bold text-purple-400 uppercase tracking-widest mb-2 flex items-center gap-2">
                                <SparklesIcon className="w-3 h-3" /> Theory Generator
                            </h4>
                            <p className="text-[10px] text-slate-400 mb-4 leading-relaxed">
                                Descrivi un concetto scientifico ("Big Bang") o finanziario ("Crollo dei Bond"). L'IA calcolerà la Gravità Metrica e la Volatilità Termica per simularlo.
                            </p>

                            <textarea
                                value={forgePrompt}
                                onChange={(e) => setForgePrompt(e.target.value)}
                                placeholder='Es: "Entanglement Quantistico" oppure "Recessione Globale"...'
                                className="w-full bg-black/60 border border-white/10 rounded p-3 text-xs text-white font-mono h-24 focus:border-purple-500 outline-none mb-3 resize-none"
                            />

                            <button
                                onClick={handleForge}
                                disabled={isForging || !forgePrompt}
                                className="w-full py-2 bg-purple-600 hover:bg-purple-500 text-white text-[10px] font-bold uppercase tracking-widest rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all shadow-[0_0_15px_rgba(168,85,247,0.3)]"
                            >
                                {isForging ? <ArrowPathIcon className="w-3 h-3 animate-spin" /> : <SparklesIcon className="w-3 h-3" />}
                                {isForging ? 'Calculating...' : 'Ignite Forge'}
                            </button>
                        </div>

                        <div className="text-[9px] text-slate-600 font-mono text-center">
                            AI-Driven Metric Tensor Calculation
                        </div>
                    </div>
                )}

                {/* VIEW: LIBRARY */}
                {view === 'LIBRARY' && (
                    <div className="space-y-3">
                        {filteredExperiments.map(exp => {
                            const isActive = activeExpId === exp.id;
                            const isCustom = exp.id.startsWith('custom_');
                            const isFinance = exp.category === 'FINANCE';

                            // Dynamic Styles based on Category
                            const activeBorder = isFinance ? 'border-green-500/50' : 'border-purple-500/50';
                            const activeBg = isFinance ? 'bg-green-900/10' : 'bg-purple-900/10';
                            const activeText = isFinance ? 'text-green-400' : 'text-purple-400';
                            const aiBadgeBg = isFinance ? 'bg-green-500' : 'bg-purple-500';

                            return (
                                <button
                                    key={exp.id}
                                    onClick={() => onSelectExperiment(exp)}
                                    className={`
                                        w-full text-left p-4 rounded border transition-all relative overflow-hidden group
                                        ${isActive
                                            ? `${activeBg} ${activeBorder} text-white shadow-[0_0_15px_rgba(0,0,0,0.3)]`
                                            : 'bg-[#050505] border-[#1f1f1f] text-slate-400 hover:border-slate-600 hover:text-slate-200'}
                                    `}
                                >
                                    {/* Active Indicator Strip */}
                                    {isActive && <div className={`absolute left-0 top-0 bottom-0 w-0.5 ${isFinance ? 'bg-green-500' : 'bg-purple-500'}`}></div>}

                                    <div className="flex justify-between items-center mb-2">
                                        <div className="flex items-center gap-2">
                                            {isCustom && <span className={`text-[8px] ${aiBadgeBg} text-black px-1 rounded font-bold uppercase`}>AI</span>}
                                            <span className="text-xs font-bold uppercase font-mono tracking-wider truncate max-w-[150px]">{exp.name}</span>
                                        </div>
                                        {isActive && <span className={`text-[9px] font-mono ${activeText} animate-pulse`}>Running</span>}
                                    </div>

                                    <p className="text-[10px] text-slate-500 leading-relaxed mb-3 border-b border-white/5 pb-2">
                                        {exp.description}
                                    </p>

                                    {/* Parameters Grid */}
                                    <div className="grid grid-cols-3 gap-2 text-[9px] font-mono text-slate-600 uppercase">
                                        <div className="bg-black/40 p-1.5 rounded border border-white/5 flex flex-col items-center">
                                            <span className="text-slate-700">{isFinance ? 'Corr.' : 'Gravity'}</span>
                                            <span className="text-slate-300">{exp.params.gravity.toFixed(1)}</span>
                                        </div>
                                        <div className="bg-black/40 p-1.5 rounded border border-white/5 flex flex-col items-center">
                                            <span className="text-slate-700">{isFinance ? 'Bias' : 'Potential'}</span>
                                            <span className="text-slate-300">{exp.params.potentialScale.toFixed(2)}</span>
                                        </div>
                                        <div className="bg-black/40 p-1.5 rounded border border-white/5 flex flex-col items-center">
                                            <span className="text-slate-700">{isFinance ? 'VIX' : 'Noise'}</span>
                                            <span className="text-slate-300">{exp.params.temperature.toFixed(1)}</span>
                                        </div>
                                    </div>
                                </button>
                            );
                        })}
                        {filteredExperiments.length === 0 && (
                            <div className="text-center py-8 opacity-50">
                                <p className="text-[9px] text-slate-600 font-mono uppercase">No Experiments Found</p>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="p-3 border-t border-[#1f1f1f] bg-[#050505] text-[9px] font-mono text-slate-700 text-center">
                OMEGA_PHYSICS_ENGINE v2.2 // GENERATIVE_MODE
            </div>
        </div>
    );
};

export default ExperimentManager;
