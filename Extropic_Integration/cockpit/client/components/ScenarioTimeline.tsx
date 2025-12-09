
import React, { useState } from 'react';
import { Scenario } from '../types';
import { SparklesIcon } from './icons/SparklesIcon';
import { PlusIcon } from './icons/PlusIcon';
import { TrashIcon } from './icons/TrashIcon';
import { ArrowPathIcon } from './icons/ArrowPathIcon';

interface ScenarioTimelineProps {
    scenarios: Scenario[];
    activeScenarioId: string | null;
    onSaveScenario: (name: string) => void;
    onLoadScenario: (id: string) => void;
    onDeleteScenario: (id: string) => void;
}

const ScenarioTimeline: React.FC<ScenarioTimelineProps> = ({ scenarios, activeScenarioId, onSaveScenario, onLoadScenario, onDeleteScenario }) => {
    const [isNaming, setIsNaming] = useState(false);
    const [newName, setNewName] = useState('');

    const handleSave = () => {
        if (!newName.trim()) return;
        onSaveScenario(newName);
        setNewName('');
        setIsNaming(false);
    };

    return (
        <div className="flex items-center gap-2 overflow-x-auto custom-scrollbar pb-2">
            <div className="flex items-center gap-2 p-1 bg-black/40 border border-white/10 rounded-lg backdrop-blur-sm mr-2 shrink-0">
                <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest px-2 font-mono">
                    Time Machine
                </span>
            </div>

            {/* BASELINE (Live) */}
            <button
                onClick={() => onLoadScenario('live')}
                className={`
                    flex items-center gap-2 px-3 py-1.5 rounded border text-xs font-mono transition-all shrink-0
                    ${activeScenarioId === null 
                        ? 'bg-neon-cyan/20 border-neon-cyan text-neon-cyan shadow-[0_0_10px_rgba(0,243,255,0.3)]' 
                        : 'bg-black/40 border-white/10 text-slate-400 hover:text-white hover:border-white/30'}
                `}
            >
                <div className={`w-2 h-2 rounded-full ${activeScenarioId === null ? 'bg-neon-cyan animate-pulse' : 'bg-slate-600'}`}></div>
                <span className="uppercase font-bold">Live State</span>
            </button>

            {/* SAVED SCENARIOS */}
            {scenarios.map(scenario => (
                <div 
                    key={scenario.id}
                    className={`
                        group flex items-center gap-2 px-3 py-1.5 rounded border text-xs font-mono transition-all shrink-0 relative pr-8 cursor-pointer
                        ${activeScenarioId === scenario.id 
                            ? 'bg-purple-900/40 border-purple-500 text-purple-300 shadow-[0_0_10px_rgba(188,19,254,0.3)]' 
                            : 'bg-black/40 border-white/10 text-slate-400 hover:text-white hover:border-white/30'}
                    `}
                    onClick={() => onLoadScenario(scenario.id)}
                >
                    <div className="flex flex-col items-start leading-none gap-0.5">
                        <span className="font-bold uppercase">{scenario.name}</span>
                        <span className="text-[8px] opacity-60">{new Date(scenario.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                    </div>
                    
                    <button 
                        onClick={(e) => { e.stopPropagation(); onDeleteScenario(scenario.id); }}
                        className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                        <TrashIcon className="w-3 h-3" />
                    </button>
                </div>
            ))}

            {/* ADD BUTTON */}
            {isNaming ? (
                <div className="flex items-center gap-1 bg-black/60 border border-neon-cyan/50 rounded px-1 py-0.5 animate-fadeIn">
                    <input 
                        autoFocus
                        type="text" 
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSave()}
                        placeholder="Scenario Name..."
                        className="bg-transparent border-none outline-none text-white text-xs font-mono w-32 px-1"
                    />
                    <button onClick={handleSave} className="text-neon-cyan hover:text-white"><PlusIcon className="w-4 h-4" /></button>
                </div>
            ) : (
                <button 
                    onClick={() => setIsNaming(true)}
                    className="p-1.5 rounded border border-dashed border-slate-700 text-slate-500 hover:text-neon-cyan hover:border-neon-cyan transition-colors"
                    title="Capture Current State"
                >
                    <PlusIcon className="w-4 h-4" />
                </button>
            )}
        </div>
    );
};

export default ScenarioTimeline;
