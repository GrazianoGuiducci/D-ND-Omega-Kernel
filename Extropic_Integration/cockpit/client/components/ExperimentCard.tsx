import React from 'react';
import { ExperimentConfig } from '../types';
import { SparklesIcon } from './icons/SparklesIcon';

interface ExperimentCardProps {
    config: ExperimentConfig;
    onLaunch: (id: string) => void;
}

const ExperimentCard: React.FC<ExperimentCardProps> = ({ config, onLaunch }) => {
    const isConcept = config.status === 'CONCEPT';
    const isComingSoon = config.status === 'COMING_SOON';
    // Allow launching CONCEPT protocols as well
    const isActive = config.status === 'ACTIVE' || config.status === 'CONCEPT' || !config.status;

    return (
        <div className={`
            group relative overflow-hidden rounded-xl border transition-all duration-300
            ${isActive
                ? 'hover:shadow-[0_0_30px_rgba(6,182,212,0.15)]'
                : 'opacity-80 hover:opacity-100'
            }
        `} style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-muted)' }}>
            {/* Background Gradient */}
            <div className={`absolute inset-0 bg-gradient-to-br ${isActive ? 'from-cyan-900/10 to-purple-900/10' : 'from-slate-900/10 to-black'} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>

            <div className="p-6 relative z-10 flex flex-col h-full">
                {/* Header */}
                <div className="flex justify-between items-start mb-4">
                    <div className="flex flex-col">
                        <span className={`text-[10px] font-mono uppercase tracking-widest mb-1 ${isActive ? 'text-neon-cyan' : 'text-slate-500'}`}>
                            {config.category}
                        </span>
                        <h3 className={`text-lg font-bold font-sans ${isActive ? 'text-white' : 'text-slate-400'}`}>
                            {config.name}
                        </h3>
                    </div>
                    {/* Status Badge */}
                    <div className={`px-2 py-1 rounded text-[9px] font-mono font-bold uppercase border ${isActive ? 'border-green-500/30 text-green-400 bg-green-900/20' :
                        isConcept ? 'border-purple-500/30 text-purple-400 bg-purple-900/20' :
                            'border-slate-700 text-slate-500 bg-slate-900'
                        }`}>
                        {config.status || 'ACTIVE'}
                    </div>
                </div>

                {/* Description */}
                <p className="text-xs text-slate-400 font-mono leading-relaxed mb-6 flex-grow">
                    {config.description}
                </p>

                {/* Specs Grid */}
                <div className="grid grid-cols-3 gap-2 mb-6 border-t pt-4" style={{ borderColor: 'var(--col-muted)' }}>
                    <div>
                        <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-sub)' }}>Gravity</div>
                        <div className="text-xs font-mono" style={{ color: 'var(--text-main)' }}>{config.params.gravity.toFixed(1)}</div>
                    </div>
                    <div>
                        <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-sub)' }}>Potential</div>
                        <div className="text-xs font-mono" style={{ color: 'var(--text-main)' }}>{config.params.potentialScale.toFixed(1)}</div>
                    </div>
                    <div>
                        <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-sub)' }}>Temp</div>
                        <div className="text-xs font-mono" style={{ color: 'var(--text-main)' }}>{config.params.temperature.toFixed(1)}K</div>
                    </div>
                </div>

                {/* Action Button */}
                <button
                    onClick={() => isActive && onLaunch(config.id)}
                    disabled={!isActive}
                    className={`
                        w-full py-3 px-4 rounded-lg font-bold text-xs tracking-widest uppercase transition-all flex items-center justify-center gap-2
                        ${isActive
                            ? 'bg-white/5 hover:bg-neon-cyan hover:text-black text-white border border-white/10 hover:border-neon-cyan shadow-lg'
                            : 'bg-white/5 text-slate-600 cursor-not-allowed border border-transparent'
                        }
                    `}
                >
                    {isActive ? (
                        <>
                            <SparklesIcon className="w-4 h-4" />
                            {isConcept ? 'Launch Concept' : 'Launch Protocol'}
                        </>
                    ) : (
                        <span>{isComingSoon ? 'Coming Soon' : 'Locked'}</span>
                    )}
                </button>
            </div>
        </div>
    );
};

export default ExperimentCard;
