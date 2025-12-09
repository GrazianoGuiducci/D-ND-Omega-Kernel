
import React from 'react';
import { SparklesIcon } from './icons/SparklesIcon';

interface AiExecutiveSummaryCardProps {
    analysis: string;
    onClose: () => void;
}

const AiExecutiveSummaryCard: React.FC<AiExecutiveSummaryCardProps> = ({ analysis, onClose }) => {
    if (!analysis) return null;

    return (
        <div className="relative group mt-2 mb-8">
            {/* Glowing border effect */}
            <div className="absolute -inset-[1px] bg-gradient-to-r from-neon-cyan via-blue-500 to-neon-purple rounded-lg opacity-50 blur-sm group-hover:opacity-75 transition duration-1000"></div>
            
            <div className="relative bg-obsidian border border-neon-cyan/30 rounded-lg p-6 overflow-hidden shadow-neon transition-colors duration-300">
                {/* Grid Background inside card */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_2px,transparent_2px),linear-gradient(90deg,rgba(255,255,255,0.05)_2px,transparent_2px)] bg-[length:20px_20px] pointer-events-none"></div>
                
                <div className="relative z-10 flex items-start gap-5">
                    <div className="bg-gradient-to-br from-neon-cyan/20 to-neon-purple/20 p-3 rounded border border-neon-cyan/30 text-neon-cyan shadow-neon shrink-0">
                        <SparklesIcon className="h-6 w-6" />
                    </div>
                    
                    <div className="flex-grow">
                        <div className="flex justify-between items-start mb-4 border-b border-white/10 pb-2">
                            <div>
                                <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 uppercase tracking-widest font-sans">
                                    Executive Intelligence
                                </h3>
                                <p className="text-[10px] font-mono text-neon-cyan uppercase tracking-wider mt-1">
                                    // AI Projection Model: GEMINI-3-PRO // Status: OPTIMAL
                                </p>
                            </div>
                            <button 
                                onClick={onClose}
                                className="text-xs font-mono text-slate-500 hover:text-red-400 transition-colors border border-white/10 hover:border-red-900/50 bg-black/20 px-3 py-1 rounded uppercase"
                            >
                                [x] Dismiss
                            </button>
                        </div>
                        
                        <div className="prose prose-invert prose-sm max-w-none text-slate-300 font-mono leading-relaxed">
                             <div className="whitespace-pre-wrap">{analysis}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AiExecutiveSummaryCard;
