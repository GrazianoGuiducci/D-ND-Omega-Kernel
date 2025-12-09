
import React, { useState } from 'react';
import { SparklesIcon } from './icons/SparklesIcon';
import { ArrowsXYIcon } from './icons/ArrowsXYIcon';
import { TrashIcon } from './icons/TrashIcon';
import { WidgetColorTheme } from '../types';
import { HOLOGRAM_CONFIG } from '../themes';

interface CardProps {
    title: string;
    icon: React.ReactNode;
    children: React.ReactNode;
    className?: string;
    onAnalysisClick?: () => void;
    onToggleSize?: () => void;
    onDelete?: () => void;
    isFullWidth?: boolean;
    colorTheme?: WidgetColorTheme; // New Prop
}

const Card: React.FC<CardProps> = ({ title, icon, children, className = '', onAnalysisClick, onToggleSize, onDelete, isFullWidth, colorTheme }) => {
    // State for managing hover tooltips
    const [tooltip, setTooltip] = useState<string | null>(null);

    // Dynamic Style Override for Local Scope
    // If colorTheme is provided, we override --col-primary and --col-primary-rgb LOCALLY for this card.
    // This means all children using text-neon-cyan, border-neon-cyan etc. will adopt the new color automatically.
    const localStyle = colorTheme ? {
        '--col-primary': HOLOGRAM_CONFIG[colorTheme].hex,
        '--col-primary-rgb': HOLOGRAM_CONFIG[colorTheme].rgb,
        backgroundColor: 'var(--bg-surface)'
    } as React.CSSProperties : {
        backgroundColor: 'var(--bg-surface)'
    };

    // Dynamic classes for border/shadow to emphasize the theme if present
    const containerClasses = colorTheme 
        ? "border border-neon-cyan/50 shadow-[0_0_20px_rgba(var(--col-primary-rgb),0.15)] hover:shadow-[0_0_30px_rgba(var(--col-primary-rgb),0.25)]" 
        : "border border-white/10 hover:border-neon-cyan/30 hover:shadow-neon/10";

    return (
        <div className={`relative group ${className} h-full flex flex-col hover:z-50 transition-all duration-200`}>
            {/* Main Container with Local Style Scope */}
            <div 
                className={`backdrop-blur-md p-5 sm:p-6 h-full flex flex-col relative shadow-lg transition-all duration-300 ${containerClasses}`} 
                style={localStyle}
            >
                {/* HUD Corner Brackets - Now reacting to local --col-primary */}
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-neon-cyan/50 rounded-tl-sm group-hover:border-neon-cyan transition-colors z-10 pointer-events-none"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-neon-cyan/50 rounded-tr-sm group-hover:border-neon-cyan transition-colors z-10 pointer-events-none"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-neon-cyan/50 rounded-bl-sm group-hover:border-neon-cyan transition-colors z-10 pointer-events-none"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-neon-cyan/50 rounded-br-sm group-hover:border-neon-cyan transition-colors z-10 pointer-events-none"></div>

                {/* Background Scanline Effect */}
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-neon-cyan/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none overflow-hidden rounded-sm"></div>

                {/* Header - Now Draggable */}
                <div className="flex items-center justify-between mb-6 relative z-10 border-b border-white/10 pb-3 tech-border-bottom cursor-move active:cursor-grabbing select-none">
                    <div className="flex items-center gap-3">
                        <div className="text-neon-cyan drop-shadow-[0_0_5px_rgba(var(--col-primary-rgb),0.8)] transition-colors duration-300">
                            {icon}
                        </div>
                        <h2 className="text-lg font-bold tracking-wider uppercase font-sans pointer-events-none truncate max-w-[150px] sm:max-w-none" style={{ color: 'var(--text-main)' }}>
                            {title}
                        </h2>
                    </div>
                    
                    <div className="flex items-center gap-2" onMouseDown={e => e.stopPropagation()}>
                        {onAnalysisClick && (
                            <div className="relative">
                                <button 
                                    onClick={onAnalysisClick}
                                    onMouseEnter={() => setTooltip('analyze')}
                                    onMouseLeave={() => setTooltip(null)}
                                    className="relative overflow-hidden group/btn flex items-center gap-2 px-3 py-1 rounded-sm border border-neon-cyan/30 hover:border-neon-cyan hover:bg-neon-cyan/10 transition-all outline-none focus:ring-1 focus:ring-neon-cyan/50"
                                    style={{ backgroundColor: 'rgba(var(--bg-base-rgb), 0.5)' }}
                                >
                                    <SparklesIcon className="h-4 w-4 text-neon-cyan group-hover/btn:text-white transition-colors" />
                                    <span className="hidden sm:inline text-xs font-mono text-neon-cyan group-hover/btn:text-[var(--text-main)] tracking-widest uppercase">
                                        Analyze
                                    </span>
                                    {/* Glitch effect bar */}
                                    <div className="absolute bottom-0 left-0 h-[2px] w-0 bg-neon-cyan group-hover/btn:w-full transition-all duration-300"></div>
                                </button>
                                {tooltip === 'analyze' && (
                                    <div className="absolute bottom-full right-0 mb-2 px-3 py-1.5 backdrop-blur border border-neon-cyan text-neon-cyan text-[10px] font-bold font-mono uppercase tracking-widest whitespace-nowrap rounded-sm shadow-[0_0_15px_rgba(var(--col-primary-rgb),0.4)] animate-fadeIn z-50 pointer-events-none" style={{ backgroundColor: 'var(--bg-base)' }}>
                                        <div className="flex items-center gap-2">
                                            <SparklesIcon className="h-3 w-3 animate-pulse" />
                                            <span>Deep Dive Analysis</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {onToggleSize && (
                            <div className="relative">
                                <button 
                                    onClick={onToggleSize}
                                    onMouseEnter={() => setTooltip('resize')}
                                    onMouseLeave={() => setTooltip(null)}
                                    className="p-1.5 rounded-sm hover:text-neon-cyan hover:bg-neon-cyan/10 transition-colors outline-none focus:ring-1 focus:ring-neon-cyan/50"
                                    style={{ color: 'var(--text-sub)' }}
                                >
                                    <ArrowsXYIcon className={`h-4 w-4 transition-transform duration-300 ${isFullWidth ? 'rotate-180' : ''}`} />
                                </button>
                                {tooltip === 'resize' && (
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 border border-neon-cyan/50 text-neon-cyan text-[10px] font-mono uppercase tracking-widest whitespace-nowrap rounded shadow-[0_0_10px_rgba(var(--col-primary-rgb),0.3)] animate-fadeIn z-50 pointer-events-none" style={{ backgroundColor: 'var(--bg-base)' }}>
                                        {isFullWidth ? 'Compress' : 'Expand'}
                                    </div>
                                )}
                            </div>
                        )}

                        {onDelete && (
                             <div className="relative border-l border-slate-700 pl-2 ml-1">
                                <button 
                                    onClick={onDelete}
                                    onMouseEnter={() => setTooltip('delete')}
                                    onMouseLeave={() => setTooltip(null)}
                                    className="p-1.5 rounded-sm hover:text-red-500 hover:bg-red-500/10 transition-colors outline-none focus:ring-1 focus:ring-red-500/50"
                                    style={{ color: 'var(--text-sub)' }}
                                >
                                    <TrashIcon className="h-4 w-4" />
                                </button>
                                {tooltip === 'delete' && (
                                    <div className="absolute bottom-full right-0 mb-2 px-2 py-1 border border-red-500 text-red-500 text-[10px] font-mono uppercase tracking-widest whitespace-nowrap rounded shadow-[0_0_10px_rgba(220,38,38,0.3)] animate-fadeIn z-50 pointer-events-none" style={{ backgroundColor: 'var(--bg-base)' }}>
                                        Destroy Widget
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* Content */}
                <div className="flex-grow relative z-10 font-mono" style={{ color: 'var(--text-sub)' }}>
                    {children}
                </div>
            </div>
        </div>
    );
};

export default Card;
