
import React, { useState, useCallback, useEffect } from 'react';
import { SquaresPlusIcon } from './icons/SquaresPlusIcon';
import { ChartBarIcon } from './icons/ChartBarIcon';
import { CurrencyDollarIcon } from './icons/CurrencyDollarIcon';
import { ScaleIcon } from './icons/ScaleIcon';
import { CpuChipIcon } from './icons/CpuChipIcon';
import { SparklesIcon } from './icons/SparklesIcon';
import { Resizer } from './Resizer';

interface SidebarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    onOpenKernel: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, onOpenKernel }) => {
    const [width, setWidth] = useState(256); // Default w-64
    const [isResizing, setIsResizing] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);

    const navItems = [
        { id: 'dashboard', icon: SquaresPlusIcon, label: 'Overview' },
        { id: 'management', icon: ChartBarIcon, label: 'P&L Control' },
        { id: 'cashflow', icon: CurrencyDollarIcon, label: 'Liquidity' },
        { id: 'rating', icon: ScaleIcon, label: 'Bankability' },
    ];

    // Resize Logic
    const startResizing = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
    }, []);

    useEffect(() => {
        if (!isResizing) return;

        const onMouseMove = (e: MouseEvent) => {
            // Constraints: Min 64px (Icon only), Max 400px
            const newWidth = Math.max(64, Math.min(e.clientX, 400));
            setWidth(newWidth);
            setIsCollapsed(newWidth < 180); // Auto-collapse logic threshold
        };

        const onMouseUp = () => {
            setIsResizing(false);
        };

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        return () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
    }, [isResizing]);

    return (
        <aside
            className="hidden md:flex flex-col border-r shrink-0 h-full relative z-30 group"
            style={{
                width: width,
                transition: isResizing ? 'none' : 'width 0.1s ease-out',
                backgroundColor: 'var(--bg-surface)',
                borderColor: 'var(--col-muted)'
            }}
        >
            {/* Logo Area */}
            <div className={`h-16 flex items-center ${isCollapsed ? 'justify-center' : 'justify-start px-6'} border-b border-white/10 overflow-hidden whitespace-nowrap`}>
                <div className="w-8 h-8 rounded bg-gradient-to-br from-neon-cyan to-blue-600 flex items-center justify-center shadow-lg shadow-neon-cyan/20 shrink-0">
                    <span className="font-bold text-black text-xs font-mono">Ω</span>
                </div>
                {!isCollapsed && (
                    <span className="ml-3 font-bold text-white tracking-widest uppercase text-sm animate-fadeIn">
                        Omega<span className="text-neon-cyan">Sys</span>
                    </span>
                )}
            </div>

            {/* Navigation */}
            <nav className="flex-1 py-6 flex flex-col gap-2 px-2 overflow-hidden">
                {navItems.map((item) => {
                    const isActive = activeTab === item.id;
                    return (
                        <button
                            key={item.id}
                            onClick={() => onTabChange(item.id)}
                            className={`
                                flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200 group relative overflow-hidden shrink-0
                                ${isActive
                                    ? 'bg-neon-cyan/10 text-neon-cyan shadow-[0_0_15px_rgba(0,243,255,0.15)] border border-neon-cyan/20'
                                    : 'text-slate-500 hover:text-white hover:bg-white/5 border border-transparent'}
                                ${isCollapsed ? 'justify-center' : ''}
                            `}
                            title={isCollapsed ? item.label : undefined}
                        >
                            <item.icon className={`w-5 h-5 shrink-0 ${isActive ? 'animate-pulse' : ''}`} />
                            {!isCollapsed && (
                                <span className="font-mono text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                                    {item.label}
                                </span>
                            )}
                            {isActive && <div className="absolute left-0 top-0 bottom-0 w-1 bg-neon-cyan rounded-r"></div>}
                        </button>
                    );
                })}

                <div className="my-4 h-px bg-white/10 mx-2"></div>

                {/* Special Kernel Button */}
                <button
                    onClick={onOpenKernel}
                    className={`flex items-center gap-3 px-3 py-3 rounded-lg text-purple-400 hover:text-purple-300 hover:bg-purple-900/20 border border-transparent hover:border-purple-500/30 transition-all group shrink-0 ${isCollapsed ? 'justify-center' : ''}`}
                    title={isCollapsed ? "Omega Kernel" : undefined}
                >
                    <CpuChipIcon className="w-5 h-5 shrink-0 group-hover:rotate-180 transition-transform duration-700" />
                    {!isCollapsed && (
                        <span className="font-mono text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                            Omega Kernel
                        </span>
                    )}
                </button>
            </nav>

            {/* Bottom Status */}
            <div className={`p-4 border-t border-white/10 overflow-hidden ${isCollapsed ? 'hidden' : 'block'}`}>
                <div className="bg-white/5 rounded p-3 border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <SparklesIcon className="w-3 h-3 text-neon-cyan" />
                        <span className="text-[10px] text-slate-400 font-mono uppercase whitespace-nowrap">System Health</span>
                    </div>
                    <div className="w-full bg-slate-700 h-1 rounded-full overflow-hidden">
                        <div className="bg-green-500 w-full h-full animate-pulse"></div>
                    </div>
                    <div className="flex justify-between mt-1 text-[9px] text-slate-500 font-mono">
                        <span>CPU: 12%</span>
                        <span>MEM: 400MB</span>
                    </div>
                </div>
            </div>

            {/* RESIZER HANDLE */}
            <div className="absolute top-0 right-0 bottom-0 translate-x-1/2 z-50">
                <Resizer onMouseDown={startResizing} isVisible={true} />
            </div>
        </aside>
    );
};

export default Sidebar;
