
import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface QuantumTooltipProps {
    children: React.ReactNode;
    title: string;
    desc?: string;
    mechanism?: string;
    expectation?: string;
    useCase?: string;
    type?: 'info' | 'warning' | 'energy' | 'physics';
    position?: 'top' | 'bottom' | 'left' | 'right';
}

const QuantumTooltip: React.FC<QuantumTooltipProps> = ({
    children,
    title,
    desc,
    mechanism,
    expectation,
    useCase,
    type = 'info',
    position = 'top'
}) => {
    const [isHovered, setIsHovered] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLDivElement>(null);

    const colors = {
        info: 'border-cyan-500 shadow-[0_0_20px_rgba(6,182,212,0.3)]',
        warning: 'border-amber-500 shadow-[0_0_20px_rgba(245,158,11,0.3)]',
        energy: 'border-purple-500 shadow-[0_0_20px_rgba(168,85,247,0.3)]',
        physics: 'border-green-500 shadow-[0_0_20px_rgba(34,197,94,0.3)]'
    };

    const typeColor = type === 'info' ? 'text-cyan-400' : type === 'warning' ? 'text-amber-400' : type === 'physics' ? 'text-green-400' : 'text-purple-400';

    const updatePosition = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            const scrollX = window.scrollX;
            const scrollY = window.scrollY;
            const gap = 10;

            let top = 0;
            let left = 0;

            switch (position) {
                case 'top':
                    top = rect.top + scrollY - gap;
                    left = rect.left + scrollX + rect.width / 2;
                    break;
                case 'bottom':
                    top = rect.bottom + scrollY + gap;
                    left = rect.left + scrollX + rect.width / 2;
                    break;
                case 'left':
                    top = rect.top + scrollY + rect.height / 2;
                    left = rect.left + scrollX - gap;
                    break;
                case 'right':
                    top = rect.top + scrollY + rect.height / 2;
                    left = rect.right + scrollX + gap;
                    break;
            }
            setCoords({ top, left });
        }
    };

    useEffect(() => {
        if (isHovered) {
            updatePosition();
            window.addEventListener('scroll', updatePosition);
            window.addEventListener('resize', updatePosition);
        }
        return () => {
            window.removeEventListener('scroll', updatePosition);
            window.removeEventListener('resize', updatePosition);
        };
    }, [isHovered]);

    // Transform logic for Portal
    let transform = 'translate(-50%, -100%)';
    if (position === 'bottom') transform = 'translate(-50%, 0)';
    if (position === 'left') transform = 'translate(-100%, -50%)';
    if (position === 'right') transform = 'translate(0, -50%)';

    const tooltipContent = isHovered && (
        <div
            className="fixed z-[9999] pointer-events-none transition-all duration-200 ease-out"
            style={{ top: coords.top, left: coords.left, transform }}
        >
            <div className={`
                w-72 border-l-2 p-0
                backdrop-blur-xl shadow-2xl
                ${colors[type]}
            `} style={{ backgroundColor: 'var(--bg-surface)' }}>
                {/* Header */}
                <div className="p-2 border-b flex justify-between items-center" style={{ backgroundColor: 'var(--bg-base)', borderColor: 'var(--col-muted)' }}>
                    <span className={`text-[10px] font-mono uppercase tracking-widest font-bold ${typeColor}`}>
                        {title}
                    </span>
                    <span className="text-[8px] text-slate-500 px-1 rounded border border-white/5" style={{ backgroundColor: 'var(--bg-surface)' }}>SYS_INFO</span>
                </div>

                <div className="p-3 space-y-3">
                    {/* Main Description */}
                    {desc && (
                        <p className="text-xs font-sans text-slate-200 leading-snug">
                            {desc}
                        </p>
                    )}

                    {/* Technical Sections */}
                    {(mechanism || expectation || useCase) && (
                        <div className="space-y-2 mt-2 pt-2 border-t border-white/10">
                            {mechanism && (
                                <div>
                                    <span className="text-[8px] uppercase tracking-wider text-slate-500 font-mono block mb-0.5">Backend Mechanism</span>
                                    <p className="text-[10px] text-slate-400 font-mono leading-tight pl-2 border-l border-slate-700">
                                        {mechanism}
                                    </p>
                                </div>
                            )}
                            {expectation && (
                                <div>
                                    <span className="text-[8px] uppercase tracking-wider text-slate-500 font-mono block mb-0.5">Visual Expectation</span>
                                    <p className="text-[10px] text-slate-400 font-mono leading-tight pl-2 border-l border-slate-700">
                                        {expectation}
                                    </p>
                                </div>
                            )}
                            {useCase && (
                                <div>
                                    <span className="text-[8px] uppercase tracking-wider text-slate-500 font-mono block mb-0.5">Strategic Use Case</span>
                                    <p className="text-[10px] text-slate-300 font-mono leading-tight pl-2 border-l border-slate-700 text-purple-200">
                                        {useCase}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    return (
        <>
            <div
                ref={triggerRef}
                className="group relative inline-block w-full"
                onMouseEnter={() => { updatePosition(); setIsHovered(true); }}
                onMouseLeave={() => setIsHovered(false)}
            >
                {children}
            </div>
            {createPortal(tooltipContent, document.body)}
        </>
    );
};

export default QuantumTooltip;
