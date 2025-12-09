import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface SmartTooltipProps {
    children: React.ReactNode;
    source?: string;
    logic?: string;
    outcome?: string;
    isVisible?: boolean;
    position?: 'top' | 'bottom' | 'left' | 'right';
    alignment?: 'start' | 'center' | 'end';
    className?: string;
}

const SmartTooltip: React.FC<SmartTooltipProps> = ({
    children,
    source = "System",
    logic = "Pass-through",
    outcome = "Action",
    isVisible = true,
    position = 'top',
    alignment = 'center',
    className = "inline-block"
}) => {
    const [isHovered, setIsHovered] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLDivElement>(null);

    const updatePosition = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            const scrollX = window.scrollX;
            const scrollY = window.scrollY;

            let top = 0;
            let left = 0;
            const gap = 10;

            // Base coordinates based on position
            switch (position) {
                case 'top':
                    top = rect.top + scrollY - gap;
                    break;
                case 'bottom':
                    top = rect.bottom + scrollY + gap;
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

            // Horizontal alignment for top/bottom positions
            if (position === 'top' || position === 'bottom') {
                switch (alignment) {
                    case 'start':
                        left = rect.left + scrollX;
                        break;
                    case 'center':
                        left = rect.left + scrollX + rect.width / 2;
                        break;
                    case 'end':
                        left = rect.right + scrollX;
                        break;
                }
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

    if (!isVisible) return <>{children}</>;

    // Calculate transform based on position and alignment
    let transform = '';

    if (position === 'top' || position === 'bottom') {
        const y = position === 'top' ? '-100%' : '0';
        const x = alignment === 'center' ? '-50%' : (alignment === 'end' ? '-100%' : '0');
        transform = `translate(${x}, ${y})`;
    } else if (position === 'left') {
        transform = 'translate(-100%, -50%)';
    } else if (position === 'right') {
        transform = 'translate(0, -50%)';
    }

    // Arrow positioning
    const getArrowClass = () => {
        let base = "absolute w-2 h-2 bg-black/95 border-r border-b border-neon-cyan/30 rotate-45 ";

        if (position === 'top') {
            base += "bottom-[-5px] border-t-0 border-l-0 ";
            if (alignment === 'center') base += "left-1/2 -translate-x-1/2";
            if (alignment === 'start') base += "left-4";
            if (alignment === 'end') base += "right-4";
        }
        if (position === 'bottom') {
            base += "top-[-5px] border-b-0 border-r-0 rotate-[225deg] ";
            if (alignment === 'center') base += "left-1/2 -translate-x-1/2";
            if (alignment === 'start') base += "left-4";
            if (alignment === 'end') base += "right-4";
        }
        if (position === 'left') base += "right-[-5px] top-1/2 -translate-y-1/2 border-l-0 border-t-0 rotate-[-45deg]";
        if (position === 'right') base += "left-[-5px] top-1/2 -translate-y-1/2 border-r-0 border-b-0 rotate-[135deg]";

        return base;
    };

    const tooltipContent = isHovered && (
        <div
            className="fixed z-[9999] pointer-events-none transition-all duration-200 ease-out"
            style={{
                top: coords.top,
                left: coords.left,
                transform
            }}
        >
            <div className="w-64 p-3 rounded-lg border backdrop-blur-md shadow-[0_0_20px_rgba(0,243,255,0.15)] relative" style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-primary)' }}>
                {/* Header / Source */}
                <div className="flex items-center gap-2 mb-2 border-b pb-1" style={{ borderColor: 'var(--col-muted)' }}>
                    <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: 'var(--col-primary)' }}></div>
                    <span className="text-[10px] font-mono uppercase tracking-widest" style={{ color: 'var(--col-primary)' }}>
                        SRC: {source}
                    </span>
                </div>

                {/* Logic Flow */}
                <div className="space-y-2">
                    <div className="flex gap-2 text-[10px] font-mono text-slate-400">
                        <span className="text-slate-600">LOGIC:</span>
                        <span className="text-slate-300 leading-tight">{logic}</span>
                    </div>

                    <div className="flex gap-2 text-[10px] font-mono text-white">
                        <span className="text-slate-600">OUT:</span>
                        <span className="text-green-400 leading-tight">{outcome}</span>
                    </div>
                </div>

                {/* Arrow */}
                <div className={getArrowClass()}></div>
            </div>
        </div>
    );

    return (
        <>
            <div
                ref={triggerRef}
                className={className}
                onMouseEnter={() => { updatePosition(); setIsHovered(true); }}
                onMouseLeave={() => setIsHovered(false)}
            >
                {children}
            </div>
            {createPortal(tooltipContent, document.body)}
        </>
    );
};

export default SmartTooltip;
