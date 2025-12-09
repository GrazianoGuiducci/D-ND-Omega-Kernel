
import React, { useState, useRef, useEffect, useContext, createContext } from 'react';
import { createPortal } from 'react-dom';
import type { TooltipContent } from '../types';

// Create a context to manage global tooltip state
export const TooltipGlobalContext = createContext<boolean>(true);

interface TooltipProps {
  content: TooltipContent | string;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'right' | 'left';
}

export const Tooltip: React.FC<TooltipProps> = ({ content, children, position = 'right' }) => {
  const isEnabled = useContext(TooltipGlobalContext); // Listen to global state
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });
  const [finalPosition, setFinalPosition] = useState(position);
  const [arrowOffset, setArrowOffset] = useState(0);
  const triggerRef = useRef<HTMLDivElement>(null);

  // Approximate dimensions for calculation
  const TOOLTIP_ESTIMATED_WIDTH = 288; // w-72 is 18rem = 288px
  const GAP = 12; // Padding from screen edge

  const updatePosition = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      let activePos = position;

      // 1. Smart Flipping Logic
      // Horizontal Flip
      if (position === 'left' && rect.left - TOOLTIP_ESTIMATED_WIDTH - GAP < 0) {
         activePos = 'right';
      } else if (position === 'right' && rect.right + TOOLTIP_ESTIMATED_WIDTH + GAP > viewportWidth) {
         activePos = 'left';
      }

      // Vertical Flip (Simple check)
      if (position === 'top' && rect.top < 200) { // arbitrarily check if near top
         activePos = 'bottom';
      } else if (position === 'bottom' && rect.bottom > viewportHeight - 200) {
         activePos = 'top';
      }

      setFinalPosition(activePos);

      // 2. Coordinate Calculation & Clamping
      let top = 0;
      let left = 0;
      let xCorrection = 0; // To shift tooltip if it hits side edges (for top/bottom)

      switch (activePos) {
        case 'right':
          top = rect.top + rect.height / 2;
          left = rect.right + GAP; 
          break;
        case 'left':
          top = rect.top + rect.height / 2;
          left = rect.left - GAP; 
          break;
        case 'bottom':
          top = rect.bottom + GAP;
          left = rect.left + rect.width / 2;
          break;
        case 'top':
        default:
          top = rect.top - GAP;
          left = rect.left + rect.width / 2;
          break;
      }

      // 3. Boundary Clamping for Top/Bottom positioning
      // If positioning is Top/Bottom, the tooltip is horizontally centered.
      // We need to ensure the *edges* of the tooltip don't go off screen.
      if (activePos === 'top' || activePos === 'bottom') {
          const halfWidth = TOOLTIP_ESTIMATED_WIDTH / 2;
          const predictedLeft = left - halfWidth;
          const predictedRight = left + halfWidth;

          if (predictedLeft < GAP) {
              xCorrection = GAP - predictedLeft;
          } else if (predictedRight > viewportWidth - GAP) {
              xCorrection = (viewportWidth - GAP) - predictedRight;
          }
          
          // Apply correction to the container position
          left += xCorrection;
      }

      setCoords({ top, left });
      setArrowOffset(-xCorrection); // Move arrow opposite way to stay on target
    }
  };

  const handleMouseEnter = () => {
    if (!isEnabled) return; // Do nothing if globally disabled
    updatePosition();
    setIsVisible(true);
  };

  useEffect(() => {
    if (isVisible) {
      const handleScroll = () => setIsVisible(false);
      window.addEventListener('scroll', handleScroll, true);
      window.addEventListener('resize', handleScroll);
      return () => {
        window.removeEventListener('scroll', handleScroll, true);
        window.removeEventListener('resize', handleScroll);
      };
    }
  }, [isVisible]);

  // Determine transform based on FINAL position
  let transformStyle = '';
  switch (finalPosition) {
      case 'right':
          transformStyle = 'translateY(-50%)';
          break;
      case 'left':
          transformStyle = 'translateX(-100%) translateY(-50%)';
          break;
      case 'bottom':
          transformStyle = 'translateX(-50%)';
          break;
      case 'top':
      default:
          transformStyle = 'translateX(-50%) translateY(-100%)';
          break;
  }

  const isComplex = typeof content !== 'string';

  const tooltipContent = (
    <div 
        className="fixed z-[9999] w-72 pointer-events-none"
        style={{ 
            top: `${coords.top}px`, 
            left: `${coords.left}px`,
            transform: transformStyle
        }}
    >
      <div className="bg-gray-900/95 backdrop-blur-xl border border-cyan-500/40 rounded-lg shadow-[0_0_25px_rgba(0,0,0,0.7)] p-3 text-left text-sm animate-in fade-in zoom-in-95 duration-150 ring-1 ring-cyan-500/20 relative">
        
        {isComplex ? (
            <>
                <div className="mb-2 pb-2 border-b border-gray-700/50">
                <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest block mb-0.5">Function</span>
                <p className="text-gray-100 font-medium leading-tight">{(content as TooltipContent).function}</p>
                </div>
                
                <div className="mb-2">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Mechanism</span>
                <p className="text-gray-400 text-xs leading-snug mt-0.5">{(content as TooltipContent).mechanism}</p>
                </div>

                <div className="mb-3">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Expectation</span>
                <p className="text-gray-400 text-xs leading-snug mt-0.5">{(content as TooltipContent).expectation}</p>
                </div>

                <div className="bg-gray-800/80 p-2 rounded border-l-2 border-cyan-500/80">
                <span className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest block">Use Case</span>
                <p className="text-gray-300 text-xs italic mt-0.5">"{(content as TooltipContent).example}"</p>
                </div>
            </>
        ) : (
            <p className="text-gray-200 text-xs leading-relaxed font-medium">
                {content as string}
            </p>
        )}
      </div>
      
      {/* Arrow Indicator */}
      {/* We translate the arrow by arrowOffset to keep it pointing at the trigger even if the box shifted */}
       <div 
        className={`absolute w-3 h-3 bg-gray-900 border-cyan-500/40 transform rotate-45
        ${finalPosition === 'top' ? '-bottom-1.5 left-1/2 border-r border-b' : ''}
        ${finalPosition === 'bottom' ? '-top-1.5 left-1/2 border-t border-l' : ''}
        ${finalPosition === 'right' ? '-left-1.5 top-1/2 -translate-y-1/2 border-b border-l' : ''}
        ${finalPosition === 'left' ? '-right-1.5 top-1/2 -translate-y-1/2 border-t border-r' : ''}
       `}
       style={{
           marginLeft: (finalPosition === 'top' || finalPosition === 'bottom') ? `${arrowOffset - 6}px` : undefined
       }}
      />
    </div>
  );

  return (
    <div 
      className="relative" 
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setIsVisible(false)}
      ref={triggerRef}
    >
      {children}
      {isVisible && isEnabled && createPortal(tooltipContent, document.body)}
    </div>
  );
};
