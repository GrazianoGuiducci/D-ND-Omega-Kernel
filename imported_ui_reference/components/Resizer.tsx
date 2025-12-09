
import React from 'react';

interface ResizerProps {
  onMouseDown: (e: React.MouseEvent) => void;
  isVisible: boolean;
}

export const Resizer: React.FC<ResizerProps> = ({ onMouseDown, isVisible }) => {
  if (!isVisible) return null;

  return (
    <div
      className="group h-full w-4 cursor-col-resize flex-shrink-0 flex flex-col justify-center items-center relative z-20 hover:bg-cyan-900/20 transition-colors duration-200"
      onMouseDown={onMouseDown}
    >
      {/* The visible thin line - now easier to see */}
      <div className="h-full w-[2px] bg-gray-600 group-hover:bg-cyan-400 group-active:bg-cyan-300 transition-colors duration-150 shadow-[0_0_5px_rgba(0,0,0,0.5)] group-hover:shadow-[0_0_10px_rgba(6,182,212,0.8)]"></div>
      
      {/* Handle Grip for better affordance - visible on hover */}
      <div className="absolute top-1/2 -translate-y-1/2 w-6 h-10 rounded-full bg-gray-800 border border-gray-500 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none shadow-lg z-30">
        <div className="w-0.5 h-4 bg-gray-400 mx-0.5"></div>
        <div className="w-0.5 h-4 bg-gray-400 mx-0.5"></div>
      </div>
    </div>
  );
};
