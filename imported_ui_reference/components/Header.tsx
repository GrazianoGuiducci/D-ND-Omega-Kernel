
import React from 'react';
import { DnaIcon } from './icons/DnaIcon';
import { HelpCircleIcon } from './icons/HelpCircleIcon';
import { Tooltip } from './Tooltip';

interface HeaderProps {
  onOpenGuide: () => void;
  areTooltipsEnabled: boolean;
  onToggleTooltips: (enabled: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({ onOpenGuide, areTooltipsEnabled, onToggleTooltips }) => {
  return (
    <header className="bg-gray-800/50 backdrop-blur-sm border-b border-gray-700/50 p-4 shadow-lg sticky top-0 z-10">
      <div className="container mx-auto flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <DnaIcon className="w-8 h-8 text-cyan-400" />
          <div>
              <h1 className="text-xl font-bold text-gray-100 tracking-wider">MMS kernel Tester</h1>
              <p className="text-xs text-gray-400">OMEGA KERNEL v2.0 // MMS Integration</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
            {/* Tooltips Toggle */}
            <div className="flex items-center gap-2 bg-gray-800/50 p-1.5 rounded-lg border border-gray-700/30">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider ml-1">Hints</span>
                <div
                    className={`w-9 h-5 flex items-center rounded-full p-1 cursor-pointer transition-colors duration-300 ${areTooltipsEnabled ? 'bg-cyan-600' : 'bg-gray-600'}`}
                    onClick={() => onToggleTooltips(!areTooltipsEnabled)}
                >
                    <div className={`bg-white w-3 h-3 rounded-full shadow-md transform transition-transform duration-300 ${areTooltipsEnabled ? 'translate-x-4' : 'translate-x-0'}`}></div>
                </div>
            </div>

            {/* Help Icon with System Tooltip */}
            <Tooltip 
                content={{ 
                    function: "User Guide & Documentation.", 
                    mechanism: "Opens the modal overlay with instructions.", 
                    expectation: "Learn how to use the MMS Kernel and Agents.", 
                    example: "Click to read about 'Intent Genesis' or 'Workflow Visualization'." 
                }} 
                position="bottom"
            >
                <button
                    onClick={onOpenGuide}
                    className="text-gray-400 hover:text-cyan-300 transition-colors duration-200 p-1"
                >
                    <HelpCircleIcon className="w-7 h-7" />
                </button>
            </Tooltip>
        </div>
      </div>
    </header>
  );
};
