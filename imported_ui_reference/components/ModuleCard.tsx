
import React from 'react';
import type { CognitiveModule } from '../types';
import { Tooltip } from './Tooltip';

interface ModuleCardProps {
  module: CognitiveModule;
  onClick: () => void;
  isSelected: boolean;
  disabled?: boolean;
}

export const ModuleCard: React.FC<ModuleCardProps> = ({ module, onClick, isSelected, disabled }) => {
  return (
    <Tooltip content={module.tooltip} position="right">
      <button
        onClick={onClick}
        disabled={disabled}
        className={`w-full p-3 rounded-md text-left transition-all duration-200 border relative overflow-hidden group flex flex-col
          ${isSelected 
            ? 'bg-cyan-900/30 border-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.15)]' 
            : 'bg-gray-800/50 border-gray-700 hover:bg-gray-700/80 hover:border-gray-500'
          }
          disabled:opacity-50 disabled:cursor-not-allowed
        `}
      >
        {/* Active Indicator Stripe */}
        {isSelected && (
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
        )}

        {/* Header: Tags only (ID removed) */}
        <div className="flex items-center justify-end w-full mb-1 pl-2 gap-1">
            {module.isCustom && (
                <span className="text-[9px] bg-purple-900/80 text-purple-300 px-1.5 py-0.5 rounded uppercase border border-purple-800 font-bold tracking-wider">
                    User
                </span>
            )}
            {module.isSystemModified && (
                <span className="text-[9px] bg-yellow-900/80 text-yellow-300 px-1.5 py-0.5 rounded uppercase border border-yellow-800 font-bold tracking-wider">
                    Mod
                </span>
            )}
        </div>

        {/* Main Name */}
        <div className="pl-2">
          <p className={`font-bold text-base truncate transition-colors ${isSelected ? 'text-cyan-100' : 'text-gray-200 group-hover:text-white'}`}>
            {module.name}
          </p>
        </div>
        
        {/* Description truncated */}
        <div className="pl-2 mt-1">
             <p className="text-xs text-gray-400 line-clamp-2 leading-snug">{module.description}</p>
        </div>
      </button>
    </Tooltip>
  );
};
