import React from 'react';
import type { CognitiveModule } from '../types';
import { ZapIcon } from './icons/ZapIcon';
import { CpuIcon } from './icons/CpuIcon';
import { TerminalIcon } from './icons/TerminalIcon';
import { translations, Language } from '../translations';
import { Tooltip } from './Tooltip';

interface WorkflowCanvasProps {
  modules: CognitiveModule[];
  activeStepIndex: number;
  language: Language;
  onRemoveModule: (moduleId: string) => void;
  onSimulateModule: (module: CognitiveModule) => void; // New prop
}

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({ 
    modules, 
    activeStepIndex, 
    language, 
    onRemoveModule,
    onSimulateModule
}) => {
  const t = translations[language].workflow;
  
  return (
    <div className="bg-gray-800/60 rounded-lg p-4 h-full flex flex-col border border-gray-700/50 shadow-2xl">
      <h2 className="text-lg font-semibold mb-1 text-cyan-300">{t.title}</h2>
      <p className="text-xs text-gray-500 mb-4">{t.removeHint}</p>
      
      <div className="flex-grow rounded-md bg-gray-900/50 p-4 border border-gray-700 overflow-y-auto">
        {modules.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-2">
            <div className="w-12 h-12 rounded-full border border-gray-800 flex items-center justify-center">
                <CpuIcon className="w-6 h-6 opacity-20" />
            </div>
            <p className="text-sm">{t.empty}</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {modules.map((module, index) => {
              const isActive = index === activeStepIndex;
              const isCompleted = index < activeStepIndex;
              
              // Fix for React Error #130: Use fallback icon if module.icon is undefined
              const IconComponent = module.icon || CpuIcon;

              return (
                <React.Fragment key={`${module.id}-${index}`}>
                  {index > 0 && (
                    <div className="flex justify-center">
                      <div className={`h-6 w-0.5 ${isCompleted ? 'bg-cyan-400' : 'bg-gray-600'}`}></div>
                    </div>
                  )}
                  
                    <div
                        className={`flex items-center gap-4 p-3 rounded-md border transition-all duration-300 relative group w-full
                        ${isActive ? 'bg-cyan-900/50 border-cyan-500 scale-105 shadow-cyan-500/20 shadow-lg' : ''}
                        ${isCompleted ? 'bg-green-900/30 border-green-700 opacity-80' : 'bg-gray-800 border-gray-700 hover:border-gray-500'}
                        `}
                    >
                        {/* Icon Circle */}
                        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center relative
                        ${isActive ? 'bg-cyan-500' : ''}
                        ${isCompleted ? 'bg-green-500' : 'bg-gray-700'}
                        `}>
                            <IconComponent className={`w-6 h-6 ${isActive || isCompleted ? 'text-white' : 'text-cyan-300'}`} />
                        </div>
                        
                        {/* Text Content (Clickable for info via Tooltip wrapper if we had one, but main action is select) */}
                        <div className="flex-grow cursor-default">
                            <p className="font-bold text-gray-100">{module.name}</p>
                            <p className="text-sm text-gray-400 line-clamp-1">{module.description}</p>
                        </div>

                        {/* Actions Area */}
                        <div className="flex items-center gap-2 z-10">
                             {/* Simulate Button */}
                             <button
                                onClick={(e) => { e.stopPropagation(); onSimulateModule(module); }}
                                className="p-1.5 rounded-md text-gray-400 hover:text-green-300 hover:bg-green-900/30 border border-transparent hover:border-green-700 transition-all"
                                title="Open Agent Simulator"
                             >
                                <TerminalIcon className="w-4 h-4" />
                             </button>

                             {/* Remove Button */}
                             {!isActive && (
                                <button
                                    onClick={(e) => { e.stopPropagation(); onRemoveModule(module.id); }}
                                    className="p-1.5 rounded-md text-gray-400 hover:text-red-300 hover:bg-red-900/30 border border-transparent hover:border-red-700 transition-all"
                                    title="Remove Agent"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                                </button>
                             )}
                        </div>
                        
                        {isActive && <div className="absolute right-2 top-2"><ZapIcon className="w-4 h-4 text-yellow-300 animate-pulse" /></div>}
                    </div>
                </React.Fragment>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};