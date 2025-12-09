
import React, { useState } from 'react';
import { ModuleCard } from './ModuleCard';
import { ZapIcon } from './icons/ZapIcon';
import { Tooltip } from './Tooltip';
import type { PreconfiguredIntent, Risultante, CognitiveModule } from '../types';
import { translations, Language } from '../translations';
import { FileUploader } from './FileUploader';
import { generateCustomProtocol } from '../services/geminiService';
import { DnaIcon } from './icons/DnaIcon';
import { ModuleEditorModal } from './ModuleEditorModal';
import { RefreshCcwIcon } from './icons/RefreshCcwIcon';
import { DEFAULT_COGNITIVE_MODULES } from '../constants';

interface ControlPanelProps {
  intent: string;
  onIntentChange: (value: string) => void;
  onSelectModule: (moduleId: string) => void;
  onClearWorkflow: () => void;
  onActivateEngine: () => void;
  onApplyPreset: (preset: PreconfiguredIntent) => void;
  onAutoSelect: () => void;
  isLoading: boolean;
  isAutoSelecting?: boolean;
  selectedModuleIds: string[];
  language: Language;
  risultante: Risultante | null;
  useContext: boolean;
  onToggleContext: (val: boolean) => void;
  systemProtocols: PreconfiguredIntent[];
  customProtocols: PreconfiguredIntent[];
  onAddCustomProtocol: (p: PreconfiguredIntent) => void;
  onForgeFromResult?: () => void;
  onRegenerateSystemProtocols: () => Promise<void>; // New prop
  
  // Module Management
  availableModules: CognitiveModule[];
  onUpdateModule: (m: CognitiveModule) => void;
  onCreateModule: (m: CognitiveModule) => void;
  onImportModules: (m: CognitiveModule[]) => void;
  onResetModule: (id: string) => void;
  onDeleteModule: (id: string) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  intent,
  onIntentChange,
  onSelectModule,
  onClearWorkflow,
  onActivateEngine,
  onApplyPreset,
  onAutoSelect,
  isLoading,
  isAutoSelecting = false,
  selectedModuleIds,
  language,
  risultante,
  useContext,
  onToggleContext,
  systemProtocols,
  customProtocols,
  onAddCustomProtocol,
  onForgeFromResult,
  onRegenerateSystemProtocols,
  availableModules,
  onUpdateModule,
  onCreateModule,
  onImportModules,
  onResetModule,
  onDeleteModule
}) => {
  const t = translations[language].control;
  const [isForgeOpen, setIsForgeOpen] = useState(false);
  const [forgeLoading, setForgeLoading] = useState(false);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [isRegeneratingProtocols, setIsRegeneratingProtocols] = useState(false);

  const handleForgeSubmit = async (files: File[]) => {
      setForgeLoading(true);
      try {
          // Use current intent as hint, or generic if empty
          const hint = intent.trim() || "Analyze these files and create a protocol.";
          
          const orchestrator = availableModules.find(m => m.id === 'AWO.sys') || DEFAULT_COGNITIVE_MODULES.find(m => m.id === 'AWO.sys');
          const orchestratorDNA = orchestrator?.instruction || "";

          const newProtocol = await generateCustomProtocol(files, hint, orchestratorDNA);
          onAddCustomProtocol(newProtocol);
          setIsForgeOpen(false);
      } catch (e) {
          console.error("Forge failed", e);
          // Could add local error state here if needed
      } finally {
          setForgeLoading(false);
      }
  };

  const handleRegenerateClick = async () => {
      setIsRegeneratingProtocols(true);
      await onRegenerateSystemProtocols();
      setIsRegeneratingProtocols(false);
  };

  return (
    <div className="bg-gray-800/60 rounded-lg p-4 h-full flex flex-col border border-gray-700/50 shadow-2xl overflow-y-auto custom-scrollbar">
      
      <ModuleEditorModal 
        isOpen={isEditorOpen}
        onClose={() => setIsEditorOpen(false)}
        modules={availableModules}
        onUpdate={onUpdateModule}
        onCreate={onCreateModule}
        onImport={onImportModules}
        onReset={onResetModule}
        onDelete={onDeleteModule}
        language={language}
      />

      <h2 className="text-lg font-semibold mb-3 text-cyan-300">{t.step1}</h2>
      
      <Tooltip 
        position="bottom"
        content={{
          function: t.intentTooltipFunc,
          mechanism: t.intentTooltipMech,
          expectation: t.intentTooltipExp,
          example: t.intentTooltipEx
        }}
      >
        <textarea
          value={intent}
          onChange={(e) => onIntentChange(e.target.value)}
          placeholder={t.placeholder}
          className={`w-full h-24 p-2 bg-gray-900/70 border rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none resize-none transition-all text-sm
            ${useContext ? 'border-cyan-500/60 shadow-[0_0_10px_rgba(6,182,212,0.1)]' : 'border-gray-600'}
          `}
          disabled={isLoading || isAutoSelecting}
        />
      </Tooltip>

      {/* Context Toggle for Refinement & Forge Result */}
      {risultante && (
        <div className="mt-2 space-y-2">
            <div className="flex items-center justify-between bg-cyan-900/20 p-2 rounded border border-cyan-900/50">
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${useContext ? 'bg-cyan-400 animate-pulse' : 'bg-gray-600'}`} />
                    <span className="text-xs font-semibold text-cyan-200">{t.refineMode}</span>
                </div>
                <button 
                    onClick={() => onToggleContext(!useContext)}
                    className={`w-10 h-5 rounded-full flex items-center p-1 transition-colors duration-300 ${useContext ? 'bg-cyan-600' : 'bg-gray-700'}`}
                    title={t.refineModeDesc}
                >
                    <div className={`w-3 h-3 bg-white rounded-full shadow-sm transform transition-transform duration-300 ${useContext ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
            </div>
            
            {/* Forge from Result Button */}
            <button 
                onClick={onForgeFromResult}
                className="w-full flex items-center justify-center gap-2 px-3 py-1.5 text-xs bg-gray-800 border border-gray-600 hover:border-cyan-500 hover:text-cyan-300 text-gray-300 rounded transition-all"
                title="Create a protocol from this successful result"
            >
                <DnaIcon className="w-3 h-3" />
                {t.forgeFromResult}
            </button>
        </div>
      )}

      {/* Initialization Protocols & Forge */}
      {!useContext && (
        <div className="mt-4 mb-4 pt-4 border-t border-gray-700/50">
            <div className="flex flex-wrap items-center justify-between mb-3 gap-y-2">
                {/* Header with Tooltip */}
                <div className="flex items-center gap-2">
                    <Tooltip content={t.protocolsHeaderTooltip} position="right">
                        <span className="text-xs font-bold text-cyan-100 uppercase tracking-wider border-b border-dashed border-gray-600 cursor-help pb-0.5">
                            {t.protocols}
                        </span>
                    </Tooltip>

                    {/* Refresh Button with Tooltip */}
                    <Tooltip content={t.refreshProtocolsTooltip} position="right">
                        <button 
                            onClick={handleRegenerateClick}
                            disabled={isRegeneratingProtocols}
                            className="p-1.5 rounded-full bg-gray-900 hover:bg-cyan-900/50 border border-gray-600 hover:border-cyan-400 text-gray-300 hover:text-cyan-300 transition-all ml-1 group shadow-sm"
                        >
                            <RefreshCcwIcon className={`w-3.5 h-3.5 ${isRegeneratingProtocols ? 'animate-spin text-cyan-400' : 'group-hover:rotate-180 transition-transform duration-500'}`} />
                        </button>
                    </Tooltip>
                </div>
                
                {/* New Protocol Button with Tooltip */}
                <Tooltip content={t.protocolForgeTooltip} position="left">
                    <button 
                        onClick={() => setIsForgeOpen(!isForgeOpen)}
                        className={`text-xs px-3 py-1 rounded transition-all flex items-center gap-1 font-semibold flex-shrink-0
                            ${isForgeOpen 
                                ? 'bg-cyan-600 text-white shadow-[0_0_10px_rgba(6,182,212,0.3)]' 
                                : 'bg-gray-900 hover:bg-cyan-900/40 text-cyan-400 hover:text-cyan-300 border border-cyan-900 hover:border-cyan-600'}
                        `}
                    >
                        <span className="text-lg leading-none -mt-0.5">+</span> {t.newProtocol}
                    </button>
                </Tooltip>
            </div>

            {/* Protocol Forge Area */}
            {isForgeOpen && (
                <div className="mb-4 p-3 bg-gray-900/80 border border-dashed border-cyan-700 rounded-lg animate-in slide-in-from-top-2 shadow-inner">
                    <h4 className="text-sm font-bold text-cyan-400 mb-1 flex items-center gap-2">
                        <DnaIcon className="w-4 h-4" />
                        {t.forgeTitle}
                    </h4>
                    <p className="text-xs text-gray-400 mb-3 leading-relaxed">{t.forgeDesc}</p>
                    <FileUploader 
                        onUpload={handleForgeSubmit} 
                        isLoading={forgeLoading} 
                        label={t.uploadLabel} 
                        generatingLabel={t.generating}
                        buttonLabel={t.generateProtocol}
                    />
                </div>
            )}

            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto pr-1 custom-scrollbar content-start">
            {/* Show regenerating skeleton or actual protocols */}
            {isRegeneratingProtocols ? (
                 <div className="w-full flex items-center justify-center py-4 text-xs text-cyan-500 animate-pulse gap-2">
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce delay-75"></div>
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce delay-150"></div>
                    Orchestrator is re-dreaming scenarios...
                 </div>
            ) : (
                [...systemProtocols, ...customProtocols].map((preset) => (
                    <Tooltip key={preset.id} content={preset.text} position="top">
                        <button
                            onClick={() => onApplyPreset(preset)}
                            disabled={isLoading || isAutoSelecting}
                            className="text-xs px-3 py-1.5 bg-gray-800 border border-gray-600 rounded-md text-gray-300 hover:text-white hover:bg-gray-700 hover:border-gray-500 transition-all duration-200 flex items-center gap-2 group disabled:opacity-50 shadow-sm hover:shadow flex-grow-0"
                        >
                            <span className={`w-2 h-2 rounded-full shadow-[0_0_5px_rgba(0,0,0,0.5)] group-hover:shadow-[0_0_8px_currentColor] transition-all
                                ${preset.category === 'Code' ? 'bg-blue-500 text-blue-400' : ''}
                                ${preset.category === 'Strategy' ? 'bg-purple-500 text-purple-400' : ''}
                                ${preset.category === 'Creative' ? 'bg-pink-500 text-pink-400' : ''}
                                ${preset.category === 'Analysis' ? 'bg-yellow-500 text-yellow-400' : ''}
                            `} />
                            {preset.label}
                        </button>
                    </Tooltip>
                ))
            )}
            </div>
        </div>
      )}

      <div className="flex items-center justify-between mb-3 mt-2 border-t border-gray-700/50 pt-4">
        <h2 className="text-lg font-semibold text-cyan-300">{t.step2}</h2>
        <Tooltip
            position="right"
            content={{
                function: t.autoTooltipFunc,
                mechanism: t.autoTooltipMech,
                expectation: t.autoTooltipExp,
                example: t.autoTooltipEx
            }}
        >
            <button
                onClick={onAutoSelect}
                disabled={!intent.trim() || isLoading || isAutoSelecting}
                className="text-xs bg-purple-900/50 border border-purple-700 text-purple-300 px-3 py-1 rounded hover:bg-purple-800 hover:border-purple-500 disabled:opacity-50 transition-all"
            >
                {isAutoSelecting ? (
                     <span className="flex items-center gap-1">
                        <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        {t.autoSelecting}
                     </span>
                ) : (
                    t.autoSelect
                )}
            </button>
        </Tooltip>
      </div>
      <p className="text-xs text-gray-400 mb-3 -mt-2">{t.step2Desc}</p>
      
      {/* Responsive Grid: Auto-fill with min-width ensures 1 column when narrow, 2 when wide */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(145px,1fr))] gap-2 mb-4">
        {availableModules.map((module) => (
          <ModuleCard
            key={module.id}
            module={module}
            onClick={() => onSelectModule(module.id)}
            isSelected={selectedModuleIds.includes(module.id)}
            disabled={isAutoSelecting}
          />
        ))}
        
        {/* Add Agent Button in Grid */}
        <Tooltip 
            content={{
                function: "Agent Architect Interface",
                mechanism: "Opens the Module Registry & Editor.",
                expectation: "Create new agents, import JSON backups, or modify existing custom modules.",
                example: "Build a 'Legal Analyst' or import a team member's 'Python Expert'."
            }}
            position="right"
        >
            <button
                onClick={() => setIsEditorOpen(true)}
                disabled={isAutoSelecting}
                className="w-full p-3 rounded-md border-2 border-dashed border-gray-700 bg-gray-900/30 hover:bg-gray-800 hover:border-cyan-500/50 transition-all text-gray-500 hover:text-cyan-300 flex flex-col items-center justify-center gap-1 min-h-[80px] group"
            >
                <span className="text-2xl font-light opacity-50 group-hover:opacity-100 transition-opacity">+</span>
                <span className="text-xs font-bold uppercase tracking-wider opacity-70 group-hover:opacity-100">{t.addAgent}</span>
            </button>
        </Tooltip>
      </div>

      <div className="mt-auto pt-4 border-t border-gray-700/50 flex flex-col gap-3">
         <Tooltip
            position="top"
            content={{
              function: t.activateTooltipFunc,
              mechanism: t.activateTooltipMech,
              expectation: t.activateTooltipExp,
              example: t.activateTooltipEx
            }}
         >
            <button
              onClick={onActivateEngine}
              disabled={isLoading || isAutoSelecting || !intent.trim() || selectedModuleIds.length === 0}
              className="w-full flex items-center justify-center gap-2 bg-cyan-600 text-white font-bold py-2 px-4 rounded-md hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              {isLoading ? (
                <><svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg> {t.processing}</>
              ) : (
                <><ZapIcon className="w-5 h-5" /> {t.activate}</>
              )}
            </button>
        </Tooltip>

        <Tooltip
          position="top"
          content={{
            function: t.resetTooltipFunc,
            mechanism: t.resetTooltipMech,
            expectation: t.resetTooltipExp,
            example: t.resetTooltipEx
          }}
        >
          <button
            onClick={onClearWorkflow}
            disabled={isLoading || isAutoSelecting}
            className="w-full bg-gray-600 text-gray-200 py-2 px-4 rounded-md hover:bg-gray-500 disabled:opacity-50 transition-colors"
          >
            {t.clear}
          </button>
        </Tooltip>
      </div>
    </div>
  );
};
