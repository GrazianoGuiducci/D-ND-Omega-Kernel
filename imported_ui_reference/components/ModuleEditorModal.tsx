import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { CognitiveModule } from '../types';
import { translations, Language } from '../translations';
import { SettingsIcon } from './icons/SettingsIcon';
import { analyzeAgentPrompt, fileToText } from '../services/geminiService';
import { DownloadIcon } from './icons/DownloadIcon';
import { ZapIcon } from './icons/ZapIcon';
import { Resizer } from './Resizer';
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import { Tooltip } from './Tooltip';

interface ModuleEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  modules: CognitiveModule[];
  onUpdate: (m: CognitiveModule) => void;
  onCreate: (m: CognitiveModule) => void;
  onImport: (m: CognitiveModule[]) => void;
  onReset: (id: string) => void;
  onDelete: (id: string) => void;
  language: Language;
}

const SNAP_THRESHOLD = 250;

export const ModuleEditorModal: React.FC<ModuleEditorModalProps> = ({
  isOpen,
  onClose,
  modules,
  onUpdate,
  onCreate,
  onImport,
  onReset,
  onDelete,
  language
}) => {
  const t = translations[language].editor;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editMode, setEditMode] = useState<'edit' | 'create'>('edit');
  
  // Form State
  const [formData, setFormData] = useState<CognitiveModule | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const importInputRef = useRef<HTMLInputElement>(null);

  // Layout State
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [modalSize, setModalSize] = useState({ width: 1100, height: 750 });
  
  // Refs for high-performance resizing
  const isResizingSidebarRef = useRef(false);
  const isResizingModalRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
  const rafRef = useRef<number | null>(null);
  
  // State for UI styling (disabling transitions during drag)
  const [isResizing, setIsResizing] = useState(false);
  
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const modalRef = useRef<HTMLDivElement>(null);

  // Monitor Window Resize for Responsive Switching
  useEffect(() => {
      const handleWindowResize = () => {
          setIsMobile(window.innerWidth < 768);
      };
      window.addEventListener('resize', handleWindowResize);
      return () => window.removeEventListener('resize', handleWindowResize);
  }, []);

  useEffect(() => {
    if (isOpen && modules.length > 0 && !selectedId) {
        setSelectedId(modules[0].id);
    }
  }, [isOpen, modules, selectedId]);

  useEffect(() => {
      if (editMode === 'edit' && selectedId) {
          const mod = modules.find(m => m.id === selectedId);
          if (mod) setFormData({ ...mod });
      } else if (editMode === 'create') {
          setFormData({
              id: `agent.${Date.now().toString().slice(-6)}`,
              name: 'New Agent',
              description: '',
              instruction: '', 
              icon: SettingsIcon, 
              tooltip: {
                  function: '',
                  mechanism: '',
                  expectation: '',
                  example: ''
              },
              isCustom: true
          });
      }
  }, [selectedId, editMode, modules]);

  // --- Resizing Logic (Optimized with Snap) ---

  const startResizingSidebar = useCallback((e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      isResizingSidebarRef.current = true;
      dragStartRef.current = { x: e.clientX, y: e.clientY, w: sidebarWidth || 280, h: 0 };
      setIsResizing(true);
  }, [sidebarWidth]);

  const startResizingModal = useCallback((e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      isResizingModalRef.current = true;
      dragStartRef.current = { x: e.clientX, y: e.clientY, w: modalSize.width, h: modalSize.height };
      setIsResizing(true);
  }, [modalSize]);

  useEffect(() => {
      const handleMouseMove = (e: MouseEvent) => {
          if (!isResizingSidebarRef.current && !isResizingModalRef.current) return;

          // Use requestAnimationFrame for fluid 60fps updates
          if (rafRef.current) cancelAnimationFrame(rafRef.current);

          rafRef.current = requestAnimationFrame(() => {
              const deltaX = e.clientX - dragStartRef.current.x;
              const deltaY = e.clientY - dragStartRef.current.y;

              if (isResizingSidebarRef.current) {
                   const rawWidth = dragStartRef.current.w + deltaX;
                   
                   // Snap Logic
                   if (rawWidth < SNAP_THRESHOLD) {
                       setSidebarWidth(0);
                   } else {
                       const newWidth = Math.max(SNAP_THRESHOLD, Math.min(500, rawWidth));
                       setSidebarWidth(newWidth);
                   }
              }
              
              if (isResizingModalRef.current) {
                  // Apply constraints
                  const newWidth = Math.max(700, Math.min(window.innerWidth - 20, dragStartRef.current.w + deltaX));
                  const newHeight = Math.max(500, Math.min(window.innerHeight - 20, dragStartRef.current.h + deltaY));
                  setModalSize({ width: newWidth, height: newHeight });
              }
          });
      };

      const handleMouseUp = () => {
          if (isResizingSidebarRef.current || isResizingModalRef.current) {
              isResizingSidebarRef.current = false;
              isResizingModalRef.current = false;
              setIsResizing(false);
              if (rafRef.current) cancelAnimationFrame(rafRef.current);
          }
      };

      // Attach globally
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
          if (rafRef.current) cancelAnimationFrame(rafRef.current);
      };
  }, []); 

  if (!isOpen) return null;

  // --- Actions ---

  const handleSave = () => {
      if (!formData) return;
      if (editMode === 'create') {
          onCreate(formData);
          setSelectedId(formData.id);
          setEditMode('edit');
      } else {
          onUpdate(formData);
      }
  };

  const handleDownloadMd = () => {
      if (!formData) return;
      const content = formData.instruction;
      const blob = new Blob([content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${formData.id}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
  };

  const handleExportJson = () => {
      if (!formData) return;
      const content = JSON.stringify(formData, null, 2);
      const blob = new Blob([content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${formData.id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0] && formData) {
          try {
              const text = await fileToText(e.target.files[0]);
              setFormData({ ...formData, instruction: text });
          } catch (err) {
              console.error("Failed to read file", err);
          }
      }
  };

  const handleBatchImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
          const files = Array.from(e.target.files) as File[];
          const importedModules: CognitiveModule[] = [];

          for (const file of files) {
              try {
                  const text = await fileToText(file);
                  if (file.name.endsWith('.json')) {
                      const parsed = JSON.parse(text);
                      if (Array.isArray(parsed)) {
                          importedModules.push(...parsed);
                      } else if (parsed.id && parsed.name) {
                          importedModules.push(parsed);
                      }
                  } else {
                      const nameMatch = text.match(/# IDENTITY:\s*(.*)/i);
                      const name = nameMatch ? nameMatch[1].trim() : file.name.replace(/\.[^/.]+$/, "");
                      
                      importedModules.push({
                          id: `imported.${Date.now()}.${Math.floor(Math.random() * 1000)}`,
                          name: name,
                          description: "Imported from file.",
                          instruction: text,
                          isCustom: true,
                          tooltip: {
                              function: "Unknown",
                              mechanism: "Imported",
                              expectation: "Unknown",
                              example: "Unknown"
                          }
                      });
                  }
              } catch (err) {
                  console.error(`Failed to import file ${file.name}`, err);
              }
          }
          
          if (importedModules.length > 0) {
              onImport(importedModules);
          }
      }
  };

  const handleAutoExtract = async () => {
      if (!formData || !formData.instruction.trim()) return;
      setIsAnalyzing(true);
      try {
          const extracted = await analyzeAgentPrompt(formData.instruction);
          setFormData({
              ...formData,
              name: extracted.name || formData.name,
              description: extracted.description || formData.description,
              tooltip: {
                  ...formData.tooltip,
                  ...extracted.tooltip
              }
          });
      } catch (e) {
          console.error(e);
      } finally {
          setIsAnalyzing(false);
      }
  };

  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-50 flex items-center justify-center p-0 md:p-6 animate-in fade-in duration-200">
      
      <div 
        ref={modalRef}
        className={`bg-gray-800 shadow-2xl border-gray-700 flex flex-col md:flex-row overflow-hidden relative ${isResizing ? 'transition-none' : 'transition-all'} duration-200 ease-out 
        ${isMobile ? 'w-full h-full rounded-none' : 'md:rounded-2xl'}`}
        style={{ 
            width: isMobile ? '100%' : `${modalSize.width}px`,
            height: isMobile ? '100%' : `${modalSize.height}px`,
        }}
      >
        <div className="contents">
        
        {/* --- Sidebar List --- */}
        <div 
            className="flex-shrink-0 h-full transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] relative z-10"
            style={{ width: isMobile ? '100%' : (sidebarWidth === 0 ? '48px' : `${sidebarWidth}px`) }}
        >
        {sidebarWidth === 0 && !isMobile ? (
             // Collapsed Gutter (Docked Tab style - Cyan Theme)
             <div className="w-full h-full">
                <button
                    onClick={() => setSidebarWidth(280)}
                    className="
                        h-full w-full
                        bg-gray-900/95 backdrop-blur-sm
                        border-r border-gray-800
                        rounded-r-2xl
                        flex flex-col items-center justify-between py-8
                        hover:bg-gray-800 hover:border-cyan-500/60
                        transition-all duration-300
                        group
                        shadow-[5px_0_15px_rgba(0,0,0,0.3)]
                        cursor-pointer
                        relative z-10
                    "
                >
                    {/* Icon */}
                    <div className="p-2 rounded-lg bg-gray-800/50 text-cyan-500 group-hover:bg-cyan-500 group-hover:text-white transition-all mb-4 shadow-[0_0_10px_rgba(6,182,212,0.2)]">
                        <ChevronDownIcon className="w-5 h-5 -rotate-90" />
                    </div>

                    {/* Vertical Text */}
                    <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                        <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-cyan-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                            <span>Registry</span>
                            <span className="w-px h-8 bg-gray-700 group-hover:bg-cyan-500/50 transition-colors"></span>
                        </div>
                    </div>

                    {/* Dot */}
                    <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-cyan-400 shadow-[0_0_5px_rgba(6,182,212,0.8)] transition-colors mt-4"></div>
                </button>
             </div>
        ) : (
            <div 
                className="bg-gray-900/60 h-full border-b md:border-b-0 md:border-r border-gray-700 flex flex-col flex-shrink-0 w-full overflow-hidden"
            >
                {/* Sidebar Header */}
                <div className="p-3 md:p-4 border-b border-gray-700/50 bg-gray-900/80 flex flex-col gap-2 md:gap-3 flex-shrink-0">
                    <div className="flex items-center justify-between">
                        <h3 className="text-base font-bold text-cyan-300 truncate mr-2 flex items-center gap-2">
                            <SettingsIcon className="w-4 h-4 text-cyan-500" />
                            {t.moduleRegistry}
                        </h3>
                        <button onClick={onClose} className="md:hidden text-gray-400 hover:text-white bg-gray-700/50 p-1.5 rounded-md">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                        </button>
                    </div>
                    <div className="flex gap-2 w-full">
                        <button 
                            onClick={() => setEditMode('create')}
                            className="flex-1 py-1.5 text-xs bg-cyan-700 hover:bg-cyan-600 text-white rounded transition-colors font-semibold shadow-lg shadow-cyan-900/20"
                        >
                            + {t.create}
                        </button>
                        <input 
                            type="file" 
                            ref={importInputRef}
                            className="hidden"
                            multiple
                            accept=".json,.md,.txt"
                            onChange={handleBatchImport}
                        />
                        <Tooltip 
                            content={{
                                function: "Import Agent DNA",
                                mechanism: "Parses external JSON/Text files to reconstruct cognitive modules.",
                                expectation: "Adds a new entity to your registry from a backup file.",
                                example: "Uploading 'coder_v2.json' to restore an agent."
                            }}
                            position="bottom"
                        >
                            <button 
                                onClick={() => importInputRef.current?.click()}
                                className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded border border-gray-600 transition-colors"
                            >
                            <DownloadIcon className="w-3 h-3 rotate-180" />
                            </button>
                        </Tooltip>
                    </div>
                </div>

                {/* Sidebar List Content */}
                <div className="flex-1 overflow-y-auto min-h-0 p-2 space-y-1 custom-scrollbar">
                    {modules.map(m => (
                        <button
                            key={m.id}
                            onClick={() => { setSelectedId(m.id); setEditMode('edit'); }}
                            className={`w-full text-left p-2 md:p-3 rounded-lg border transition-all duration-200 group relative
                                ${selectedId === m.id && editMode === 'edit' 
                                    ? 'bg-cyan-900/30 border-cyan-500/50 shadow-[inset_0_0_10px_rgba(6,182,212,0.1)]' 
                                    : 'bg-transparent border-transparent hover:bg-gray-800 hover:border-gray-700'}
                            `}
                        >
                            <div className="flex flex-col w-full min-w-0">
                                <div className="flex items-center w-full mb-0.5 md:mb-1">
                                    <span className={`font-bold text-xs truncate flex-1 mr-2 ${selectedId === m.id ? 'text-cyan-200' : 'text-gray-300 group-hover:text-white'}`} title={m.name}>
                                        {m.name}
                                    </span>
                                    
                                    <div className="flex gap-1 flex-shrink-0">
                                        {m.isCustom && <span className="text-[8px] bg-purple-900/80 text-purple-300 px-1 rounded border border-purple-800 font-mono">USR</span>}
                                        {m.isSystemModified && <span className="text-[8px] bg-yellow-900/80 text-yellow-300 px-1 rounded border border-yellow-800 font-mono">MOD</span>}
                                    </div>
                                </div>
                                
                                <p className="text-[10px] text-gray-500 truncate w-full block leading-tight opacity-80 group-hover:opacity-100 transition-opacity">
                                    {m.description || "No description."}
                                </p>
                            </div>
                            
                            {selectedId === m.id && editMode === 'edit' && (
                                <div className="absolute left-0 top-2 bottom-2 w-0.5 bg-cyan-400 rounded-r shadow-[0_0_5px_rgba(6,182,212,0.8)]"></div>
                            )}
                        </button>
                    ))}
                </div>
            </div>
        )}
        </div>

        {/* Sidebar Resizer (Desktop Only) */}
        {!isMobile && sidebarWidth > 0 && (
            <div className="hidden md:block h-full flex-shrink-0 relative z-10 -ml-1">
                <Resizer onMouseDown={startResizingSidebar} isVisible={true} />
            </div>
        )}

        {/* --- Main Edit Area --- */}
        <div className="flex-1 flex flex-col bg-gray-800 h-full min-w-0">
            {/* Header */}
            <div className="flex-shrink-0 h-14 md:h-16 px-4 md:px-6 border-b border-gray-700 flex justify-between items-center bg-gray-800 z-10">
                <div className="min-w-0 mr-4 flex-1">
                    <div className="flex items-center gap-2">
                         <h2 className="text-base md:text-lg font-bold text-gray-100 truncate">
                            {editMode === 'create' ? t.createNew : formData?.name}
                        </h2>
                        {editMode === 'edit' && <span className="text-[10px] text-gray-500 font-mono bg-gray-900 px-1.5 py-0.5 rounded hidden sm:inline-block">{formData?.id}</span>}
                    </div>
                    <p className="text-[10px] md:text-xs text-gray-400 mt-0.5 truncate">Configure Identity, Axioms & Routing Logic.</p>
                </div>
                {/* Desktop Close Button */}
                <button onClick={onClose} className="hidden md:block text-gray-400 hover:text-white bg-gray-700/50 hover:bg-red-900/50 hover:text-red-200 p-1.5 rounded-md transition-colors">
                    <span className="sr-only">Close</span>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
            </div>

            {/* Content Container - Flex column for independent scrolling */}
            {formData ? (
                <div className="flex-1 flex flex-col min-h-0 overflow-y-auto custom-scrollbar">
                    
                    {/* 1. AGENT CONFIGURATION (Fixed/Scrollable at Top) */}
                    <div className="flex-shrink-0 p-4 md:p-6 border-b border-gray-700/50 bg-gray-800 z-10">
                        <div className="flex justify-between items-center mb-4">
                             <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                                Agent Configuration <span className="text-gray-600 ml-2">v2.0 Spec</span>
                             </h4>
                             
                             {/* Auto-Scan Button */}
                             <Tooltip 
                                content={{
                                    function: "Auto-Scan DNA",
                                    mechanism: "Analyzes the Instruction text using Gemini.",
                                    expectation: "Auto-fills Name, Description, and Routing Logic based on the prompt.",
                                    example: "Paste a prompt below, then click this."
                                }}
                                position="left"
                             >
                                 <button
                                    onClick={handleAutoExtract}
                                    disabled={!formData.instruction || isAnalyzing}
                                    className="flex items-center gap-2 px-3 py-1.5 bg-cyan-900/30 border border-cyan-500/30 text-cyan-400 rounded hover:bg-cyan-900/50 disabled:opacity-50 transition-all text-[10px] font-bold uppercase tracking-wider"
                                >
                                    {isAnalyzing ? (
                                        <span className="animate-pulse">Scanning...</span>
                                    ) : (
                                        <>
                                            <ZapIcon className="w-3 h-3" />
                                            Auto-Scan
                                        </>
                                    )}
                                </button>
                             </Tooltip>
                        </div>
                        
                        <div className="grid grid-cols-12 gap-3 md:gap-4">
                            <div className="col-span-12 md:col-span-8">
                                <label className="block text-[10px] text-gray-400 uppercase font-bold mb-1">{t.name}</label>
                                <input 
                                    type="text" 
                                    value={formData.name} 
                                    onChange={e => setFormData({...formData, name: e.target.value})}
                                    className="w-full bg-gray-950 border border-gray-700 p-2 rounded text-gray-200 text-xs focus:border-cyan-500 focus:outline-none font-bold placeholder-gray-700"
                                />
                            </div>
                            <div className="col-span-12 md:col-span-4">
                                <label className="block text-[10px] text-gray-400 uppercase font-bold mb-1">ID (Unique)</label>
                                <input 
                                    type="text" 
                                    value={formData.id} 
                                    disabled={editMode === 'edit'} 
                                    onChange={e => setFormData({...formData, id: e.target.value})}
                                    className="w-full bg-gray-950 border border-gray-700 p-2 rounded text-gray-400 text-xs focus:border-cyan-500 focus:outline-none disabled:opacity-60 font-mono"
                                />
                            </div>
                            <div className="col-span-12">
                                <label className="block text-[10px] text-gray-400 uppercase font-bold mb-1">{t.description}</label>
                                <input 
                                    type="text"
                                    value={formData.description} 
                                    onChange={e => setFormData({...formData, description: e.target.value})}
                                    className="w-full bg-gray-950 border border-gray-700 p-2 rounded text-gray-300 text-xs focus:border-cyan-500 focus:outline-none"
                                />
                            </div>
                        </div>

                        <div className="space-y-3 pt-4">
                             <div className="flex items-center justify-between">
                                <p className="text-[10px] text-gray-500 font-mono">{t.routingDesc}</p>
                             </div>
                             <div>
                                <label className="block text-[10px] text-cyan-700 uppercase font-bold mb-1">{t.mechanism} (Logic Physics)</label>
                                <input 
                                    value={formData.tooltip.mechanism} 
                                    onChange={e => setFormData({...formData, tooltip: {...formData.tooltip, mechanism: e.target.value}})}
                                    className="w-full bg-gray-950 border border-gray-800 p-2 rounded text-gray-400 text-xs focus:border-cyan-800 focus:outline-none font-mono"
                                    placeholder="e.g. Delta-Link Scan, Recursive Logic..."
                                />
                            </div>
                             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                 <div>
                                    <label className="block text-[10px] text-gray-500 uppercase font-bold mb-1">{t.function}</label>
                                    <input 
                                        type="text" 
                                        value={formData.tooltip.function} 
                                        onChange={e => setFormData({...formData, tooltip: {...formData.tooltip, function: e.target.value}})}
                                        className="w-full bg-gray-950 border border-gray-800 p-2 rounded text-gray-400 text-xs focus:border-cyan-800 focus:outline-none"
                                    />
                                </div>
                                <div>
                                    <label className="block text-[10px] text-gray-500 uppercase font-bold mb-1">{t.expectation}</label>
                                    <input 
                                        type="text" 
                                        value={formData.tooltip.expectation} 
                                        onChange={e => setFormData({...formData, tooltip: {...formData.tooltip, expectation: e.target.value}})}
                                        className="w-full bg-gray-950 border border-gray-800 p-2 rounded text-gray-400 text-xs focus:border-cyan-800 focus:outline-none"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 2. CORE IDENTITY (Fill remaining space, Scrollable) */}
                    <div className="flex-1 flex flex-col min-h-[300px] p-4 md:p-6 pt-2 relative">
                        <div className="flex flex-wrap justify-between items-center mb-2 gap-2 flex-shrink-0">
                             <label className="text-xs font-bold text-cyan-400 uppercase tracking-widest flex items-center gap-2">
                                <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full shadow-[0_0_5px_currentColor]"></span>
                                {t.coreIdentity}
                             </label>
                             <div className="flex gap-2 flex-wrap">
                                <button 
                                    onClick={handleExportJson}
                                    className="flex items-center gap-1 text-[10px] bg-gray-900 hover:bg-gray-700 text-gray-300 px-2 py-1 rounded border border-gray-700 transition-colors"
                                >
                                    <DownloadIcon className="w-3 h-3" />
                                    JSON
                                </button>
                                <div className="w-px h-4 bg-gray-700 mx-1 hidden sm:block"></div>
                                <input 
                                    type="file" 
                                    ref={fileInputRef} 
                                    className="hidden" 
                                    accept=".md,.txt,.json"
                                    onChange={handleFileUpload}
                                />
                                <button 
                                    onClick={() => fileInputRef.current?.click()}
                                    className="flex items-center gap-1 text-[10px] bg-gray-900 hover:bg-gray-700 text-gray-300 px-2 py-1 rounded border border-gray-700 transition-colors"
                                >
                                    <DownloadIcon className="w-3 h-3 rotate-180" />
                                    Load
                                </button>
                             </div>
                        </div>
                        <div className="flex-1 relative group border border-gray-700 rounded-md overflow-hidden">
                            <textarea 
                                value={formData.instruction} 
                                onChange={e => setFormData({...formData, instruction: e.target.value})}
                                className="w-full h-full min-h-[300px] bg-[#111827] p-4 text-gray-300 font-mono text-xs md:text-sm focus:border-cyan-500 focus:ring-1 focus:ring-cyan-900 focus:outline-none resize-y leading-relaxed shadow-inner selection:bg-cyan-900 selection:text-cyan-100 custom-scrollbar block"
                                spellCheck={false}
                                placeholder="# PASTE YOUR AGENT SYSTEM PROMPT HERE..."
                            />
                        </div>
                    </div>
                </div>
            ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-gray-500 gap-2 h-full">
                    <div className="w-12 h-12 rounded-full border border-gray-800 flex items-center justify-center">
                        <SettingsIcon className="w-6 h-6 opacity-20" />
                    </div>
                    <p className="text-sm">{t.selectModule}</p>
                </div>
            )}

            {/* Footer Actions */}
            <div className="flex-shrink-0 p-3 md:p-4 border-t border-gray-700 flex flex-wrap justify-between items-center bg-gray-800 z-10 gap-2">
                <div className="flex gap-2">
                    {editMode === 'edit' && formData?.isSystemModified && (
                        <button 
                            onClick={() => onReset(formData.id)}
                            className="px-2 md:px-3 py-2 bg-yellow-900/10 border border-yellow-800 text-yellow-600 hover:text-yellow-400 text-xs rounded hover:bg-yellow-900/20 transition-colors font-bold uppercase tracking-wide"
                        >
                            {t.resetFactory}
                        </button>
                    )}
                     {editMode === 'edit' && formData?.isCustom && (
                        <button 
                            onClick={() => { onDelete(formData.id); setSelectedId(null); setFormData(null); }}
                            className="px-2 md:px-3 py-2 bg-red-900/10 border border-red-900 text-red-700 hover:text-red-400 text-xs rounded hover:bg-red-900/20 transition-colors font-bold uppercase tracking-wide"
                        >
                            {t.delete}
                        </button>
                    )}
                </div>
                <button 
                    onClick={handleSave}
                    disabled={!formData}
                    className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded transition-all shadow-[0_0_15px_rgba(6,182,212,0.2)] hover:shadow-[0_0_20px_rgba(6,182,212,0.4)] disabled:opacity-50 disabled:shadow-none w-full md:w-auto text-sm"
                >
                    {t.save}
                </button>
            </div>
        </div>
        
        </div>
        
        {/* Corner Resizer (Desktop Only) */}
        <div 
            className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 hidden md:flex items-end justify-end p-1 group"
            onMouseDown={startResizingModal}
        >
            <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-cyan-400 transition-colors"></div>
        </div>

      </div>
    </div>
  );
};