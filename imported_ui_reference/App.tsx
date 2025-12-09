
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Header } from './components/Header';
import { ControlPanel } from './components/ControlPanel';
import { WorkflowCanvas } from './components/WorkflowCanvas';
import { OutputPanel } from './components/OutputPanel';
import { UserGuide } from './components/UserGuide';
import { Resizer } from './components/Resizer';
import { runAcosWorkflow, suggestWorkflow, generateProtocolFromContext, regenerateSystemProtocols, generateCustomProtocol } from './services/geminiService';
import type { CognitiveModule, ApiState, Risultante, WorkflowStep, PreconfiguredIntent, Source } from './types';
import { DEFAULT_COGNITIVE_MODULES, PRECONFIGURED_INTENTS } from './constants';
import { Language } from './translations';
import { AgentSimulatorModal } from './components/AgentSimulatorModal';
import { ChevronDownIcon } from './components/icons/ChevronDownIcon';
import { TooltipGlobalContext } from './components/Tooltip';
import { CpuIcon } from './components/icons/CpuIcon';

const useIsDesktop = () => {
  const [isDesktop, setIsDesktop] = useState(true);
  useEffect(() => {
    const checkSize = () => setIsDesktop(window.innerWidth >= 1024);
    checkSize();
    window.addEventListener('resize', checkSize);
    return () => window.removeEventListener('resize', checkSize);
  }, []);
  return isDesktop;
};

const STORAGE_KEY = 'mms_custom_modules';

// Configuration Constants
const COLLAPSE_THRESHOLD = 250; // Width below which the panel enters "Squeezed" visual state
const MIN_TAB_WIDTH = 48;       // The hard minimum width (w-12)
const RESIZER_WIDTH = 16;       // Approximate width of resizer area

const App: React.FC = () => {
  // --- Application State ---
  
  // Tooltip State
  const [areTooltipsEnabled, setAreTooltipsEnabled] = useState(true);

  // Initialize Modules
  const [availableModules, setAvailableModules] = useState<CognitiveModule[]>(() => {
      try {
          const saved = localStorage.getItem(STORAGE_KEY);
          if (saved) {
              const customModules: CognitiveModule[] = JSON.parse(saved);
              const merged = [...DEFAULT_COGNITIVE_MODULES];
              customModules.forEach(savedMod => {
                  const index = merged.findIndex(m => m.id === savedMod.id);
                  if (index >= 0) merged[index] = savedMod;
                  else merged.push(savedMod);
              });
              return merged;
          }
      } catch (e) {
          console.error("Failed to load custom agents", e);
      }
      return DEFAULT_COGNITIVE_MODULES;
  });
  
  useEffect(() => {
      const modulesToSave = availableModules.filter(m => m.isCustom || m.isSystemModified);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(modulesToSave));
  }, [availableModules]);
  
  const [intent, setIntent] = useState<string>('');
  const [selectedModules, setSelectedModules] = useState<CognitiveModule[]>([]);
  const [apiState, setApiState] = useState<ApiState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [risultante, setRisultante] = useState<Risultante | null>(null);
  const [processingSteps, setProcessingSteps] = useState<WorkflowStep[]>([]);
  const [activeStepIndex, setActiveStepIndex] = useState<number>(-1);
  const [isGuideOpen, setIsGuideOpen] = useState<boolean>(true);
  const [lastPrompt, setLastPrompt] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[] | undefined>(undefined);
  const [isAutoSelecting, setIsAutoSelecting] = useState(false);
  const [useContext, setUseContext] = useState(false);
  const [systemProtocols, setSystemProtocols] = useState<PreconfiguredIntent[]>(PRECONFIGURED_INTENTS);
  const [customProtocols, setCustomProtocols] = useState<PreconfiguredIntent[]>([]);
  const [simulatorModule, setSimulatorModule] = useState<CognitiveModule | null>(null);
  const [isSimulatorOpen, setIsSimulatorOpen] = useState(false);
  const [language, setLanguage] = useState<Language>('en');

  // --- Layout State ---
  const isDesktop = useIsDesktop();
  const [containerWidth, setContainerWidth] = useState(window.innerWidth);
  const [leftWidth, setLeftWidth] = useState(350);
  const [middleWidth, setMiddleWidth] = useState(500);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Measure Container Width on Resize
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (let entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Calculate derived Right Width
  const overhead = isDesktop ? (RESIZER_WIDTH * 2) : 0;
  // While dragging, leftWidth might be small (e.g. 100px), we use it directly.
  // When collapsed (0), we usually treat it as MIN_TAB_WIDTH (48px) for calculation purposes IF it's visible.
  const effectiveLeft = leftWidth === 0 ? MIN_TAB_WIDTH : leftWidth; 
  const effectiveMiddle = middleWidth === 0 ? MIN_TAB_WIDTH : middleWidth;
  
  const rightWidth = Math.max(0, containerWidth - effectiveLeft - effectiveMiddle - overhead);

  // Visual State Triggers
  const isLeftSqueezed = leftWidth > 0 && leftWidth < COLLAPSE_THRESHOLD;
  const isMiddleSqueezed = middleWidth > 0 && middleWidth < COLLAPSE_THRESHOLD;
  const isRightSqueezed = rightWidth < COLLAPSE_THRESHOLD;

  // --- Helper: Get System DNA ---
  const getSystemDNA = useCallback(() => {
      const orchestrator = availableModules.find(m => m.id === 'AWO.sys') || DEFAULT_COGNITIVE_MODULES.find(m => m.id === 'AWO.sys')!;
      const selectedKernel = selectedModules.find(m => m.id.includes('MMS.kernel'));
      const availableKernel = availableModules.find(m => m.id === 'MMS.kernel.v3') || availableModules.find(m => m.id === 'MMS.kernel.v2') || DEFAULT_COGNITIVE_MODULES.find(m => m.id === 'MMS.kernel.v3')!;
      const finalKernel = selectedKernel || availableKernel;
      return { orchestrator: orchestrator.instruction, kernel: finalKernel.instruction };
  }, [availableModules, selectedModules]);

  // --- Module Handlers ---
  const handleUpdateModule = useCallback((updatedModule: CognitiveModule) => {
      setAvailableModules(prev => prev.map(m => {
              if (m.id === updatedModule.id) {
                  const isSystem = DEFAULT_COGNITIVE_MODULES.some(dm => dm.id === m.id);
                  return { ...updatedModule, isSystemModified: isSystem ? true : undefined };
              }
              return m;
          }));
      setSelectedModules(prev => prev.map(m => m.id === updatedModule.id ? updatedModule : m));
  }, []);

  const handleCreateModule = useCallback((newModule: CognitiveModule) => {
      setAvailableModules(prev => [...prev, { ...newModule, isCustom: true }]);
  }, []);

  const handleImportModules = useCallback((importedModules: CognitiveModule[]) => {
      setAvailableModules(prev => {
          const existingIds = new Set(prev.map(m => m.id));
          const uniqueNew = importedModules.filter(m => !existingIds.has(m.id));
          if (uniqueNew.length === 0) return prev;
          const processedImports = uniqueNew.map(m => ({ ...m, isCustom: true }));
          return [...prev, ...processedImports];
      });
  }, []);

  const handleResetModule = useCallback((moduleId: string) => {
      const defaultModule = DEFAULT_COGNITIVE_MODULES.find(m => m.id === moduleId);
      if (defaultModule) {
          setAvailableModules(prev => prev.map(m => m.id === moduleId ? defaultModule : m));
          setSelectedModules(prev => prev.map(m => m.id === moduleId ? defaultModule : m));
      }
  }, []);

  const handleDeleteModule = useCallback((moduleId: string) => {
      setAvailableModules(prev => prev.filter(m => m.id !== moduleId));
      setSelectedModules(prev => prev.filter(m => m.id !== moduleId));
  }, []);

  const handleSelectModule = useCallback((moduleId: string) => {
    setSelectedModules(prev => {
        const exists = prev.some(m => m.id === moduleId);
        if (exists) return prev.filter(m => m.id !== moduleId);
        else {
            const moduleToAdd = availableModules.find(m => m.id === moduleId);
            return moduleToAdd ? [...prev, moduleToAdd] : prev;
        }
    });
  }, [availableModules]);

  const handleRemoveModule = useCallback((moduleId: string) => {
      setSelectedModules(prev => prev.filter(m => m.id !== moduleId));
  }, []);

  const handleSimulateModule = useCallback((module: CognitiveModule) => {
      setSimulatorModule(module);
      setIsSimulatorOpen(true);
  }, []);

  const handleClearWorkflow = useCallback(() => {
    setSelectedModules([]);
    setIntent('');
    setRisultante(null);
    setProcessingSteps([]);
    setApiState('idle');
    setError(null);
    setActiveStepIndex(-1);
    setLastPrompt(null);
    setSources(undefined);
    setUseContext(false);
  }, []);

  const handleApplyPreset = useCallback((preset: PreconfiguredIntent) => {
    setIntent(preset.text);
    const modulesToSelect = preset.recommendedModules
      .map(id => availableModules.find(m => m.id === id))
      .filter((m): m is CognitiveModule => !!m);
    setSelectedModules(modulesToSelect);
    setRisultante(null);
    setProcessingSteps([]);
    setApiState('idle');
    setActiveStepIndex(-1);
    setSources(undefined);
    setUseContext(false);
  }, [availableModules]);

  const handleAddCustomProtocol = useCallback((protocol: PreconfiguredIntent) => {
      setCustomProtocols(prev => [...prev, protocol]);
  }, []);

  const handleRegenerateSystemProtocols = useCallback(async () => {
      try {
          const { orchestrator } = getSystemDNA();
          const newProtocols = await regenerateSystemProtocols(availableModules, orchestrator);
          setSystemProtocols(newProtocols);
      } catch (e) {
          console.error("Failed to regenerate protocols", e);
          setError("Orchestrator failed to regenerate system protocols.");
      }
  }, [availableModules, getSystemDNA]);

  const handleForgeFromResult = useCallback(async () => {
      if (!risultante) return;
      setApiState('loading');
      try {
          const { orchestrator } = getSystemDNA();
          const newProtocol = await generateProtocolFromContext(intent, risultante, orchestrator);
          handleAddCustomProtocol(newProtocol);
          setApiState('idle');
      } catch (e) {
          console.error(e);
          setError("Failed to forge protocol from result.");
          setApiState('error');
      }
  }, [risultante, intent, handleAddCustomProtocol, getSystemDNA]);

  const handleAutoSelect = useCallback(async () => {
      if(!intent.trim()) return;
      setIsAutoSelecting(true);
      if (!useContext) {
          setRisultante(null);
          setProcessingSteps([]);
          setActiveStepIndex(-1);
          setSources(undefined);
      }
      setSelectedModules([]);
      try {
          const { orchestrator } = getSystemDNA();
          const suggestedIds = await suggestWorkflow(intent, availableModules, orchestrator);
          const modulesToSelect = suggestedIds
             .map(id => availableModules.find(m => m.id === id))
             .filter((m): m is CognitiveModule => !!m);
          setSelectedModules(modulesToSelect);
      } catch (e) {
          console.error(e);
          setError("Failed to auto-orchestrate modules.");
      } finally {
          setIsAutoSelecting(false);
      }
  }, [intent, useContext, availableModules, getSystemDNA]);

  const handleActivateEngine = async () => {
    if (!intent.trim() || selectedModules.length === 0) {
      setError("Please provide an intent and select at least one module.");
      return;
    }
    setApiState('loading');
    setError(null);
    if (!useContext) setRisultante(null);
    setProcessingSteps([]);
    setLastPrompt(null);
    setSources(undefined);
    setActiveStepIndex(0);

    const stepInterval = setInterval(() => {
      setActiveStepIndex(prev => {
        if (prev < selectedModules.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 700);

    try {
      const contextToPass = useContext ? risultante : null;
      const { orchestrator, kernel } = getSystemDNA();
      const result = await runAcosWorkflow(intent, selectedModules, orchestrator, kernel, contextToPass);
      clearInterval(stepInterval);
      setRisultante(result.risultante);
      setProcessingSteps(result.steps);
      setLastPrompt(result.prompt);
      setSources(result.sources);
      setApiState('success');
      setActiveStepIndex(selectedModules.length);
    } catch (e) {
      clearInterval(stepInterval);
      console.error(e);
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      setError(`Failed to run workflow. ${errorMessage}`);
      setApiState('error');
      setActiveStepIndex(-1);
    }
  };

  // --- Logic: Resizing with Hard Constraints & Progressive Squeeze Effect (Bi-Directional) ---
  const startResizing = useCallback((panel: 'left' | 'middle') => (mouseDownEvent: React.MouseEvent) => {
    mouseDownEvent.preventDefault();
    setIsDragging(true);
    const startX = mouseDownEvent.clientX;
    
    // Snapshot of initial state
    const startLeftW = leftWidth || 0;
    const startMiddleW = middleWidth || 0;
    const currentContainerW = containerRef.current ? containerRef.current.clientWidth : window.innerWidth;
    
    // Mutable tracking vars to ensure onMouseUp has the absolute latest values
    let currLeft = startLeftW;
    let currMiddle = startMiddleW;

    const onMouseMove = (mouseMoveEvent: MouseEvent) => {
      const deltaX = mouseMoveEvent.clientX - startX;
      const overhead = (RESIZER_WIDTH * 2);
      const minW = MIN_TAB_WIDTH;

      if (panel === 'left') {
          // --- RESIZING LEFT HANDLE ---
          let newLeft = Math.max(minW, startLeftW + deltaX);

          // PUSH LOGIC: Left pushes Middle -> Middle pushes Right
          // We calculate what Middle needs to be to fit Left.
          // Constraints: Middle >= minW, Right >= minW.
          
          const maxLeftAllowed = currentContainerW - overhead - minW - minW; // Max if Middle & Right are minW
          if (newLeft > maxLeftAllowed) newLeft = maxLeftAllowed;

          // How much space is left for Middle + Right?
          const spaceForRest = currentContainerW - newLeft - overhead;
          
          // Current logic: Preserve Right's existing width as much as possible?
          // No, default behavior is Middle shrinks first.
          
          // Logic:
          // 1. Calculate space available for Middle if Right stays as is (or minW).
          // 2. Shrink Middle.
          
          // Let's assume Middle handles the pressure first.
          const currentRight = currentContainerW - startLeftW - overhead - startMiddleW;
          
          // If Middle shrinks to minW, then Right shrinks.
          let availableForMiddle = spaceForRest - Math.max(minW, currentRight); 
          
          // Can we fit newLeft?
          if (availableForMiddle < minW) {
             // We need to shrink Right to make space for Middle=minW
             // But we handled maxLeftAllowed above, so this block shouldn't theoretically be reached if logic is sound.
             // Just strictly cap Middle.
             currMiddle = minW;
             setMiddleWidth(0); // visual snap
          } else {
             // Middle can absorb it?
             // Actually, resizing Left usually eats into Middle directly.
             // newMiddle = startMiddle - deltaX (roughly)
             
             let newMiddle = startMiddleW - (newLeft - startLeftW);
             
             // If newMiddle < minW, we clamp it and Left stops growing?
             // Or we push into Right? Standard UI: Pushes neighbor.
             if (newMiddle < minW) {
                 newMiddle = minW;
                 // Recalculate Left based on clamped Middle
                 // But wait, if we push Middle to min, does Left keep growing and push Right?
                 // In this simple model: Left resizes Middle. Resizing Middle resizes Right.
                 // So drag Left -> impacts Middle.
                 // If Middle hits minW, Left stops. (We don't cascade push Right in this implementation to keep it simple).
                 newLeft = currentContainerW - overhead - newMiddle - currentRight;
             }
             
             currMiddle = newMiddle;
             setMiddleWidth(newMiddle);
          }
          
          currLeft = newLeft;
          setLeftWidth(newLeft);

      } else {
          // --- RESIZING MIDDLE HANDLE ---
          // This handle sits between Middle and Right.
          // Moving it changes Middle width directly.
          
          let newMiddle = Math.max(minW, startMiddleW + deltaX);
          
          // Constraint: Left is fixed. Right must be >= minW.
          const currentLeft = (currLeft === 0 ? minW : currLeft);
          const maxMiddle = currentContainerW - currentLeft - overhead - minW;
          
          if (newMiddle > maxMiddle) {
              newMiddle = maxMiddle;
          }
          
          currMiddle = newMiddle;
          setMiddleWidth(newMiddle);
      }
    };

    const onMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.userSelect = '';
      
      let finalLeft = currLeft;
      let finalMiddle = currMiddle;
      const minW = MIN_TAB_WIDTH;
      const threshold = COLLAPSE_THRESHOLD;

      // --- FULL SANITIZATION PASS ---
      // We check L, M, R sequentially. If any is in "Ghost Zone" (min < w < threshold), we snap it to 0.
      
      // 1. Sanitize LEFT
      if (finalLeft > 0 && finalLeft < threshold) {
          finalLeft = 0;
      }
      // Note: If Left snaps to 0, it frees space. Who gets it? 
      // Usually Middle. But let's calculate exact positions at the end.

      // 2. Sanitize MIDDLE
      if (finalMiddle > 0 && finalMiddle < threshold) {
          // Middle snaps closed.
          // Space freed goes to LEFT if we were dragging Left handle (user pulled left back).
          // Space freed goes to RIGHT if we were dragging Middle handle (user pulled middle back).
          
          // To simplify: If Middle closes, we extend the panel that was being expanded.
          // Or simpler: Left absorbs the gap to maintain Right's position (stability).
          
          const freed = finalMiddle; // effective width to 0
          
          finalMiddle = 0;
          
          if (panel === 'left') {
              // We were adjusting Left/Middle boundary. 
              // If Middle collapsed, Left likely pushed it. So Left takes the space.
              // BUT if we were shrinking Left, and Middle was small... ambiguous.
              // Let's assume Left takes it for stability.
              finalLeft += (freed - minW); // Note: 0 is visually 48px, so we add the diff.
              if(finalLeft < minW) finalLeft = minW; // Safety
          }
          // If panel == 'middle', doing nothing effectively gives space to Right (since R = Cont - L - M)
      }

      // 3. Sanitize RIGHT (The Ghost Check)
      // Calculate resulting Right
      const effL = finalLeft === 0 ? minW : finalLeft;
      const effM = finalMiddle === 0 ? minW : finalMiddle;
      const availR = currentContainerW - effL - effM - overhead;
      
      if (availR > minW && availR < threshold) {
          // Right is in Ghost Zone. Snap it closed.
          // We need to fill the gap (availR - minW).
          // Priority: Expand Middle if it's open. Else expand Left.
          const gap = availR - minW;
          
          if (finalMiddle > 0) {
              finalMiddle += gap;
          } else {
              finalLeft += gap;
          }
      }

      // Final Apply
      setLeftWidth(finalLeft);
      setMiddleWidth(finalMiddle);
    };

    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [leftWidth, middleWidth]);


  // --- Gutter Components (Reused for Squeeze State) ---
  
  const LeftGutterButton = (
    <div className="w-full flex-shrink-0 h-full flex flex-col">
       <button
          onClick={() => {
              // Open Left. 
              // We need 350px.
              // Shrink Middle first. If Middle goes Ghost -> Close Middle.
              // Then Shrink Right (implicitly).
              
              if(!containerRef.current) return;
              const w = containerRef.current.clientWidth;
              const overhead = RESIZER_WIDTH * 2;
              const minW = MIN_TAB_WIDTH;
              
              const targetLeft = 350;
              
              // Current Middle
              let currM = middleWidth === 0 ? minW : middleWidth;
              // Current Right space?
              // Actually, simplified logic:
              // Try to take space from Middle.
              
              // Max possible Left = W - overhead - minW(M) - minW(R)
              const maxL = w - overhead - minW - minW;
              const finalL = Math.min(targetLeft, maxL);
              
              // Remaining space for M + R
              const remaining = w - overhead - finalL;
              
              // We want to keep Right > threshold if possible, or minW.
              // And Middle > threshold or minW.
              
              // Naive approach: Set Left. Let the layout logic in render handle Right.
              // But we need to update Middle explicitly.
              
              // New Middle = Remaining - Right.
              // Right is the passive variable. We must set Middle such that Right is valid.
              // Let's prioritize keeping Right as is, shrinking Middle.
              
              const currR = w - overhead - (leftWidth===0?minW:leftWidth) - currM;
              
              // Desired Middle = Remaining - currR
              let newM = remaining - currR;
              
              // If newM < threshold, close Middle.
              if (newM < COLLAPSE_THRESHOLD) {
                  newM = 0; // Close Middle
                  // Now Right gets all remaining space (Remaining - 48).
                  // Is that valid?
                  const newR = remaining - minW;
                  if (newR < COLLAPSE_THRESHOLD) {
                      // Right is also too small?
                      // This shouldn't happen on desktop usually unless window is tiny.
                      // If so, we just set M=0, L=350, Right gets whatever (ghost state inevitable on tiny screens).
                  }
              }
              
              setLeftWidth(finalL);
              setMiddleWidth(newM);
          }}
          className="h-full w-full bg-gray-900/95 backdrop-blur-sm border-r border-gray-800 rounded-r-2xl flex flex-col items-center justify-between py-8 hover:bg-gray-800 hover:border-purple-500/60 transition-all duration-300 group shadow-[5px_0_15px_rgba(0,0,0,0.3)] cursor-pointer relative z-10"
       >
            <div className="p-2 rounded-lg bg-gray-800/50 text-purple-500 group-hover:bg-purple-500 group-hover:text-white transition-all mb-4 shadow-[0_0_10px_rgba(168,85,247,0.2)]">
                <ChevronDownIcon className="w-5 h-5 -rotate-90" />
            </div>
            <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-purple-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                    <span>Control Matrix</span>
                    <span className="w-px h-8 bg-gray-700 group-hover:bg-purple-500/50 transition-colors"></span>
                </div>
            </div>
            <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-purple-400 shadow-[0_0_5px_rgba(168,85,247,0.8)] transition-colors mt-4"></div>
       </button>
    </div>
  );

  const MiddleGutterButton = (
    <div className="w-full flex-shrink-0 h-full flex flex-col mx-1">
        <button 
            onClick={() => {
                if (containerRef.current) {
                    const w = containerRef.current.clientWidth;
                    const overhead = RESIZER_WIDTH * 2;
                    const minW = MIN_TAB_WIDTH;
                    
                    const curL = leftWidth === 0 ? minW : leftWidth;
                    const targetM = 500;
                    
                    // Max M = W - overhead - curL - minW(R)
                    const maxM = w - overhead - curL - minW;
                    let finalM = Math.min(targetM, maxM);
                    
                    // Check if Right becomes Ghost
                    const resultingR = w - overhead - curL - finalM;
                    if (resultingR < COLLAPSE_THRESHOLD && resultingR > minW) {
                        // Right is squeezed. 
                        // We must Close Right? Or Shrink Middle?
                        // User clicked Open Middle. Priority to Middle.
                        // Close Right.
                        // To Close Right, we expand Middle to consume the space.
                        // Right -> minW. Middle -> W - overhead - curL - minW.
                        finalM = w - overhead - curL - minW; 
                    }
                    
                    setMiddleWidth(finalM);
                } else {
                    setMiddleWidth(500);
                }
            }}
            className="h-full w-full bg-gray-900/95 backdrop-blur-sm border-x border-gray-800 rounded-xl flex flex-col items-center justify-between py-8 hover:bg-gray-800 hover:border-blue-500/60 transition-all duration-300 group shadow-[0_0_15px_rgba(0,0,0,0.3)] cursor-pointer relative z-10"
        >
            <div className="p-2 rounded-lg bg-gray-800/50 text-blue-500 group-hover:bg-blue-500 group-hover:text-white transition-all mb-4 shadow-[0_0_10px_rgba(59,130,246,0.2)]">
                <CpuIcon className="w-5 h-5" />
            </div>
            <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-blue-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                    <span>Workflow Canvas</span>
                    <span className="w-px h-8 bg-gray-700 group-hover:bg-blue-500/50 transition-colors"></span>
                </div>
            </div>
            <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-blue-400 shadow-[0_0_5px_rgba(59,130,246,0.8)] transition-colors mt-4"></div>
        </button>
    </div>
  );

  return (
    <TooltipGlobalContext.Provider value={areTooltipsEnabled}>
      <div className={`min-h-screen bg-gray-900 text-gray-200 font-sans flex flex-col ${isDragging ? 'cursor-col-resize' : ''}`}>
        <UserGuide isOpen={isGuideOpen} onClose={() => setIsGuideOpen(false)} />
        <Header onOpenGuide={() => setIsGuideOpen(true)} areTooltipsEnabled={areTooltipsEnabled} onToggleTooltips={setAreTooltipsEnabled} />
        <AgentSimulatorModal isOpen={isSimulatorOpen} onClose={() => setIsSimulatorOpen(false)} module={simulatorModule} onCreateModule={handleCreateModule} />

        <main ref={containerRef} className="flex-grow p-4 overflow-hidden flex flex-col lg:flex-row h-[calc(100vh-80px)] w-full relative">
          
          {/* --- LEFT PANEL --- */}
          {/* 
            Logic: 
            1. If leftWidth == 0 -> Render fixed 48px gutter (Snapped State).
            2. If leftWidth < THRESHOLD -> Render Gutter Content but with `leftWidth` (Squeezed State).
            3. Else -> Render Full Content.
          */}
          <div 
              className={`flex-shrink-0 h-full mb-4 lg:mb-0 ${isDragging ? '' : 'transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)]'}`}
              style={{ width: isDesktop ? (leftWidth === 0 ? '48px' : `${leftWidth}px`) : '100%' }}
          >
              {isDesktop && (leftWidth === 0 || isLeftSqueezed) ? (
                  LeftGutterButton
              ) : (
                  <ControlPanel
                      intent={intent}
                      onIntentChange={setIntent}
                      onSelectModule={handleSelectModule}
                      onClearWorkflow={handleClearWorkflow}
                      onActivateEngine={handleActivateEngine}
                      onApplyPreset={handleApplyPreset}
                      onAutoSelect={handleAutoSelect}
                      isLoading={apiState === 'loading'}
                      isAutoSelecting={isAutoSelecting}
                      selectedModuleIds={selectedModules.map(m => m.id)}
                      language={language}
                      risultante={risultante}
                      useContext={useContext}
                      onToggleContext={setUseContext}
                      systemProtocols={systemProtocols} 
                      customProtocols={customProtocols}
                      onAddCustomProtocol={handleAddCustomProtocol}
                      onForgeFromResult={handleForgeFromResult}
                      onRegenerateSystemProtocols={handleRegenerateSystemProtocols} 
                      availableModules={availableModules}
                      onUpdateModule={handleUpdateModule}
                      onCreateModule={handleCreateModule}
                      onImportModules={handleImportModules}
                      onResetModule={handleResetModule}
                      onDeleteModule={handleDeleteModule}
                  />
              )}
          </div>

          {/* Resizer 1 */}
          <div className={`hidden lg:flex h-full flex-shrink-0 items-center justify-center bg-gray-900 ${leftWidth === 0 ? 'hidden' : ''}`}>
              <Resizer onMouseDown={startResizing('left')} isVisible={true} />
          </div>

          {/* --- MIDDLE PANEL --- */}
          <div 
              className={`h-full flex flex-col mb-4 lg:mb-0 ${isDragging ? '' : 'transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)]'} flex-shrink-0`}
              style={{ width: isDesktop ? (middleWidth === 0 ? '48px' : `${middleWidth}px`) : '100%' }}
          >
              {isDesktop && (middleWidth === 0 || isMiddleSqueezed) ? (
                  MiddleGutterButton
              ) : (
                  <WorkflowCanvas 
                      modules={selectedModules}
                      activeStepIndex={activeStepIndex}
                      language={language}
                      onRemoveModule={handleRemoveModule}
                      onSimulateModule={handleSimulateModule}
                  />
              )}
          </div>

          {/* Resizer 2 */}
          <div className={`hidden lg:flex h-full flex-shrink-0 items-center justify-center bg-gray-900 ${middleWidth === 0 || rightWidth === 0 ? 'hidden' : ''}`}>
              <Resizer onMouseDown={startResizing('middle')} isVisible={true} />
          </div>

          {/* --- RIGHT PANEL --- */}
          <div className="flex-1 min-w-0 h-full overflow-hidden flex flex-col">
             {isRightSqueezed && isDesktop ? (
                // SQUEEZED GUTTER STATE (But filling the remaining width)
                <div className="h-full w-full flex flex-col ml-auto">
                    <button
                        onClick={() => {
                           if (containerRef.current) {
                               const w = containerRef.current.clientWidth;
                               const overhead = (RESIZER_WIDTH * 2);
                               const minW = MIN_TAB_WIDTH;
                               const curL = leftWidth === 0 ? minW : leftWidth;
                               const curM = middleWidth === 0 ? minW : middleWidth;
                               
                               const targetR = 400;
                               // Available for R = W - overhead - L - M
                               const availR = w - overhead - curL - curM;
                               const needed = targetR - availR;
                               
                               if (needed <= 0) {
                                   // Already have space? No action needed usually, but logic shouldn't reach here if button visible.
                                   return; 
                               }

                               // We need space. 
                               // Strategy: Shrink Middle first.
                               // If Middle hits Ghost -> Close Middle.
                               // If still needed -> Shrink Left.
                               
                               let newM = Math.max(minW, curM - needed);
                               let stillNeeded = needed - (curM - newM);
                               
                               if (newM < COLLAPSE_THRESHOLD && newM > minW) {
                                   // Middle is in Ghost Zone. Close it.
                                   newM = 0; // Visual 48px
                                   // We gained extra space by closing it fully.
                                   // Extra = (curM or newM?) -> Logic: oldM - 48.
                               }
                               
                               // Calculate Right with newM
                               const tempM = newM === 0 ? minW : newM;
                               const tempR = w - overhead - curL - tempM;
                               
                               if (tempR < targetR) {
                                   // Still need space. Shrink Left.
                                   const neededFromLeft = targetR - tempR;
                                   let newL = Math.max(minW, curL - neededFromLeft);
                                   
                                   if (newL < COLLAPSE_THRESHOLD && newL > minW) {
                                       newL = 0;
                                   }
                                   setLeftWidth(newL);
                               }
                               
                               setMiddleWidth(newM);
                           }
                        }}
                        className="
                            h-full w-full
                            bg-gray-900/95 backdrop-blur-sm
                            border-l border-gray-800
                            rounded-l-2xl
                            flex flex-col items-center justify-between py-8
                            hover:bg-gray-800 hover:border-cyan-500/60
                            transition-all duration-300
                            group
                            shadow-[-5px_0_15px_rgba(0,0,0,0.3)]
                            cursor-pointer
                            relative z-10
                        "
                    >
                        <div className="p-2 rounded-lg bg-gray-800/50 text-cyan-500 group-hover:bg-cyan-500 group-hover:text-white transition-all mb-4 shadow-[0_0_10px_rgba(6,182,212,0.2)]">
                             <ChevronDownIcon className="w-5 h-5 rotate-90" />
                        </div>

                        <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                            <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-cyan-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                                 <span>Output Manifestation</span>
                                 <span className="w-px h-8 bg-gray-700 group-hover:bg-cyan-500/50 transition-colors"></span>
                            </div>
                        </div>
                        
                        <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-cyan-400 shadow-[0_0_5px_rgba(6,182,212,0.8)] transition-colors mt-4"></div>
                    </button>
                </div>
             ) : (
                // FULL CONTENT STATE
                <OutputPanel
                  state={apiState}
                  error={error}
                  risultante={risultante}
                  processingSteps={processingSteps}
                  prompt={lastPrompt}
                  sources={sources}
                  language={language}
                />
             )}
          </div>

        </main>
      </div>
    </TooltipGlobalContext.Provider>
  );
};

export default App;
