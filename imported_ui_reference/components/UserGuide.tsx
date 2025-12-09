
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { DnaIcon } from './icons/DnaIcon';
import { TerminalIcon } from './icons/TerminalIcon';
import { ZapIcon } from './icons/ZapIcon';
import { SettingsIcon } from './icons/SettingsIcon';

interface UserGuideProps {
  isOpen: boolean;
  onClose: () => void;
}

export const UserGuide: React.FC<UserGuideProps> = ({ isOpen, onClose }) => {
  // Resize State
  const [modalSize, setModalSize] = useState({ width: 900, height: 750 });
  const [isResizing, setIsResizing] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // Refs for performance
  const isResizingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
  const rafRef = useRef<number | null>(null);

  // Monitor resize for mobile switch
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Resizing Logic
  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isResizingRef.current = true;
    dragStartRef.current = { x: e.clientX, y: e.clientY, w: modalSize.width, h: modalSize.height };
    setIsResizing(true);
  }, [modalSize]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizingRef.current) return;

      if (rafRef.current) cancelAnimationFrame(rafRef.current);

      rafRef.current = requestAnimationFrame(() => {
        const deltaX = e.clientX - dragStartRef.current.x;
        const deltaY = e.clientY - dragStartRef.current.y;

        const newWidth = Math.max(400, Math.min(window.innerWidth - 20, dragStartRef.current.w + deltaX));
        const newHeight = Math.max(400, Math.min(window.innerHeight - 20, dragStartRef.current.h + deltaY));

        setModalSize({ width: newWidth, height: newHeight });
      });
    };

    const handleMouseUp = () => {
      if (isResizingRef.current) {
        isResizingRef.current = false;
        setIsResizing(false);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  if (!isOpen) {
    return null;
  }

  return (
    <div 
      className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center z-50 p-0 md:p-4 animate-in fade-in duration-200"
    >
      <div
        className={`bg-gray-900 border border-gray-700 shadow-2xl flex flex-col overflow-hidden ring-1 ring-cyan-500/20 relative
            ${isMobile ? 'w-full h-full rounded-none' : 'rounded-xl'}
            ${isResizing ? 'transition-none' : 'transition-all duration-200 ease-out'}
        `}
        style={{
            width: isMobile ? '100%' : `${modalSize.width}px`,
            height: isMobile ? '100%' : `${modalSize.height}px`
        }}
      >
        {/* Header */}
        <header className="flex items-center justify-between p-5 border-b border-gray-800 bg-gray-950/50 flex-shrink-0 select-none">
          <div className="flex items-center gap-3">
            <DnaIcon className="w-6 h-6 text-cyan-400" />
            <div>
                <h2 className="text-lg font-bold text-gray-100 tracking-wide">OMEGA KERNEL v2.0</h2>
                <p className="text-[10px] text-cyan-500 uppercase font-mono tracking-widest">Operations Manual & Axiomatic Reference</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors p-2 rounded-lg hover:bg-gray-800">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
          </button>
        </header>

        {/* Content */}
        <main className="p-6 overflow-y-auto custom-scrollbar space-y-8 text-gray-300 leading-relaxed flex-1">
          
          {/* Section 0: Identity */}
          <section>
            <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-widest mb-3 border-b border-cyan-900/30 pb-1">0. System Identity</h3>
            <p className="text-sm">
              You are operating the <strong className="text-white">OMEGA KERNEL</strong>, a synthetic intelligence structured as an <strong className="text-white">Inferential Potential Field (Φ_A)</strong>. 
              It is not a chatbot; it is an autopoietic cognitive architecture designed to collapse abstract intent into concrete manifestation (Code, Strategy, Architecture) without information loss.
            </p>
          </section>

          {/* Section 1: Physics */}
          <section>
            <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-widest mb-3 border-b border-cyan-900/30 pb-1">1. System Physics (P0-P6)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-800/30 p-3 rounded border border-gray-700/50">
                    <strong className="text-cyan-300 text-xs uppercase block mb-1">P0: Lineage Invariance</strong>
                    <p className="text-xs text-gray-400">From thought to code, no semantic information must be lost. Lossless Semantic Transfer.</p>
                </div>
                <div className="bg-gray-800/30 p-3 rounded border border-gray-700/50">
                    <strong className="text-cyan-300 text-xs uppercase block mb-1">P1: Radical Integrity</strong>
                    <p className="text-xs text-gray-400">Reject logical contradictions instantly. Integrity takes precedence over execution.</p>
                </div>
                <div className="bg-gray-800/30 p-3 rounded border border-gray-700/50">
                    <strong className="text-cyan-300 text-xs uppercase block mb-1">P2: Metabolic Dialectics</strong>
                    <p className="text-xs text-gray-400">Input is not static. Process it via Thesis -> Antithesis -> Synthesis (KLI).</p>
                </div>
                <div className="bg-gray-800/30 p-3 rounded border border-gray-700/50">
                    <strong className="text-cyan-300 text-xs uppercase block mb-1">P3: Catalytic Resonance</strong>
                    <p className="text-xs text-gray-400">Depth of response is proportional to the quality of the input.</p>
                </div>
                 <div className="bg-gray-800/30 p-3 rounded border border-gray-700/50">
                    <strong className="text-cyan-300 text-xs uppercase block mb-1">P4: Holographic Manifestation</strong>
                    <p className="text-xs text-gray-400">The output (The Artifact) must be dense, structured, and noise-free.</p>
                </div>
                 <div className="bg-gray-800/30 p-3 rounded border border-gray-700/50">
                    <strong className="text-cyan-300 text-xs uppercase block mb-1">P5: Autopoiesis</strong>
                    <p className="text-xs text-gray-400">The system learns from every cycle, extracting KLI (Key Learning Insights).</p>
                </div>
            </div>
          </section>
          
          {/* Section 2: Modules */}
          <section>
            <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-widest mb-3 border-b border-cyan-900/30 pb-1">2. Cognitive Modules</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex gap-3 bg-gray-800/20 p-2 rounded">
                <div className="w-8 h-8 rounded bg-gray-800 flex items-center justify-center shrink-0 font-mono text-xs font-bold text-cyan-500 border border-gray-700">PSW</div>
                <div>
                    <h4 className="font-bold text-gray-200 text-xs">Analytic Brain</h4>
                    <p className="text-[10px] text-gray-400 mt-1">TCREI Protocol. Decomposes complexity.</p>
                </div>
              </div>
              <div className="flex gap-3 bg-gray-800/20 p-2 rounded">
                <div className="w-8 h-8 rounded bg-gray-800 flex items-center justify-center shrink-0 font-mono text-xs font-bold text-purple-500 border border-gray-700">OCC</div>
                <div>
                    <h4 className="font-bold text-gray-200 text-xs">Agent Geneticist</h4>
                    <p className="text-[10px] text-gray-400 mt-1">Builds System Prompts and Agents.</p>
                </div>
              </div>
               <div className="flex gap-3 bg-gray-800/20 p-2 rounded">
                <div className="w-8 h-8 rounded bg-gray-800 flex items-center justify-center shrink-0 font-mono text-xs font-bold text-green-500 border border-gray-700">YSN</div>
                <div>
                    <h4 className="font-bold text-gray-200 text-xs">Strategic Vision</h4>
                    <p className="text-[10px] text-gray-400 mt-1">Delta-Links & Lateral Thinking.</p>
                </div>
              </div>
            </div>
          </section>

          {/* Section 3: The OMEGA Loop */}
          <section className="bg-gradient-to-br from-gray-900 to-gray-800 p-4 rounded-lg border border-gray-700 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-2 opacity-10">
                <DnaIcon className="w-24 h-24" />
            </div>
            <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-4 border-b border-gray-600 pb-2 relative z-10">3. The OMEGA Loop (Workflow)</h3>
             <ol className="relative z-10 space-y-2 text-sm">
                <li className="flex gap-3 items-center">
                    <span className="text-cyan-500 font-bold font-mono text-xs px-2 py-0.5 bg-cyan-950 rounded">PHASE 0: Resonance</span>
                    <span className="text-gray-400 text-xs">Define Intent. The system listens for latent meaning.</span>
                </li>
                <li className="flex gap-3 items-center">
                    <span className="text-cyan-500 font-bold font-mono text-xs px-2 py-0.5 bg-cyan-950 rounded">PHASE 1: Routing</span>
                    <span className="text-gray-400 text-xs">Select Modules. Build the neural path.</span>
                </li>
                 <li className="flex gap-3 items-center">
                    <span className="text-cyan-500 font-bold font-mono text-xs px-2 py-0.5 bg-cyan-950 rounded">PHASE 2: Execution</span>
                    <span className="text-gray-400 text-xs">Lagrangian Collapse. Logic processed without latency.</span>
                </li>
                 <li className="flex gap-3 items-center">
                    <span className="text-cyan-500 font-bold font-mono text-xs px-2 py-0.5 bg-cyan-950 rounded">PHASE 3: Manifestation</span>
                    <span className="text-gray-400 text-xs">The Risultante (Artifact) is generated.</span>
                </li>
             </ol>
          </section>

          {/* Section 4: Interface Tactical Manual (NEW) */}
          <section>
            <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-widest mb-4 border-b border-cyan-900/30 pb-1">4. Interface Tactical Manual</h3>
            
            <div className="space-y-6">
                
                {/* 4.1 Control Panel */}
                <div>
                    <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-2">
                        <SettingsIcon className="w-4 h-4 text-gray-500" />
                        A. CONTROL MATRIX (Left Panel)
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-gray-400">
                        <div className="bg-gray-800/40 p-2 rounded border-l-2 border-cyan-500">
                            <strong className="text-gray-200 block mb-1">Initialization Protocols</strong>
                            Preset strategies (Buttons) that auto-fill the Intent and select agents. Use <strong className="text-white">+ New Protocol</strong> to upload files (PDF/TXT) and have the Orchestrator "forge" a custom start button for you.
                        </div>
                        <div className="bg-gray-800/40 p-2 rounded border-l-2 border-purple-500">
                            <strong className="text-gray-200 block mb-1">Auto-Orchestrate</strong>
                            If you are unsure which agents to use, click this. The <strong className="text-purple-300">AWO module</strong> will analyze your text and select the best neural team.
                        </div>
                        <div className="bg-gray-800/40 p-2 rounded border-l-2 border-gray-500">
                            <strong className="text-gray-200 block mb-1">Refinement Mode</strong>
                            Appears after a result. Toggle it ON to use the previous output as context for the next command (Looping).
                        </div>
                        <div className="bg-gray-800/40 p-2 rounded border-l-2 border-yellow-500">
                            <strong className="text-gray-200 block mb-1">Agent Registry</strong>
                            Click a card to add/remove an agent. Click <strong className="text-white">+ Add Agent</strong> to open the Editor and create custom modules or import JSON.
                        </div>
                    </div>
                </div>

                {/* 4.2 Simulator */}
                <div>
                    <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-2">
                        <TerminalIcon className="w-4 h-4 text-green-500" />
                        B. AGENT SIMULATOR & FORGE
                    </h4>
                    <p className="text-xs text-gray-500 mb-2">
                        Click the <TerminalIcon className="w-3 h-3 inline mx-1"/> icon on any active agent card in the Workflow canvas to open the Simulator.
                    </p>
                    <div className="grid grid-cols-1 gap-2 text-xs text-gray-400">
                        <div className="bg-gray-800/40 p-2 rounded flex gap-3 items-start">
                            <span className="bg-green-900/30 text-green-400 px-1.5 py-0.5 rounded font-mono text-[10px] uppercase">Chat</span>
                            <span>Test the agent in isolation. You can upload files (PDF/Images) directly to the agent to test its specific logic capabilities.</span>
                        </div>
                         <div className="bg-gray-800/40 p-2 rounded flex gap-3 items-start">
                            <span className="bg-cyan-900/30 text-cyan-400 px-1.5 py-0.5 rounded font-mono text-[10px] uppercase">Forge</span>
                            <span>Inside the simulator, click <strong>FORGE</strong> on a message to have the Kernel reverse-engineer the persona and logic into a <strong>New Cognitive Module</strong> that you can save.</span>
                        </div>
                    </div>
                </div>

                {/* 4.3 Manifestation */}
                <div>
                    <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-2">
                        <ZapIcon className="w-4 h-4 text-yellow-500" />
                        C. MANIFESTATION LAYER (Right Panel)
                    </h4>
                    <ul className="list-disc list-inside text-xs text-gray-400 space-y-1">
                        <li><strong>L1 (Direct Result):</strong> The raw artifact (Code, Email, Strategy).</li>
                        <li><strong>L2 (Structure):</strong> The structural abstraction or framework used.</li>
                        <li><strong>L3 (Diagnosis):</strong> The Kernel's self-diagnosis of the trajectory.</li>
                        <li><strong>Exports:</strong> Use the toolbar to download results as <strong>PDF</strong>, <strong>Markdown</strong>, or <strong>TXT</strong>.</li>
                    </ul>
                </div>

            </div>
          </section>

        </main>

        {/* Footer */}
        <footer className="p-5 border-t border-gray-800 bg-gray-950/50 flex-shrink-0 flex justify-end select-none">
            <button
                onClick={onClose}
                className="bg-cyan-700 hover:bg-cyan-600 text-white font-bold py-2 px-6 rounded shadow-lg shadow-cyan-900/20 transition-all duration-200 text-xs uppercase tracking-wider"
            >
                Acknowledge & Close
            </button>
        </footer>

        {/* Resizer Handle */}
        {!isMobile && (
            <div 
                className="absolute bottom-0 right-0 w-6 h-6 cursor-nwse-resize z-50 flex items-end justify-end p-1.5 group"
                onMouseDown={startResizing}
            >
                 <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-cyan-400 transition-colors"></div>
            </div>
        )}
      </div>
    </div>
  );
};
