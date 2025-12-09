import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { CognitiveModule, ChatMessage } from '../types';
import { TerminalIcon } from './icons/TerminalIcon';
import { generateAgentStarters, chatWithAgent, fileToText, fileToBase64, generateModuleFromHistory } from '../services/geminiService';
import { Resizer } from './Resizer';
import { CopyIcon } from './icons/CopyIcon';
import { PaperclipIcon } from './icons/PaperclipIcon';
import { DnaIcon } from './icons/DnaIcon';
import { jsPDF } from "jspdf";
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import { SendIcon } from './icons/SendIcon';
import { TrashIcon } from './icons/TrashIcon';
import { StopCircleIcon } from './icons/StopCircleIcon';
import { RefreshCcwIcon } from './icons/RefreshCcwIcon';
import { Tooltip } from './Tooltip';

interface AgentSimulatorModalProps {
  isOpen: boolean;
  onClose: () => void;
  module: CognitiveModule | null;
  onCreateModule: (module: CognitiveModule) => void;
}

const SNAP_THRESHOLD = 250;

export const AgentSimulatorModal: React.FC<AgentSimulatorModalProps> = ({
  isOpen,
  onClose,
  module,
  onCreateModule
}) => {
  // UI Layout State
  const [sidebarWidth, setSidebarWidth] = useState(350);
  const [modalSize, setModalSize] = useState({ width: 1100, height: 750 });
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [activeTab, setActiveTab] = useState<'simulator' | 'forge'>('simulator');
  
  // Resizing State
  const [isResizing, setIsResizing] = useState(false); 
  const isResizingSidebarRef = useRef(false);
  const isResizingModalRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
  const rafRef = useRef<number | null>(null);

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [starters, setStarters] = useState<string[]>([]);
  const [areStartersLoading, setAreStartersLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Forge Agent State
  const [forgingAgent, setForgingAgent] = useState(false);
  const [forgedModule, setForgedModule] = useState<CognitiveModule | null>(null);

  // Monitor window resize for responsive layout
  useEffect(() => {
      const handleResize = () => setIsMobile(window.innerWidth < 768);
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize modal state when opening
  useEffect(() => {
      if (isOpen && module) {
          setMessages([]);
          setInputValue('');
          setStarters([]);
          setActiveTab('simulator');
          setForgedModule(null);
          loadStarters(module.instruction);
          // Reset Layout
          setSidebarWidth(350);
      }
  }, [isOpen, module]);

  // Auto-scroll chat
  useEffect(() => {
      if (activeTab === 'simulator') {
          messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }
  }, [messages, activeTab]);

  // Auto-resize textarea
  useEffect(() => {
      if (textareaRef.current) {
          textareaRef.current.style.height = 'auto';
          textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
      }
  }, [inputValue]);

  // --- Smooth Resizing Logic with Snap ---
  const startResizingSidebar = useCallback((e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      isResizingSidebarRef.current = true;
      dragStartRef.current = { x: e.clientX, y: e.clientY, w: sidebarWidth || 350, h: 0 };
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

          if (rafRef.current) cancelAnimationFrame(rafRef.current);

          rafRef.current = requestAnimationFrame(() => {
              const deltaX = e.clientX - dragStartRef.current.x;
              const deltaY = e.clientY - dragStartRef.current.y;

              if (isResizingSidebarRef.current) {
                  const rawWidth = dragStartRef.current.w + deltaX;
                  
                  // Logic: If < SNAP_THRESHOLD, snap to 0. Else min width 250.
                  if (rawWidth < SNAP_THRESHOLD) {
                      setSidebarWidth(0);
                  } else {
                      const newWidth = Math.max(SNAP_THRESHOLD, Math.min(600, rawWidth));
                      setSidebarWidth(newWidth);
                  }
              }

              if (isResizingModalRef.current) {
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

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
          if (rafRef.current) cancelAnimationFrame(rafRef.current);
      };
  }, []);

  // --- Chat Logic ---
  const loadStarters = async (dna: string) => {
      setAreStartersLoading(true);
      try {
          const questions = await generateAgentStarters(dna);
          setStarters(questions);
      } catch (e) {
          console.error("Failed to load starters", e);
      } finally {
          setAreStartersLoading(false);
      }
  };

  const handleResetChat = () => {
      setMessages([]);
      setInputValue('');
  };

  const handleSendMessage = async (text: string) => {
      if (!text.trim() || !module) return;
      
      const userMsg: ChatMessage = { role: 'user', text: text };
      setMessages(prev => [...prev, userMsg]);
      setInputValue('');
      setIsTyping(true);

      if (textareaRef.current) textareaRef.current.style.height = 'auto';

      try {
          const responseText = await chatWithAgent(module.instruction, [...messages, userMsg], text, []);
          const modelMsg: ChatMessage = { role: 'model', text: responseText };
          setMessages(prev => [...prev, modelMsg]);
      } catch (e) {
          setMessages(prev => [...prev, { role: 'model', text: "Error: Agent unreachable." }]);
      } finally {
          setIsTyping(false);
      }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (!isTyping) {
              handleSendMessage(inputValue);
          }
      }
  };

  const handleStopGeneration = () => {
      setIsTyping(false);
  };

  // --- File Upload Logic (Enhanced for PDF) ---
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0 && module) {
          const files: File[] = Array.from(e.target.files);
          
          const attachments: { mimeType: string; data: string }[] = [];
          let textContext = "";
          let displayMessage = "";

          for (const file of files) {
              // Robust MIME detection
              let mimeType = file.type;
              // Force detection if missing or generic
              if (!mimeType || mimeType === 'application/octet-stream') {
                   const ext = file.name.split('.').pop()?.toLowerCase();
                   if (ext === 'pdf') mimeType = 'application/pdf';
                   else if (ext === 'png') mimeType = 'image/png';
                   else if (ext === 'jpg' || ext === 'jpeg') mimeType = 'image/jpeg';
                   else if (ext === 'webp') mimeType = 'image/webp';
              }

              // Binary Path (Images & PDF)
              if (mimeType === 'application/pdf' || mimeType.startsWith('image/')) {
                  try {
                      const b64 = await fileToBase64(file);
                      attachments.push({ mimeType: mimeType, data: b64 });
                      displayMessage += `[FILE ATTACHED: ${file.name} (${mimeType})]\n`;
                  } catch (err) {
                       console.error(`Failed to read binary file ${file.name}`, err);
                       displayMessage += `[ERROR: Could not read ${file.name}]\n`;
                  }
              } 
              // Text Path (Everything else)
              else {
                  try {
                      const text = await fileToText(file);
                      textContext += `\n--- FILE CONTENT: ${file.name} ---\n${text}\n-------------------\n`;
                      displayMessage += `[TEXT FILE LOADED: ${file.name}]\n`;
                  } catch (err) {
                      console.error(`Failed to read text file ${file.name}`, err);
                      displayMessage += `[ERROR: Could not read ${file.name}]\n`;
                  }
              }
          }

          if (displayMessage) {
              const uiText = `${displayMessage}\n${textContext ? "[Text content injected into context]" : ""}`;
              
              const contextMsg: ChatMessage = { 
                  role: 'user', 
                  text: uiText
              };
              setMessages(prev => [...prev, contextMsg]);
              
              setIsTyping(true);
              try {
                   const finalPrompt = `[SYSTEM INJECTION: USER UPLOADED FILES]\n${textContext}\n\nINSTRUCTION: Analyze the attached files and/or text content provided above. Acknowledge receipt and summarize findings.`;
                   const responseText = await chatWithAgent(module.instruction, [...messages, contextMsg], finalPrompt, attachments);
                   setMessages(prev => [...prev, { role: 'model', text: responseText }]);
              } catch (err) {
                   setMessages(prev => [...prev, { role: 'model', text: "Error analyzing files." }]);
              } finally {
                  setIsTyping(false);
              }
          }
      }
  };

  // --- Export Logic ---
  const exportChat = (format: 'txt' | 'md' | 'pdf') => {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `simulation-${module?.name.replace(/\s+/g, '_')}-${timestamp}.${format}`;
      let content = "";

      if (format === 'txt') {
          content = messages.map(m => `[${m.role.toUpperCase()}]: ${m.text}`).join('\n\n');
      } else if (format === 'md') {
          content = messages.map(m => `**${m.role.toUpperCase()}**\n\n${m.text}`).join('\n\n---\n\n');
      }

      if (format === 'pdf') {
          const doc = new jsPDF();
          let y = 10;
          doc.setFontSize(10);
          messages.forEach(m => {
              const role = m.role.toUpperCase();
              const text = doc.splitTextToSize(`${role}: ${m.text}`, 180);
              if (y + text.length * 5 > 280) { doc.addPage(); y = 10; }
              doc.text(text, 10, y);
              y += text.length * 5 + 5;
          });
          doc.save(filename);
      } else {
          const blob = new Blob([content], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
      }
  };

  const copyChat = async () => {
      const content = messages.map(m => `[${m.role.toUpperCase()}]: ${m.text}`).join('\n\n');
      try {
          await navigator.clipboard.writeText(content);
      } catch (err) {
          console.error("Copy failed", err);
      }
  };

  const copyMessage = async (text: string) => {
       try {
          await navigator.clipboard.writeText(text);
      } catch (err) {
          console.error("Copy failed", err);
      }
  };

  // --- Forge Logic ---
  const handleForgeAgent = async (sourceText?: string) => {
      setForgingAgent(true);
      try {
          const contextToAnalyze = sourceText || messages.map(m => `${m.role}: ${m.text}`).join('\n');
          if (!contextToAnalyze.trim()) return;

          const newModule = await generateModuleFromHistory(contextToAnalyze);
          setForgedModule(newModule);
          setActiveTab('forge');
      } catch (e) {
          console.error("Forge failed", e);
      } finally {
          setForgingAgent(false);
      }
  };

  const saveForgedModule = () => {
      if (forgedModule) {
          onCreateModule(forgedModule);
          setActiveTab('simulator'); 
          alert(`Agent "${forgedModule.name}" added to your registry!`);
          setForgedModule(null);
      }
  };

  if (!isOpen || !module) return null;

  return (
    <div className="fixed inset-0 bg-gray-900/90 backdrop-blur-sm z-[60] flex items-center justify-center p-0 md:p-4 animate-in fade-in duration-200">
      <div 
        className={`bg-gray-800 border border-gray-700 shadow-2xl flex flex-col overflow-hidden relative
            ${isMobile ? 'w-full h-full' : 'md:rounded-2xl'}
            ${isResizing ? 'transition-none' : 'transition-all duration-200 ease-out'}
        `}
        style={{ 
            width: isMobile ? '100%' : `${modalSize.width}px`,
            height: isMobile ? '100%' : `${modalSize.height}px`
        }}
      >
        
        {/* Top Bar: Tabs & Close */}
        <div className="h-10 bg-gray-950 border-b border-gray-700 flex items-center justify-between px-2 select-none flex-shrink-0">
             <div className="flex gap-1 h-full pt-1">
                 <button 
                    onClick={() => setActiveTab('simulator')}
                    className={`px-4 h-full rounded-t-md text-xs font-bold flex items-center gap-2 transition-colors
                        ${activeTab === 'simulator' ? 'bg-gray-800 text-green-400 border-t border-x border-gray-700' : 'text-gray-500 hover:text-gray-300'}
                    `}
                 >
                     <TerminalIcon className="w-3.5 h-3.5" />
                     SIMULATOR
                 </button>
                 {forgedModule && (
                     <button 
                        onClick={() => setActiveTab('forge')}
                        className={`px-4 h-full rounded-t-md text-xs font-bold flex items-center gap-2 transition-colors animate-in slide-in-from-bottom-1
                            ${activeTab === 'forge' ? 'bg-gray-800 text-cyan-400 border-t border-x border-gray-700' : 'text-gray-500 hover:text-gray-300'}
                        `}
                     >
                         <DnaIcon className="w-3.5 h-3.5" />
                         FORGED AGENT
                     </button>
                 )}
             </div>
             
             <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono text-gray-600 hidden md:inline-block">
                        {module.name} :: gemini-3-pro
                </span>
                <button onClick={onClose} className="text-gray-500 hover:text-white p-1 hover:bg-red-900/30 rounded transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
             </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 flex flex-col relative overflow-hidden bg-gray-900 min-h-0">
            
            {/* TAB 1: SIMULATOR */}
            {activeTab === 'simulator' && (
                <div className="flex-1 flex flex-col md:flex-row h-full overflow-hidden">
                     
                     {/* Sidebar: DNA */}
                     <div 
                        className="flex-shrink-0 h-full transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] relative z-10"
                        style={{ width: isMobile ? '0px' : (sidebarWidth === 0 ? '48px' : `${sidebarWidth}px`) }}
                     >
                        {sidebarWidth === 0 && !isMobile ? (
                            // Gutter Button (Collapsed State)
                            <div className="h-full w-full">
                                <button
                                    onClick={() => setSidebarWidth(350)}
                                    className="
                                        h-full w-full
                                        bg-gray-900/95 backdrop-blur-sm
                                        border-r border-gray-800
                                        rounded-r-2xl
                                        flex flex-col items-center justify-between py-8
                                        hover:bg-gray-800 hover:border-green-500/60
                                        transition-all duration-300
                                        group
                                        shadow-[5px_0_15px_rgba(0,0,0,0.3)]
                                        cursor-pointer
                                        relative z-10
                                    "
                                >
                                    {/* Icon */}
                                    <div className="p-2 rounded-lg bg-gray-800/50 text-green-500 group-hover:bg-green-500 group-hover:text-white transition-all mb-4 shadow-[0_0_10px_rgba(34,197,94,0.2)]">
                                        <ChevronDownIcon className="w-5 h-5 -rotate-90" />
                                    </div>

                                    {/* Vertical Text */}
                                    <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                                        <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-green-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                                            <span>System DNA</span>
                                            <span className="w-px h-8 bg-gray-700 group-hover:bg-green-500/50 transition-colors"></span>
                                        </div>
                                    </div>

                                    {/* Dot */}
                                    <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-green-400 shadow-[0_0_5px_rgba(34,197,94,0.8)] transition-colors mt-4"></div>
                                </button>
                            </div>
                        ) : (
                            // Full Content (Expanded State)
                            <div className="bg-gray-900/50 flex flex-col h-full border-r border-gray-700 relative w-full overflow-hidden">
                                <div className="p-2 bg-gray-900/80 border-b border-gray-700/50 text-xs font-bold text-gray-500 uppercase tracking-widest sticky top-0 flex justify-between items-center flex-shrink-0">
                                    <span className="flex items-center gap-2"><DnaIcon className="w-3 h-3"/> System DNA</span>
                                </div>
                                <div className="flex-1 overflow-auto p-4 custom-scrollbar">
                                    <div className="prose prose-invert prose-xs max-w-none font-mono whitespace-pre-wrap text-gray-500 selection:bg-green-900 selection:text-green-100 h-full text-[10px] leading-relaxed">
                                        {module.instruction}
                                    </div>
                                </div>
                            </div>
                        )}
                     </div>
                     
                    {/* Resizer */}
                    {!isMobile && sidebarWidth > 0 && <Resizer onMouseDown={startResizingSidebar} isVisible={true} />}

                    {/* Chat Area */}
                    <div className="flex-1 flex flex-col min-w-0 bg-gray-950 relative h-full">
                        
                        {/* Chat Toolbar */}
                        <div className="h-10 border-b border-gray-800 flex items-center justify-between px-3 bg-gray-900/30 flex-shrink-0">
                            {/* LEFT: Reset Session */}
                            <div className="flex items-center gap-2">
                                    <Tooltip content={{ function: "Reset Session", mechanism: "Clears active memory.", expectation: "Wipes chat history.", example: "Start new test." }} position="right">
                                    <button onClick={handleResetChat} className="p-2 hover:bg-red-900/20 rounded text-gray-500 hover:text-red-400 transition-colors">
                                        <RefreshCcwIcon className="w-4 h-4" />
                                    </button>
                                    </Tooltip>
                            </div>

                            {/* RIGHT: Exports and Tools */}
                            <div className="flex items-center gap-2">
                                <Tooltip content={{ function: "Copy Chat", mechanism: "Clipboard access.", expectation: "Raw text copy.", example: "Paste in doc." }} position="bottom">
                                    <button onClick={() => copyChat()} className="p-1.5 hover:bg-gray-800 rounded text-gray-500 hover:text-white transition-colors"><CopyIcon className="w-4 h-4" /></button>
                                </Tooltip>
                                
                                <div className="h-4 w-px bg-gray-800 mx-1"></div>
                                
                                <Tooltip content={{ function: "Export TXT", mechanism: "Generates .txt", expectation: "Download log.", example: "Archiving." }} position="bottom">
                                    <button onClick={() => exportChat('txt')} className="p-1.5 hover:bg-gray-800 rounded text-gray-500 hover:text-white text-[10px] font-bold">TXT</button>
                                </Tooltip>
                                    <Tooltip content={{ function: "Export MD", mechanism: "Generates .md", expectation: "Markdown log.", example: "Format preserved." }} position="bottom">
                                    <button onClick={() => exportChat('md')} className="p-1.5 hover:bg-gray-800 rounded text-gray-500 hover:text-white text-[10px] font-bold">MD</button>
                                </Tooltip>
                                    <Tooltip content={{ function: "Export PDF", mechanism: "Generates .pdf", expectation: "PDF Document.", example: "Sharing." }} position="bottom">
                                    <button onClick={() => exportChat('pdf')} className="p-1.5 hover:bg-gray-800 rounded text-gray-500 hover:text-white text-[10px] font-bold">PDF</button>
                                </Tooltip>

                                <div className="h-4 w-px bg-gray-800 mx-1"></div>

                                <Tooltip 
                                    content={{ 
                                        function: "Forge Agent from Context", 
                                        mechanism: "Analyzes the current conversation history using the Kernel.", 
                                        expectation: "Extracts the Persona, Logic, and Tone into a new reusable Module.", 
                                        example: "Use this after a successful roleplay." 
                                    }} 
                                    position="left"
                                >
                                    <button 
                                        onClick={() => handleForgeAgent()} 
                                        disabled={forgingAgent || messages.length === 0}
                                        className="flex items-center gap-1 px-2 py-1 bg-cyan-900/30 text-cyan-400 hover:bg-cyan-900/50 hover:text-cyan-200 rounded text-[10px] font-bold transition-colors border border-cyan-900 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {forgingAgent ? <span className="animate-spin">⟳</span> : <DnaIcon className="w-3 h-3" />}
                                        FORGE
                                    </button>
                                </Tooltip>
                            </div>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar scroll-smooth relative">
                            {messages.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-full text-center p-8 opacity-60">
                                    <div className="w-12 h-12 rounded-full flex items-center justify-center border border-gray-800 mb-4 text-gray-600">
                                        <TerminalIcon className="w-6 h-6" />
                                    </div>
                                    <p className="text-sm text-gray-500 mb-6 max-w-sm">
                                        Simulate interactions to test logic. Upload files for context. Forge new agents from results.
                                    </p>
                                    {!areStartersLoading && starters.map((s, i) => (
                                        <button key={i} onClick={() => handleSendMessage(s)} className="text-xs text-green-600 hover:text-green-400 hover:underline mb-1 block font-mono">
                                            $ {s}
                                        </button>
                                    ))}
                                </div>
                            )}

                            {messages.map((msg, idx) => (
                                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} group`}>
                                    <div className={`max-w-[90%] rounded-xl p-3 text-sm leading-relaxed shadow-lg relative
                                        ${msg.role === 'user' ? 'bg-cyan-900/20 border border-cyan-800/50 text-cyan-100 rounded-tr-none' : 'bg-gray-900 border border-gray-800 text-gray-300 rounded-tl-none'}
                                    `}>
                                        <pre className="whitespace-pre-wrap font-sans">{msg.text}</pre>
                                        <div className={`absolute -bottom-6 ${msg.role === 'user' ? 'right-0' : 'left-0'} flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200`}>
                                            <button onClick={() => copyMessage(msg.text)} className="p-1 bg-gray-800 border border-gray-700 rounded text-gray-400 hover:text-white" title="Copy"><CopyIcon className="w-3 h-3" /></button>
                                            <button onClick={() => handleForgeAgent(msg.text)} className="p-1 bg-gray-800 border border-gray-700 rounded text-cyan-400 hover:text-cyan-200" title="Forge Agent from this"><DnaIcon className="w-3 h-3" /></button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                            
                            {isTyping && <div className="text-xs text-gray-500 animate-pulse font-mono pl-2">Processing stream...</div>}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Professional Input Area */}
                        <div className="bg-gray-950 border-t border-gray-800 flex-shrink-0">
                            {/* Attachment Bar (Top Right) */}
                            <div className="px-4 py-1 flex justify-end bg-gray-950 border-b border-gray-800/30">
                                <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileUpload} multiple />
                                <button 
                                    onClick={() => fileInputRef.current?.click()}
                                    className="flex items-center gap-1.5 text-[10px] font-bold text-gray-500 hover:text-cyan-400 transition-colors py-1"
                                    title="Upload PDF/Images/Text"
                                >
                                    <PaperclipIcon className="w-3 h-3" />
                                    <span>ATTACH CONTEXT</span>
                                </button>
                            </div>

                            {/* Main Input */}
                            <div className="px-4 pb-4 pt-2">
                                <div className="relative bg-gray-900 border border-gray-800 rounded-xl shadow-inner focus-within:border-cyan-900 focus-within:ring-1 focus-within:ring-cyan-900/50 transition-all">
                                    <textarea 
                                        ref={textareaRef}
                                        value={inputValue} 
                                        onChange={e => setInputValue(e.target.value)}
                                        onKeyDown={handleKeyDown}
                                        placeholder="Type a message... (Shift+Enter for new line)"
                                        className="w-full bg-transparent text-gray-200 p-3 pr-24 max-h-60 min-h-[50px] resize-none focus:outline-none custom-scrollbar text-sm leading-relaxed"
                                        rows={1}
                                    />
                                    
                                    {/* Control Center (Bottom Right) */}
                                    <div className="absolute bottom-2 right-2 flex items-center gap-1.5 bg-gray-900/80 backdrop-blur-sm rounded-lg p-1 border border-gray-800/50">
                                        {/* Clear Button */}
                                        <Tooltip content={{ function: "Clear Input", mechanism: "Wipes buffer.", expectation: "Empty field.", example: "Cancel typing." }} position="top">
                                            <button 
                                                onClick={() => setInputValue('')}
                                                className="p-1.5 text-gray-500 hover:text-red-400 hover:bg-red-900/20 rounded transition-colors"
                                            >
                                                <TrashIcon className="w-4 h-4" />
                                            </button>
                                        </Tooltip>
                                        
                                        <div className="w-px h-4 bg-gray-700 mx-0.5"></div>

                                        {/* Stop/Send Toggle */}
                                        {isTyping ? (
                                            <Tooltip content={{ function: "Stop", mechanism: "Aborts stream.", expectation: "Halts generation.", example: "Stop output." }} position="top">
                                                <button 
                                                    onClick={handleStopGeneration}
                                                    className="p-1.5 text-red-500 hover:text-red-400 hover:bg-red-900/20 rounded transition-colors animate-pulse"
                                                >
                                                    <StopCircleIcon className="w-5 h-5" />
                                                </button>
                                            </Tooltip>
                                        ) : (
                                            <Tooltip content={{ function: "Send", mechanism: "Submits input.", expectation: "Agent response.", example: "Execute." }} position="top">
                                                <button 
                                                    onClick={() => handleSendMessage(inputValue)}
                                                    disabled={!inputValue.trim()}
                                                    className="p-1.5 text-cyan-500 hover:text-cyan-300 hover:bg-cyan-900/20 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                                                >
                                                    <SendIcon className="w-5 h-5" />
                                                </button>
                                            </Tooltip>
                                        )}
                                    </div>
                                </div>
                                <div className="text-[10px] text-gray-600 text-right mt-1 pr-1">
                                    Gemini 3.0 Pro Preview • MMS Kernel Integrated
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* TAB 2: FORGE AGENT EDITOR */}
            {activeTab === 'forge' && forgedModule && (
                <div className="flex-1 bg-gray-900 p-6 overflow-y-auto custom-scrollbar animate-in slide-in-from-right-4">
                    <div className="max-w-3xl mx-auto space-y-6">
                        
                        {/* Info Banner */}
                        <div className="bg-cyan-900/20 border border-cyan-800/50 p-4 rounded-lg flex gap-3 items-start animate-in fade-in slide-in-from-top-2">
                             <DnaIcon className="w-6 h-6 text-cyan-400 flex-shrink-0 mt-0.5" />
                             <div>
                                <h4 className="text-sm font-bold text-cyan-300 mb-1">Agent Forged from Context</h4>
                                <p className="text-xs text-gray-400 leading-relaxed">
                                    The Kernel has analyzed your conversation history and extracted the implicit Persona, Logic, and System Prompt. 
                                    You can now refine this new cognitive module and save it to your registry.
                                </p>
                             </div>
                        </div>

                        <div className="flex items-center justify-between border-b border-gray-800 pb-4">
                            <h3 className="text-xl font-bold text-cyan-400 flex items-center gap-2">
                                Forged Cognitive Module
                            </h3>
                            <div className="flex gap-3">
                                <button 
                                    onClick={() => setForgedModule(null)} 
                                    className="px-4 py-2 text-gray-400 hover:text-white text-xs uppercase font-bold tracking-wider"
                                >
                                    Discard
                                </button>
                                <button 
                                    onClick={saveForgedModule}
                                    className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded shadow-lg shadow-cyan-900/20 text-sm font-bold transition-all transform hover:scale-105"
                                >
                                    Save to Registry
                                </button>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Agent Name</label>
                                <input 
                                    value={forgedModule.name} 
                                    onChange={e => setForgedModule({...forgedModule, name: e.target.value})}
                                    className="w-full bg-gray-950 border border-gray-800 p-3 rounded text-gray-100 focus:border-cyan-500 focus:outline-none"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Function</label>
                                <input 
                                    value={forgedModule.tooltip.function} 
                                    onChange={e => setForgedModule({...forgedModule, tooltip: {...forgedModule.tooltip, function: e.target.value}})}
                                    className="w-full bg-gray-950 border border-gray-800 p-3 rounded text-gray-400 focus:border-cyan-500 focus:outline-none"
                                />
                            </div>
                            <div className="col-span-full">
                                <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Description</label>
                                <input 
                                    value={forgedModule.description} 
                                    onChange={e => setForgedModule({...forgedModule, description: e.target.value})}
                                    className="w-full bg-gray-950 border border-gray-800 p-3 rounded text-gray-300 focus:border-cyan-500 focus:outline-none"
                                />
                            </div>
                            <div className="col-span-full h-96">
                                <label className="block text-xs font-bold text-cyan-600 uppercase mb-1">System DNA (Instruction)</label>
                                <textarea 
                                    value={forgedModule.instruction} 
                                    onChange={e => setForgedModule({...forgedModule, instruction: e.target.value})}
                                    className="w-full h-full bg-gray-950 border border-gray-800 p-4 rounded text-gray-300 font-mono text-xs leading-relaxed focus:border-cyan-500 focus:outline-none resize-none"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>

        {/* Corner Resizer (Desktop Only) */}
        {!isMobile && (
            <div 
                className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 flex items-end justify-end p-1 group"
                onMouseDown={startResizingModal}
            >
                <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-green-400 transition-colors"></div>
            </div>
        )}
      </div>
    </div>
  );
};