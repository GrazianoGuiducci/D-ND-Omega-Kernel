import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { DOCS_DATA, DocSectionId, DocBlock } from '../services/docsData';
import { SparklesIcon } from './icons/SparklesIcon';
import { SquaresPlusIcon } from './icons/SquaresPlusIcon';
import { SwatchIcon } from './icons/SwatchIcon';
import { BookOpenIcon } from './icons/BookOpenIcon';
import { ClockIcon } from './icons/ClockIcon';
import { CpuChipIcon } from './icons/CpuChipIcon';
import { Resizer } from './Resizer';

interface DocumentationModalProps {
    isOpen: boolean;
    onClose: () => void;
}

const SNAP_THRESHOLD = 250;

const DocumentationModal: React.FC<DocumentationModalProps> = ({ isOpen, onClose }) => {
    // Modal & Resizing State
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
    const [modalSize, setModalSize] = useState({ width: 1100, height: 750 });
    const [sidebarWidth, setSidebarWidth] = useState(280);
    const [isResizing, setIsResizing] = useState(false);

    const isResizingSidebarRef = useRef(false);
    const isResizingModalRef = useRef(false);
    const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
    const rafRef = useRef<number | null>(null);

    // Content State
    const [activeTabId, setActiveTabId] = useState<DocSectionId | 'library'>('identity');
    const [libraryFiles, setLibraryFiles] = useState<string[]>([]);
    const [activeDocContent, setActiveDocContent] = useState<string | null>(null);
    const [isLoadingLibrary, setIsLoadingLibrary] = useState(false);

    // --- RESIZING LOGIC ---
    useEffect(() => {
        const handleResize = () => setIsMobile(window.innerWidth < 768);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

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
            if (rafRef.current) cancelAnimationFrame(rafRef.current);

            rafRef.current = requestAnimationFrame(() => {
                const deltaX = e.clientX - dragStartRef.current.x;
                const deltaY = e.clientY - dragStartRef.current.y;

                if (isResizingSidebarRef.current) {
                    const rawWidth = dragStartRef.current.w + deltaX;
                    if (rawWidth < SNAP_THRESHOLD) {
                        setSidebarWidth(0);
                    } else {
                        const newWidth = Math.max(SNAP_THRESHOLD, Math.min(500, rawWidth));
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

    // --- CONTENT LOGIC ---
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        };
        if (isOpen) window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [isOpen, onClose]);

    useEffect(() => {
        if (isOpen) {
            fetch('/api/docs')
                .then(res => res.json())
                .then(data => setLibraryFiles(data.files || []))
                .catch(err => console.error("Failed to fetch docs library", err));
        }
    }, [isOpen]);

    const handleLoadDoc = async (filename: string) => {
        setIsLoadingLibrary(true);
        try {
            const res = await fetch(`/api/docs/${filename}`);
            const data = await res.json();
            setActiveDocContent(data.content);
        } catch (e) {
            setActiveDocContent("Error loading document.");
        } finally {
            setIsLoadingLibrary(false);
        }
    };

    const activeSection = DOCS_DATA.find(d => d.id === activeTabId);

    const renderIcon = (type: string, className: string) => {
        switch (type) {
            case 'sparkle': return <SparklesIcon className={className} />;
            case 'square': return <SquaresPlusIcon className={className} />;
            case 'swatch': return <SwatchIcon className={className} />;
            case 'book': return <BookOpenIcon className={className} />;
            case 'clock': return <ClockIcon className={className} />;
            case 'cpu': return <CpuChipIcon className={className} />;
            case 'library': return <BookOpenIcon className={className} />;
            default: return <div className={`w-4 h-4 rounded-full border-2 border-current flex items-center justify-center text-[8px] font-bold ${className}`}>!</div>;
        }
    };

    const renderBlock = (block: DocBlock, idx: number) => {
        switch (block.type) {
            case 'text':
                return <p key={idx} className="text-slate-300 text-sm leading-relaxed mb-4">{block.value}</p>;
            case 'alert':
                return (
                    <div key={idx} className="bg-slate-900/50 border-l-2 border-neon-cyan p-4 mb-4 rounded-r">
                        <strong className="block text-neon-cyan text-xs uppercase tracking-wider mb-1">{block.title}</strong>
                        <p className="text-slate-400 text-xs">{block.value}</p>
                    </div>
                );
            case 'code':
                return (
                    <div key={idx} className="bg-black/50 p-4 rounded border border-white/10 mb-4 font-mono text-xs">
                        {block.title && <div className="text-slate-500 mb-2 border-b border-slate-800 pb-1">{block.title}</div>}
                        <code className="text-green-400 block whitespace-pre-wrap">{block.value}</code>
                        {block.meta && <div className="text-slate-600 mt-2 text-[10px] italic">// {block.meta}</div>}
                    </div>
                );
            case 'list':
                return (
                    <div key={idx} className="mb-4">
                        {block.title && <h4 className="text-white font-bold text-sm mb-2">{block.title}</h4>}
                        <ul className="space-y-2">
                            {(block.value as string[]).map((item, i) => (
                                <li key={i} className="flex items-start gap-2 text-slate-400 text-xs">
                                    <span className="text-neon-cyan mt-1">›</span>
                                    {item}
                                </li>
                            ))}
                        </ul>
                    </div>
                );
            case 'step':
                return (
                    <div key={idx} className="flex gap-4 mb-4 group">
                        <div className="flex flex-col items-center">
                            <div className="w-6 h-6 rounded-full bg-slate-800 border border-slate-600 flex items-center justify-center text-[10px] font-bold text-white group-hover:border-neon-cyan group-hover:text-neon-cyan transition-colors">
                                {idx + 1}
                            </div>
                            <div className="w-0.5 h-full bg-slate-800 my-1 group-last:hidden"></div>
                        </div>
                        <div>
                            <h5 className="text-slate-200 font-bold text-sm group-hover:text-neon-cyan transition-colors">{block.title}</h5>
                            <p className="text-slate-400 text-xs leading-relaxed">{block.value}</p>
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    if (!isOpen) return null;

    return createPortal(
        <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-[100] flex items-center justify-center p-0 md:p-6">
            <div
                className={`bg-[#050b14] shadow-2xl border border-white/10 flex flex-col overflow-hidden relative
                    ${isMobile ? 'w-full h-full rounded-none' : 'md:rounded-2xl'}
                    ${isResizing ? 'transition-none' : 'transition-all duration-200 ease-out'}`}
                style={{
                    width: isMobile ? '100%' : `${modalSize.width}px`,
                    height: isMobile ? '100%' : `${modalSize.height}px`,
                }}
            >
                {/* Header */}
                <div className="h-12 bg-white/5 border-b border-white/10 flex items-center justify-between px-4 shrink-0">
                    <div className="flex items-center gap-2">
                        <BookOpenIcon className="w-4 h-4 text-neon-cyan" />
                        <span className="text-sm font-bold text-white uppercase tracking-widest">System Documentation</span>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-white p-1 hover:bg-white/10 rounded transition-colors">
                        ✕
                    </button>
                </div>

                {/* Content */}
                <div className="flex flex-1 min-h-0">
                    {/* Sidebar */}
                    <div
                        className="flex-shrink-0 h-full transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] relative z-10 bg-black/40"
                        style={{ width: isMobile ? '100%' : (sidebarWidth === 0 ? '48px' : `${sidebarWidth}px`) }}
                    >
                        {sidebarWidth === 0 && !isMobile ? (
                            <div className="w-full h-full">
                                <button
                                    onClick={() => setSidebarWidth(280)}
                                    className="h-full w-full bg-black/40 border-r border-white/10 flex flex-col items-center justify-between py-8 hover:bg-white/5 hover:border-neon-cyan/50 transition-all duration-300 group"
                                >
                                    <div className="p-2 rounded-lg bg-white/5 text-neon-cyan group-hover:bg-neon-cyan group-hover:text-black transition-all mb-4">
                                        <span className="block w-5 h-5 rotate-[-90deg]">›</span>
                                    </div>
                                    <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                                        <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-slate-500 group-hover:text-neon-cyan transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                                            <span>Sections</span>
                                            <span className="w-px h-8 bg-white/10 group-hover:bg-neon-cyan/50 transition-colors"></span>
                                        </div>
                                    </div>
                                </button>
                            </div>
                        ) : (
                            <div className="h-full border-r border-white/10 flex flex-col w-full overflow-hidden">
                                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                                    {DOCS_DATA.map((section) => (
                                        <button
                                            key={section.id}
                                            onClick={() => { setActiveTabId(section.id); setActiveDocContent(null); }}
                                            className={`
                                                w-full flex items-center gap-3 px-4 py-3 rounded-lg text-xs font-bold uppercase tracking-wide transition-all
                                                ${activeTabId === section.id
                                                    ? 'bg-white/5 text-white border border-white/10 shadow-lg'
                                                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.02] border border-transparent'}
                                            `}
                                        >
                                            {renderIcon(section.iconType, `w-4 h-4 ${activeTabId === section.id ? section.color : 'text-slate-600'}`)}
                                            {section.label}
                                        </button>
                                    ))}

                                    <div className="my-2 border-t border-white/5"></div>
                                    <button
                                        onClick={() => setActiveTabId('library')}
                                        className={`
                                            w-full flex items-center gap-3 px-4 py-3 rounded-lg text-xs font-bold uppercase tracking-wide transition-all
                                            ${activeTabId === 'library'
                                                ? 'bg-white/5 text-white border border-white/10 shadow-lg'
                                                : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.02] border border-transparent'}
                                        `}
                                    >
                                        {renderIcon('library', `w-4 h-4 ${activeTabId === 'library' ? 'text-yellow-400' : 'text-slate-600'}`)}
                                        Library (External)
                                    </button>
                                </div>
                                <div className="p-4 border-t border-white/10 bg-black/20">
                                    <div className="text-[10px] font-mono text-slate-600">
                                        SYS_STATUS: <span className="text-green-500">ONLINE</span><br />
                                        DOCS_VER: 2024.12.01
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Gutter Resizer */}
                    {!isMobile && sidebarWidth > 0 && (
                        <Resizer onMouseDown={startResizingSidebar} isVisible={true} />
                    )}

                    {/* Main Content */}
                    <div className="flex-1 flex flex-col bg-black/10 min-w-0 relative">
                        {/* Background Grid */}
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[length:40px_40px] pointer-events-none"></div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar p-8 md:p-12 relative z-10">
                            {activeTabId === 'library' ? (
                                <div>
                                    <div className="mb-8 pb-6 border-b border-white/10">
                                        <h1 className="text-3xl font-bold mb-2 text-yellow-400">External Library</h1>
                                        <p className="text-lg text-slate-400 font-light">Documentazione caricata dal Backend.</p>
                                    </div>

                                    <div className="grid grid-cols-1 gap-4 mb-8">
                                        {libraryFiles.map(file => (
                                            <button
                                                key={file}
                                                onClick={() => handleLoadDoc(file)}
                                                className="text-left p-4 bg-white/5 hover:bg-white/10 border border-white/5 rounded transition-colors flex justify-between items-center group"
                                            >
                                                <span className="text-sm font-mono text-slate-300 group-hover:text-white">{file}</span>
                                                <span className="text-xs text-slate-500">OPEN &rarr;</span>
                                            </button>
                                        ))}
                                    </div>

                                    {activeDocContent && (
                                        <div className="animate-slide-up-fade bg-black/40 p-6 rounded border border-white/10">
                                            <pre className="whitespace-pre-wrap font-mono text-xs text-slate-300 leading-relaxed">
                                                {activeDocContent}
                                            </pre>
                                        </div>
                                    )}
                                </div>
                            ) : activeSection ? (
                                <>
                                    <div className="mb-8 pb-6 border-b border-white/10">
                                        <h1 className={`text-3xl font-bold mb-2 ${activeSection.color}`}>{activeSection.title}</h1>
                                        <p className="text-lg text-slate-400 font-light">{activeSection.subtitle}</p>
                                    </div>
                                    <div className="space-y-6 animate-slide-up-fade">
                                        {activeSection.content.map((block, idx) => renderBlock(block, idx))}
                                    </div>
                                </>
                            ) : null}
                        </div>
                    </div>
                </div>

                {/* Corner resizer */}
                {!isMobile && (
                    <div
                        className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 flex items-end justify-end p-1 group"
                        onMouseDown={startResizingModal}
                    >
                        <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-neon-cyan transition-colors" />
                    </div>
                )}
            </div>
        </div>,
        document.body
    );
};

export default DocumentationModal;
