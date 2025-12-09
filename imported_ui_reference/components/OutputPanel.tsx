import React, { useState } from 'react';
import type { ApiState, Risultante, WorkflowStep, Source } from '../types';
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import { GlobeIcon } from './icons/GlobeIcon';
import { CpuIcon } from './icons/CpuIcon';
import { CopyIcon } from './icons/CopyIcon';
import { DownloadIcon } from './icons/DownloadIcon';
import { translations, Language } from '../translations';
import { jsPDF } from "jspdf";

interface OutputPanelProps {
  state: ApiState;
  error: string | null;
  risultante: Risultante | null;
  processingSteps: WorkflowStep[];
  prompt: string | null;
  sources?: Source[];
  language: Language;
}

const LoadingState: React.FC<{ t: any }> = ({ t }) => (
  <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-2">
    <div className="relative w-12 h-12 flex items-center justify-center">
        <svg className="animate-spin h-10 w-10 text-cyan-400 absolute" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
    </div>
    <p className="text-lg font-semibold mt-2">{t.loading}</p>
    <p className="text-sm text-gray-500">{t.loadingSub}</p>
  </div>
);

const IdleState: React.FC<{ t: any }> = ({ t }) => (
  <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-2">
    <div className="w-12 h-12 rounded-full border border-gray-800 flex items-center justify-center">
        <CpuIcon className="w-6 h-6 opacity-20" />
    </div>
    <p className="text-sm max-w-xs text-center">{t.idle}</p>
  </div>
);

const ErrorState: React.FC<{ error: string, t: any }> = ({ error, t }) => (
  <div className="p-4 bg-red-900/30 border border-red-700 rounded-md text-red-300 h-full flex flex-col items-center justify-center text-center">
    <h3 className="font-bold mb-2 text-lg">{t.error}</h3>
    <p className="text-sm opacity-90">{error}</p>
  </div>
);

const SuccessState: React.FC<{ steps: WorkflowStep[]; risultante: Risultante; prompt: string | null; sources?: Source[]; t: any }> = ({ steps, risultante, prompt, sources, t }) => {
    const [isLogOpen, setIsLogOpen] = useState(true);
    const [isSourcesOpen, setIsSourcesOpen] = useState(!!(sources && sources.length > 0));
    const [isPromptOpen, setIsPromptOpen] = useState(false);
    const [copyStatus, setCopyStatus] = useState<'idle' | 'copied'>('idle');

    const formatContent = (format: 'txt' | 'md') => {
        const date = new Date().toLocaleString();
        const header = format === 'md' ? `# ACOS Resultant - ${date}\n\n` : `ACOS Resultant - ${date}\n\n`;
        
        const l1 = format === 'md' 
            ? `## ${t.l1}\n${risultante.livello1}\n\n` 
            : `[${t.l1}]\n${risultante.livello1}\n\n`;
            
        const l2 = format === 'md'
            ? `## ${t.l2}\n${risultante.livello2}\n\n`
            : `[${t.l2}]\n${risultante.livello2}\n\n`;

        const l3 = format === 'md'
            ? `## ${t.l3}\n- **${t.diagnosis}**: ${risultante.livello3.diagnosi}\n- **${t.trajectory}**: ${risultante.livello3.traiettoria}\n- **${t.synthesis}**: ${risultante.livello3.sintesi}\n- **${t.imprint}**: ${risultante.livello3.impronta}\n`
            : `[${t.l3}]\n${t.diagnosis}: ${risultante.livello3.diagnosi}\n${t.trajectory}: ${risultante.livello3.traiettoria}\n${t.synthesis}: ${risultante.livello3.sintesi}\n${t.imprint}: ${risultante.livello3.impronta}\n`;

        return `${header}${l1}${l2}${l3}`;
    };

    const handleCopy = async () => {
        const content = formatContent('txt');
        try {
            await navigator.clipboard.writeText(content);
            setCopyStatus('copied');
            setTimeout(() => setCopyStatus('idle'), 2000);
        } catch (err) {
            console.error('Failed to copy', err);
        }
    };

    const handleDownload = (format: 'txt' | 'md') => {
        const content = formatContent(format);
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `acos-result.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handlePdf = () => {
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.getWidth();
        const margin = 15;
        const maxLineWidth = pageWidth - (margin * 2);
        
        let y = 20;

        const addText = (text: string, fontSize: number, isBold: boolean = false) => {
            doc.setFontSize(fontSize);
            doc.setFont(undefined, isBold ? 'bold' : 'normal');
            
            const lines = doc.splitTextToSize(text, maxLineWidth);
            
            // Check for page break
            if (y + (lines.length * fontSize * 0.5) > doc.internal.pageSize.getHeight() - margin) {
                doc.addPage();
                y = 20;
            }

            doc.text(lines, margin, y);
            y += (lines.length * fontSize * 0.4) + 5; 
        };

        // Header
        addText(`ACOS Resultant - ${new Date().toLocaleDateString()}`, 16, true);
        y += 5;

        // Level 1
        addText(t.l1, 12, true);
        addText(risultante.livello1, 10);
        y += 5;

        // Level 2
        addText(t.l2, 12, true);
        addText(risultante.livello2, 10);
        y += 5;

        // Level 3
        addText(t.l3, 12, true);
        addText(`${t.diagnosis}:`, 10, true);
        addText(risultante.livello3.diagnosi, 10);
        
        addText(`${t.trajectory}:`, 10, true);
        addText(risultante.livello3.traiettoria, 10);

        addText(`${t.synthesis}:`, 10, true);
        addText(risultante.livello3.sintesi, 10);

        addText(`${t.imprint}:`, 10, true);
        addText(risultante.livello3.impronta, 10);

        doc.save("acos-result.pdf");
    };

    return (
    <div className="space-y-4 pb-10 relative">
      
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-end gap-2 sticky top-0 z-10 bg-gray-800/95 backdrop-blur p-2 rounded-b-lg border-b border-gray-700 shadow-md mb-4 -mt-2 -mx-2">
          <button 
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors border border-gray-600"
            title={t.copy}
          >
            <CopyIcon className="w-3.5 h-3.5" />
            {copyStatus === 'copied' ? <span className="text-green-400">{t.copied}</span> : t.copy}
          </button>
          
          <div className="w-px h-4 bg-gray-600 mx-1"></div>

          <button 
            onClick={() => handleDownload('txt')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors border border-gray-600"
            title={t.saveTxt}
          >
            <DownloadIcon className="w-3.5 h-3.5" />
            TXT
          </button>
           <button 
            onClick={() => handleDownload('md')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors border border-gray-600"
            title={t.saveMd}
          >
            <DownloadIcon className="w-3.5 h-3.5" />
            MD
          </button>
           <button 
            onClick={handlePdf}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors border border-gray-600"
            title={t.savePdf}
          >
            <DownloadIcon className="w-3.5 h-3.5" />
            PDF
          </button>
      </div>

      {/* Processing Log Collapsible */}
      <div className="border border-gray-700/50 rounded-md overflow-hidden bg-gray-900/30 shadow-sm">
        <button
            onClick={() => setIsLogOpen(!isLogOpen)}
            className="flex items-center justify-between w-full p-3 text-left bg-gray-800/50 hover:bg-gray-800 transition-colors group"
        >
            <div className="flex items-center gap-2">
                <CpuIcon className="w-4 h-4 text-cyan-400 group-hover:text-cyan-300" />
                <h3 className="text-sm font-semibold text-gray-200 group-hover:text-white">{t.logs}</h3>
            </div>
            <ChevronDownIcon className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${isLogOpen ? 'rotate-180' : ''}`} />
        </button>
        
        {isLogOpen && (
            <div className="p-3 bg-gray-900/20 border-t border-gray-700/50 animate-in slide-in-from-top-1 duration-200">
                <ul className="space-y-2 text-sm text-gray-400">
                {steps.map((step, index) => (
                    <li key={index} className="flex gap-2 items-start">
                    <span className="text-green-400 mt-0.5 font-bold">✓</span>
                    <span><strong className="text-gray-300">{step.moduleName}:</strong> {step.description}</span>
                    </li>
                ))}
                </ul>
            </div>
        )}
      </div>

      {/* Network Sources Collapsible (Only if sources exist) */}
      {sources && sources.length > 0 && (
        <div className="border border-green-900/50 rounded-md overflow-hidden bg-green-900/10 shadow-sm">
            <button
                onClick={() => setIsSourcesOpen(!isSourcesOpen)}
                className="flex items-center justify-between w-full p-3 text-left bg-green-900/20 hover:bg-green-900/30 transition-colors group"
            >
                <div className="flex items-center gap-2">
                    <GlobeIcon className="w-4 h-4 text-green-400" />
                    <h3 className="text-sm font-semibold text-gray-200 group-hover:text-white">{t.sources}</h3>
                    <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded-full bg-green-950/80 border border-green-800">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        <span className="text-[10px] font-bold text-green-400 tracking-wider">{t.live}</span>
                    </div>
                </div>
                <ChevronDownIcon className={`w-4 h-4 text-green-400/70 transition-transform duration-200 ${isSourcesOpen ? 'rotate-180' : ''}`} />
            </button>

            {isSourcesOpen && (
                <div className="p-3 bg-gray-900/40 border-t border-green-900/30 grid grid-cols-1 gap-2 animate-in slide-in-from-top-1 duration-200">
                    {sources.map((source, idx) => (
                        <a 
                            key={idx} 
                            href={source.uri} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 p-2 bg-green-900/10 border border-green-800/50 rounded hover:bg-green-900/30 hover:border-green-600 transition-all group/link"
                            title={source.title}
                        >
                            <span className="font-mono text-[10px] text-green-600 group-hover/link:text-green-400">[{idx + 1}]</span>
                            <span className="text-xs text-green-300 group-hover/link:text-green-100 truncate underline decoration-green-800 group-hover/link:decoration-green-400">{source.title}</span>
                        </a>
                    ))}
                </div>
            )}
        </div>
      )}

      {/* Main Result (Risultante) */}
      <div className="mt-6 animate-in fade-in duration-500 delay-100">
        <div className="flex items-center gap-2 mb-4 border-b border-gray-700 pb-2">
            <h3 className="text-lg font-semibold text-cyan-300">{t.title}</h3>
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">{t.subTitle}</span>
        </div>
        
        <div className="space-y-5">
          <div className="p-4 bg-gray-800/40 border border-gray-700 rounded-lg shadow-lg hover:border-gray-600 transition-colors">
            <h4 className="font-bold text-cyan-100 mb-2 flex items-center gap-2">
                <span className="text-xs bg-cyan-900/50 text-cyan-400 px-1.5 py-0.5 rounded border border-cyan-800">L1</span> 
                {t.l1}
            </h4>
            <p className="text-gray-300 leading-relaxed whitespace-pre-wrap text-sm md:text-base">{risultante.livello1}</p>
          </div>
          
          <div className="p-4 bg-gray-800/40 border border-gray-700 rounded-lg shadow-lg hover:border-gray-600 transition-colors">
            <h4 className="font-bold text-purple-200 mb-2 flex items-center gap-2">
                 <span className="text-xs bg-purple-900/50 text-purple-400 px-1.5 py-0.5 rounded border border-purple-800">L2</span>
                 {t.l2}
            </h4>
            <p className="text-gray-300 leading-relaxed whitespace-pre-wrap text-sm md:text-base">{risultante.livello2}</p>
          </div>
          
          <div className="p-4 bg-gray-800/40 border border-gray-700 rounded-lg shadow-lg hover:border-gray-600 transition-colors">
            <h4 className="font-bold text-orange-200 mb-3 flex items-center gap-2">
                <span className="text-xs bg-orange-900/50 text-orange-400 px-1.5 py-0.5 rounded border border-orange-800">L3</span>
                {t.l3}
            </h4>
            <div className="grid grid-cols-1 gap-4 text-sm">
              <div className="bg-gray-900/30 p-3 rounded border-l-2 border-gray-600">
                  <strong className="text-gray-400 block text-xs uppercase tracking-wider mb-1">{t.diagnosis}</strong>
                  <p className="text-gray-300">{risultante.livello3.diagnosi}</p>
              </div>
              <div className="bg-gray-900/30 p-3 rounded border-l-2 border-gray-600">
                  <strong className="text-gray-400 block text-xs uppercase tracking-wider mb-1">{t.trajectory}</strong>
                  <p className="text-gray-300">{risultante.livello3.traiettoria}</p>
              </div>
              <div className="bg-gray-900/30 p-3 rounded border-l-2 border-gray-600">
                  <strong className="text-gray-400 block text-xs uppercase tracking-wider mb-1">{t.synthesis}</strong>
                  <p className="text-gray-300">{risultante.livello3.sintesi}</p>
              </div>
              <div className="bg-cyan-900/20 p-3 rounded border-l-2 border-cyan-500">
                  <strong className="text-cyan-400 block text-xs uppercase tracking-wider mb-1">{t.imprint}</strong>
                  <p className="text-gray-300 italic">"{risultante.livello3.impronta}"</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Engine Instructions (Prompt) Collapsible - Debug */}
      {prompt && (
        <div className="mt-8 pt-4 border-t border-gray-700/30 opacity-60 hover:opacity-100 transition-opacity">
             <div className="border border-gray-700/30 rounded-md overflow-hidden bg-gray-900/20">
                <button
                    onClick={() => setIsPromptOpen(!isPromptOpen)}
                    className="flex items-center justify-between w-full p-2 text-left hover:bg-gray-800/40 transition-colors"
                >
                     <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{t.debug}</h4>
                     <ChevronDownIcon className={`w-3 h-3 text-gray-500 transition-transform duration-200 ${isPromptOpen ? 'rotate-180' : ''}`} />
                </button>
                {isPromptOpen && (
                    <div className="p-3 bg-gray-900/80 border-t border-gray-700/30">
                        <pre className="text-[10px] text-gray-500 whitespace-pre-wrap font-mono break-words leading-tight overflow-x-auto max-h-64">
                        {prompt}
                        </pre>
                    </div>
                )}
             </div>
        </div>
      )}
    </div>
    );
};

export const OutputPanel: React.FC<OutputPanelProps> = ({ state, error, risultante, processingSteps, prompt, sources, language }) => {
  const t = translations[language].output;

  const renderContent = () => {
    switch (state) {
      case 'loading':
        return <LoadingState t={t} />;
      case 'error':
        return <ErrorState error={error || 'An unknown error occurred.'} t={t} />;
      case 'success':
        if (risultante) {
          return <SuccessState steps={processingSteps} risultante={risultante} prompt={prompt} sources={sources} t={t} />;
        }
        return <ErrorState error={t.missingResult} t={t} />;
      case 'idle':
      default:
        return <IdleState t={t} />;
    }
  };

  return (
    <div className="bg-gray-800/60 rounded-lg p-4 h-full flex flex-col border border-gray-700/50 shadow-2xl overflow-y-auto">
       {renderContent()}
    </div>
  );
};