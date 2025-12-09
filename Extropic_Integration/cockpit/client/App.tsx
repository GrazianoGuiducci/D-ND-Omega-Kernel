import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import Sidebar from './components/Sidebar';
import DocumentationModal from './components/DocumentationModal';
import AiAnalysisModal from './components/AiAnalysisModal';
import ExperimentalForgeModal from './components/ExperimentalForgeModal';
import OmegaCockpit from './components/OmegaCockpit';
import SystemStatusFooter from './components/SystemStatusFooter';
import { THEMES, Theme } from './themes';
import { generateManagementData, generateCashFlowData, generateRatingData, generateFinanceData } from './services/mockDataService';
import { MonthlyData, CashFlowDataPoint, RatingData, FinanceOpportunity, UploadedFile, Scenario, SystemLog } from './types';
import { getKernelIntent } from './services/kernelBridge.service';


const App: React.FC = () => {
    // THEME STATE
    const [currentTheme, setCurrentTheme] = useState<Theme>('deepVoid');

    // DATA LAYER INITIALIZATION (For Finance Mode - Kept for potential future simulation use)
    const [managementData, setManagementData] = useState<MonthlyData[]>([]);
    const [cashFlowData, setCashFlowData] = useState<CashFlowDataPoint[]>([]);
    const [ratingData, setRatingData] = useState<RatingData | null>(null);
    const [financeData, setFinanceData] = useState<FinanceOpportunity[]>([]);

    // SCENARIO STATE
    const [scenarios, setScenarios] = useState<Scenario[]>([]);
    const [activeScenarioId, setActiveScenarioId] = useState<string | null>(null);

    // SYSTEM LOGS
    const [systemLogs, setSystemLogs] = useState<SystemLog[]>([]);
    const addLog = (source: SystemLog['source'], message: string, type: SystemLog['type'] = 'info') => {
        setSystemLogs(prev => [{
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            source,
            message,
            type
        }, ...prev].slice(0, 50));
    };

    // UI STATE
    // Default to 'mission_control' view
    const [activeView, setActiveView] = useState<'kernel' | 'mission_control' | 'financial_lab'>('mission_control');
    const [selectedExperimentId, setSelectedExperimentId] = useState<string | undefined>(undefined);
    const [isBuilderOpen, setIsBuilderOpen] = useState(false);
    const [builderTab, setBuilderTab] = useState<'registry' | 'forge'>('registry');
    const [showTooltips, setShowTooltips] = useState(true); // Default ON

    // MODAL STATES
    const [isDocsOpen, setIsDocsOpen] = useState(false);
    const [isAiOpen, setIsAiOpen] = useState(false);
    const [isForgeOpen, setIsForgeOpen] = useState(false);

    // AI INTELLIGENCE LAYER
    const [aiPrompt, setAiPrompt] = useState('');
    const [aiResponse, setAiResponse] = useState('');
    const [isAiLoading, setIsAiLoading] = useState(false);
    const [aiError, setAiError] = useState<string | null>(null);
    const [executiveSummary, setExecutiveSummary] = useState<string | null>(null);

    // HYDRATION EFFECT
    useEffect(() => {
        const timer = setTimeout(() => {
            setManagementData(generateManagementData());
            setCashFlowData(generateCashFlowData());
            setRatingData(generateRatingData());
            setFinanceData(generateFinanceData());
            addLog('SYSTEM', 'Mission Control Systems Online', 'success');
        }, 500);
        return () => clearTimeout(timer);
    }, []);

    // THEME ENGINE ACTIVATION
    useEffect(() => {
        try {
            const themeVars = THEMES[currentTheme].vars;
            const root = document.documentElement;
            Object.entries(themeVars).forEach(([key, value]) => {
                root.style.setProperty(key, value as string);
            });
            document.body.style.backgroundColor = themeVars['--bg-base'];
            document.body.style.color = themeVars['--text-main'];
        } catch (e) {
            console.warn("Theme Engine Error:", e);
        }
    }, [currentTheme]);

    const handleToggleTheme = () => {
        const themes: Theme[] = ['deepVoid', 'cyberpunk', 'matrix', 'vaporwave', 'orbital'];
        const currentIndex = themes.indexOf(currentTheme);
        const nextTheme = themes[(currentIndex + 1) % themes.length];
        setCurrentTheme(nextTheme);
        addLog('USER', `Theme Switched to ${nextTheme.toUpperCase()}`, 'info');
    };

    const handleAiRequest = async (file: UploadedFile | null) => {
        setIsAiLoading(true);
        setAiError(null);
        addLog('AI', 'Processing Neural Request...', 'info');

        const context = JSON.stringify({
            management: managementData.slice(-3),
            cashFlow: cashFlowData.filter(d => d.type === 'Forecast').slice(0, 3),
            rating: ratingData
        });

        try {
            const response = await getKernelIntent(aiPrompt, context, file || undefined);
            setAiResponse(response.analysis);
            setExecutiveSummary(response.analysis.substring(0, 300) + "...");

            if (response.dashboardUpdates) {
                if (response.dashboardUpdates.managementData) setManagementData(response.dashboardUpdates.managementData);
                if (response.dashboardUpdates.cashFlowData) setCashFlowData(response.dashboardUpdates.cashFlowData);
                if (response.dashboardUpdates.ratingData) setRatingData(response.dashboardUpdates.ratingData);
                if (response.dashboardUpdates.financeData) setFinanceData(response.dashboardUpdates.financeData);
                addLog('KERNEL', 'State Injection Completed.', 'success');
            } else {
                addLog('AI', 'Analysis Complete.', 'success');
            }
        } catch (err: any) {
            setAiError(err.message || "Neural Link Failed");
            addLog('SYSTEM', `AI Error: ${err.message}`, 'error');
        } finally {
            setIsAiLoading(false);
        }
    };

    const handleSmartAction = (context: string, extraData?: any) => {
        setAiPrompt(`Analizza i dati di ${context} e suggerisci ottimizzazioni.`);
        setIsAiOpen(true);
        addLog('USER', `Smart Action: ${context}`, 'info');
    };

    const handleInjectExperiment = (code: string, filename: string) => {
        addLog('KERNEL', `Injecting Experiment: ${filename}`, 'warning');
        console.log("INJECTING CODE:", code);
        setIsForgeOpen(false);
        // TODO: Call backend API to save file
    };

    // SCENARIO HANDLERS
    const saveScenario = (name: string) => {
        if (!ratingData) return;
        const newScenario: Scenario = {
            id: Date.now().toString(),
            name,
            timestamp: Date.now(),
            data: {
                management: [...managementData],
                cashFlow: [...cashFlowData],
                rating: { ...ratingData },
                finance: [...financeData]
            }
        };
        setScenarios(prev => [...prev, newScenario]);
        addLog('SYSTEM', `Scenario Saved: "${name}"`, 'success');
    };

    const loadScenario = (id: string) => {
        if (id === 'live') {
            setActiveScenarioId(null);
            addLog('SYSTEM', 'Switched to Live/Edit Mode', 'info');
            return;
        }
        const scenario = scenarios.find(s => s.id === id);
        if (scenario) {
            setManagementData(scenario.data.management);
            setCashFlowData(scenario.data.cashFlow);
            setRatingData(scenario.data.rating);
            setFinanceData(scenario.data.finance);
            setActiveScenarioId(id);
            addLog('SYSTEM', `Scenario Loaded: "${scenario.name}"`, 'warning');
        }
    };

    const deleteScenario = (id: string) => {
        setScenarios(prev => prev.filter(s => s.id !== id));
        if (activeScenarioId === id) setActiveScenarioId(null);
        addLog('USER', 'Scenario Deleted', 'info');
    };

    return (
        <div className="h-screen flex overflow-hidden font-mono bg-black text-white selection:bg-neon-cyan/30">
            {/* Background Texture */}
            <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-10 pointer-events-none"></div>

            {/* MAIN CONTENT AREA */}
            <div className="flex-1 flex flex-col relative z-10 min-w-0 overflow-hidden">
                <Header
                    onAskAiClick={() => setIsAiOpen(true)}
                    onOpenDocs={() => setIsDocsOpen(true)}
                    onToggleTheme={handleToggleTheme}
                    currentTheme={currentTheme}
                    onToggleKernel={() => setActiveView(activeView === 'kernel' ? 'mission_control' : 'kernel')}
                    showTooltips={showTooltips}
                    onToggleTooltips={() => setShowTooltips(!showTooltips)}

                />

                <main className="flex-grow overflow-hidden relative">
                    {activeView === 'kernel' ? (
                        <OmegaCockpit
                            onClose={() => setActiveView('mission_control')}
                            onOpenDocs={() => setIsDocsOpen(true)}
                            initialExperimentId={selectedExperimentId}
                            showTooltips={showTooltips}
                        />
                    ) : (
                        <div className="h-full overflow-hidden">
                            {ratingData ? (
                                <Dashboard
                                    activeView={activeView}
                                    activeExperimentId={selectedExperimentId}
                                    onLaunchExperiment={(id) => {
                                        setSelectedExperimentId(id);
                                        // Check if it's a Financial Protocol
                                        if (['liquidity_annealing', 'volatility_harvesting', 'arbitrage_crystallization'].includes(id)) {
                                            setActiveView('financial_lab');
                                            addLog('SYSTEM', `Initializing Financial Lab: ${id}`, 'success');
                                        } else {
                                            setActiveView('kernel');
                                            addLog('SYSTEM', `Launching Protocol: ${id}`, 'success');
                                        }
                                    }}
                                    onBackToMissionControl={() => setActiveView('mission_control')}

                                    // Financial Data
                                    managementData={managementData}
                                    cashFlowData={cashFlowData}
                                    ratingData={ratingData}
                                    financeData={financeData}

                                    // AI & Logic
                                    executiveSummary={executiveSummary}
                                    onCloseSummary={() => setExecutiveSummary(null)}
                                    onSmartAction={handleSmartAction}
                                    currentTheme={currentTheme}

                                    // Builder State
                                    isBuilderOpen={isBuilderOpen}
                                    setIsBuilderOpen={setIsBuilderOpen}
                                    builderTab={builderTab}

                                    // Forge State
                                    onOpenForge={() => setIsForgeOpen(true)}

                                    // Scenarios
                                    scenarios={scenarios}
                                    activeScenarioId={activeScenarioId}
                                    onSaveScenario={saveScenario}
                                    onLoadScenario={loadScenario}
                                    onDeleteScenario={deleteScenario}

                                    // Tooltips
                                    showTooltips={showTooltips}
                                />
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center">
                                    <div className="w-16 h-16 border-4 border-neon-cyan border-t-transparent rounded-full animate-spin"></div>
                                    <div className="mt-4 font-mono text-xs text-neon-cyan animate-pulse">BOOTSTRAPPING MISSION CONTROL...</div>
                                </div>
                            )}
                        </div>
                    )}
                </main>

                <SystemStatusFooter
                    logs={systemLogs}
                    kernelStatus={activeView === 'kernel' ? 'ACTIVE' : 'BACKGROUND'}
                />
            </div>

            {/* MODALS */}
            <DocumentationModal isOpen={isDocsOpen} onClose={() => setIsDocsOpen(false)} />

            <ExperimentalForgeModal
                isOpen={isForgeOpen}
                onClose={() => setIsForgeOpen(false)}
                onInject={handleInjectExperiment}
            />

            <AiAnalysisModal
                isOpen={isAiOpen}
                onClose={() => setIsAiOpen(false)}
                prompt={aiPrompt}
                setPrompt={setAiPrompt}
                response={aiResponse}
                isLoading={isAiLoading}
                error={aiError}
                onAsk={handleAiRequest}
            />
        </div>
    );
};

export default App;
