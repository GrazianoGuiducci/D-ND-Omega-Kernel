import React, { useState, useEffect } from 'react';
import { EXPERIMENTS } from '../services/omegaPhysics';
import ExperimentCard from './ExperimentCard';
import { Resizer } from './Resizer';
import SmartTooltip from './SmartTooltip';

// Financial Imports
import ManagementControlCard from './ManagementControlCard';
import CashFlowCard from './CashFlowCard';
import BankabilityRatingCard from './BankabilityRatingCard';
import SubsidizedFinanceCard from './SubsidizedFinanceCard';
import AiExecutiveSummaryCard from './AiExecutiveSummaryCard';
import DynamicWidgetCard from './DynamicWidgetCard';
import WidgetBuilderModal from './WidgetBuilderModal';
import { WidgetConfig, Scenario, MonthlyData, CashFlowDataPoint, RatingData, FinanceOpportunity } from '../types';
import { Theme } from '../themes';

interface DashboardProps {
    activeView?: 'mission_control' | 'financial_lab' | 'kernel';
    activeExperimentId?: string; // NEW: To know which protocol is driving the lab
    onLaunchExperiment?: (experimentId: string) => void;
    onBackToMissionControl?: () => void;
    onOpenForge?: () => void;

    // Financial Data
    managementData?: MonthlyData[];
    cashFlowData?: CashFlowDataPoint[];
    ratingData?: RatingData | null;
    financeData?: FinanceOpportunity[];

    // AI & Logic
    executiveSummary?: string | null;
    onCloseSummary?: () => void;
    onSmartAction?: (context: string, extraData?: any) => void;
    currentTheme?: Theme;

    // Builder State
    isBuilderOpen?: boolean;
    setIsBuilderOpen?: (isOpen: boolean) => void;
    builderTab?: 'registry' | 'forge';

    // Scenarios
    scenarios?: Scenario[];
    activeScenarioId?: string | null;
    onSaveScenario?: (name: string) => void;
    onLoadScenario?: (id: string) => void;
    onDeleteScenario?: (id: string) => void;

    showTooltips?: boolean;
}

const Dashboard: React.FC<DashboardProps> = ({
    activeView = 'mission_control',
    activeExperimentId,
    onLaunchExperiment,
    onBackToMissionControl,
    onOpenForge,
    managementData = [],
    cashFlowData = [],
    ratingData,
    financeData = [],
    executiveSummary,
    onCloseSummary,
    onSmartAction,
    currentTheme = 'deepVoid',
    isBuilderOpen = false,
    setIsBuilderOpen,
    builderTab = 'registry',
    scenarios = [],
    activeScenarioId,
    onSaveScenario,
    onLoadScenario,
    onDeleteScenario,
    showTooltips = true
}) => {
    // Layout State
    const [leftPanelWidth, setLeftPanelWidth] = useState(280);
    const [rightPanelWidth, setRightPanelWidth] = useState(320);
    const [isResizingLeft, setIsResizingLeft] = useState(false);
    const [isResizingRight, setIsResizingRight] = useState(false);

    // Filter State
    const [filterCategory, setFilterCategory] = useState<'ALL' | 'PHYSICS' | 'CONCEPT' | 'FINANCE'>('ALL');

    // Widget State (for Financial Lab)
    const [widgets, setWidgets] = useState<WidgetConfig[]>([]);

    // --- PERSISTENCE: LOAD WIDGETS ---
    useEffect(() => {
        const loadWidgets = async () => {
            try {
                const res = await fetch('/api/widgets');
                if (res.ok) {
                    const data = await res.json();
                    if (data.widgets && Array.isArray(data.widgets)) {
                        setWidgets(data.widgets);
                    }
                }
            } catch (e) {
                console.error("Failed to load widgets", e);
            }
        };
        loadWidgets();
    }, []);

    // --- PERSISTENCE: SAVE WIDGETS ---
    useEffect(() => {
        if (widgets.length === 0) return; // Don't wipe on empty init

        const saveWidgets = async () => {
            try {
                await fetch('/api/widgets', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ widgets })
                });
            } catch (e) {
                console.error("Failed to save widgets", e);
            }
        };

        // Debounce save to avoid spamming server
        const timer = setTimeout(saveWidgets, 1000);
        return () => clearTimeout(timer);
    }, [widgets]);

    // Resizing Logic (Left)
    useEffect(() => {
        if (!isResizingLeft) return;
        const onMouseMove = (e: MouseEvent) => setLeftPanelWidth(Math.max(200, Math.min(e.clientX, 500)));
        const onMouseUp = () => setIsResizingLeft(false);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        return () => { document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); };
    }, [isResizingLeft]);

    // Resizing Logic (Right)
    useEffect(() => {
        if (!isResizingRight) return;
        const onMouseMove = (e: MouseEvent) => setRightPanelWidth(Math.max(250, Math.min(window.innerWidth - e.clientX, 500)));
        const onMouseUp = () => setIsResizingRight(false);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        return () => { document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); };
    }, [isResizingRight]);

    // Helper to get active config
    const activeConfig = activeExperimentId ? EXPERIMENTS.find(e => e.id === activeExperimentId) : null;

    // --- RENDER: FINANCIAL LAB CONTENT ---
    const renderFinancialLab = () => (
        <div className="h-full overflow-y-auto custom-scrollbar bg-[#050505] relative flex flex-col">
            {/* Header / Toolbar for Lab */}
            <div className="shrink-0 flex justify-between items-center px-6 py-4 sticky top-0 z-30 bg-[#050505]/95 backdrop-blur-md border-b border-white/10 shadow-lg">
                <div className="flex items-center gap-6">
                    <button
                        onClick={onBackToMissionControl}
                        className="group flex items-center gap-2 text-xs font-mono text-slate-400 hover:text-neon-cyan transition-colors"
                    >
                        <span className="group-hover:-translate-x-1 transition-transform">←</span>
                        MISSION_CONTROL
                    </button>

                    <div className="h-8 w-px bg-white/10"></div>

                    <div>
                        <h1 className="text-lg font-bold text-white tracking-widest uppercase flex items-center gap-3">
                            <span className="text-neon-cyan">♦</span>
                            Financial Determinism Lab
                        </h1>
                        {activeConfig && (
                            <div className="flex gap-4 text-[10px] font-mono text-slate-500 mt-1">
                                <span className="text-cyan-400 font-bold">PROTOCOL: {activeConfig.name}</span>
                                <span className="text-slate-700">|</span>
                                <span className="flex items-center gap-1">
                                    <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></span>
                                    TEMP: {activeConfig.params.temperature}K
                                </span>
                                <span className="flex items-center gap-1">
                                    <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
                                    GRAVITY: {activeConfig.params.gravity}
                                </span>
                            </div>
                        )}
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex flex-col items-end mr-4">
                        <span className="text-[9px] text-slate-500 uppercase tracking-widest">Simulation Status</span>
                        <span className="text-xs font-mono text-green-400 animate-pulse">● RUNNING</span>
                    </div>

                    <SmartTooltip source="Widget Forge" logic="Dynamic UI" outcome="Custom View" isVisible={showTooltips} position="left">
                        <button
                            onClick={() => setIsBuilderOpen && setIsBuilderOpen(true)}
                            className="text-xs font-mono bg-neon-cyan/5 text-neon-cyan border border-neon-cyan/30 px-4 py-2 rounded hover:bg-neon-cyan/10 transition-all hover:shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                        >
                            + WIDGET FORGE
                        </button>
                    </SmartTooltip>
                </div>
            </div>

            {/* Executive Summary Overlay */}
            {executiveSummary && (
                <div className="px-6 pt-6">
                    <AiExecutiveSummaryCard summary={executiveSummary} onClose={onCloseSummary || (() => { })} />
                </div>
            )}

            {/* Main Content Area */}
            <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-8 pb-20 max-w-[1600px] mx-auto w-full">

                {/* LEFT COLUMN: INTERNAL THERMODYNAMICS (Energy & Entropy) */}
                <div className="lg:col-span-8 space-y-8">
                    {/* Section Header */}
                    <div className="flex items-center gap-3 pb-2 border-b border-white/10">
                        <div className="text-neon-cyan text-lg">⚡</div>
                        <div>
                            <h3 className="text-sm font-bold text-white uppercase tracking-widest">Internal Thermodynamics</h3>
                            <p className="text-[10px] text-slate-500 font-mono">System Energy (P&L) & Entropy Flux (Cash Flow)</p>
                        </div>
                    </div>

                    <SmartTooltip source="ERP Connector" logic="P&L Aggregation" outcome="Margin Analysis" isVisible={showTooltips} position="top">
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg blur opacity-0 group-hover:opacity-100 transition duration-500"></div>
                            <div className="relative">
                                <ManagementControlCard
                                    data={managementData}
                                    onAction={() => onSmartAction && onSmartAction('Management Control P&L')}
                                    currentTheme={currentTheme}
                                />
                            </div>
                        </div>
                    </SmartTooltip>

                    <SmartTooltip source="Treasury Stream" logic="Liquidity Forecast" outcome="Solvency Check" isVisible={showTooltips} position="bottom">
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg blur opacity-0 group-hover:opacity-100 transition duration-500"></div>
                            <div className="relative">
                                <CashFlowCard
                                    data={cashFlowData}
                                    onAction={() => onSmartAction && onSmartAction('Cash Flow Liquidity')}
                                    currentTheme={currentTheme}
                                />
                            </div>
                        </div>
                    </SmartTooltip>
                </div>

                {/* RIGHT COLUMN: STRUCTURAL INTEGRITY (Stability & External) */}
                <div className="lg:col-span-4 space-y-8">
                    {/* Section Header */}
                    <div className="flex items-center gap-3 pb-2 border-b border-white/10">
                        <div className="text-purple-400 text-lg">🛡️</div>
                        <div>
                            <h3 className="text-sm font-bold text-white uppercase tracking-widest">Structural Integrity</h3>
                            <p className="text-[10px] text-slate-500 font-mono">Coherence (Rating) & Negentropy (Subsidies)</p>
                        </div>
                    </div>

                    {ratingData && (
                        <SmartTooltip source="Basel Engine" logic="Risk Scoring" outcome="Credit Rating" isVisible={showTooltips} position="left">
                            <div className="relative group">
                                <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg blur opacity-0 group-hover:opacity-100 transition duration-500"></div>
                                <div className="relative">
                                    <BankabilityRatingCard
                                        data={ratingData}
                                        onAction={() => onSmartAction && onSmartAction('Bankability Rating')}
                                        currentTheme={currentTheme}
                                    />
                                </div>
                            </div>
                        </SmartTooltip>
                    )}

                    <SmartTooltip source="Gov API" logic="Eligibility Match" outcome="Funding Ops" isVisible={showTooltips} position="left">
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-lg blur opacity-0 group-hover:opacity-100 transition duration-500"></div>
                            <div className="relative">
                                <SubsidizedFinanceCard
                                    data={financeData}
                                    onAction={() => onSmartAction && onSmartAction('Subsidized Finance Opportunities')}
                                />
                            </div>
                        </div>
                    </SmartTooltip>

                    {/* Dynamic Widgets Container */}
                    {widgets.length > 0 && (
                        <div className="pt-6 border-t border-white/10">
                            <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-4">Custom Modules</div>
                            <div className="space-y-4">
                                {widgets.map(widget => (
                                    <div key={widget.id} className="relative group">
                                        <div className="absolute -inset-0.5 bg-white/5 rounded-lg blur opacity-0 group-hover:opacity-100 transition duration-500"></div>
                                        <DynamicWidgetCard
                                            config={widget}
                                            data={
                                                widget.dataSource === 'management' ? managementData :
                                                    widget.dataSource === 'cashflow' ? cashFlowData :
                                                        widget.dataSource === 'rating' && ratingData ? [ratingData] :
                                                            []
                                            }
                                            onAction={() => onSmartAction && onSmartAction(`Custom Widget: ${widget.title}`)}
                                            onDelete={() => setWidgets(prev => prev.filter(w => w.id !== widget.id))}
                                            currentTheme={currentTheme}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Widget Builder Modal */}
            <WidgetBuilderModal
                isOpen={isBuilderOpen}
                onClose={() => setIsBuilderOpen && setIsBuilderOpen(false)}
                onSave={(w) => setWidgets(prev => [...prev, w])}
                modules={widgets}
                onUpdateModule={(updated) => setWidgets(prev => prev.map(w => w.id === updated.id ? updated : w))}
                onDeleteModule={(id) => setWidgets(prev => prev.filter(w => w.id !== id))}
                initialTab={builderTab}
            />
        </div>
    );

    // --- RENDER: MISSION CONTROL CONTENT ---
    const filteredExperiments = EXPERIMENTS.filter(exp =>
        filterCategory === 'ALL' ? true : exp.category === filterCategory
    );

    const renderMissionControl = () => (
        <div className="flex-1 overflow-y-auto custom-scrollbar p-8 relative z-10" style={{ backgroundColor: 'var(--bg-base)' }}>
            {/* Background Grid */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(30,30,30,0.2)_1px,transparent_1px),linear-gradient(90deg,rgba(30,30,30,0.2)_1px,transparent_1px)] bg-[length:40px_40px] pointer-events-none opacity-20" style={{ backgroundImage: `linear-gradient(var(--grid-color) 1px, transparent 1px), linear-gradient(90deg, var(--grid-color) 1px, transparent 1px)` }}></div>

            <div className="max-w-6xl mx-auto relative z-10">
                <div className="flex justify-between items-end mb-8">
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight mb-2" style={{ color: 'var(--text-main)' }}>Active Protocols</h1>
                        <p className="text-sm font-mono" style={{ color: 'var(--text-sub)' }}>Select a protocol to initialize the Omega Kernel.</p>
                    </div>
                    <div className="text-right flex gap-3">
                        <SmartTooltip source="Experimental Forge" logic="Protocol Genesis" outcome="New Experiment" isVisible={showTooltips} position="left">
                            <button
                                onClick={onOpenForge}
                                className="text-xs font-mono bg-neon-purple/10 text-neon-purple border border-neon-purple/30 px-3 py-1 rounded hover:bg-neon-purple/20 flex items-center gap-2"
                            >
                                <span className="text-lg">⚡</span> OPEN FORGE
                            </button>
                        </SmartTooltip>

                        <SmartTooltip source="Module Loader" logic="Availability Check" outcome="Ready State" isVisible={showTooltips} position="left">
                            <span className="text-xs font-mono text-neon-cyan border border-neon-cyan/30 px-2 py-1 rounded bg-neon-cyan/10 cursor-help flex items-center h-full">
                                {filteredExperiments.length} MODULES READY
                            </span>
                        </SmartTooltip>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {filteredExperiments.map(exp => (
                        <ExperimentCard
                            key={exp.id}
                            config={exp}
                            onLaunch={(id) => onLaunchExperiment && onLaunchExperiment(id)}
                        />
                    ))}
                </div>
            </div>
        </div>
    );

    return (
        <div className="flex h-full overflow-hidden font-sans relative" style={{ backgroundColor: 'var(--bg-base)', color: 'var(--text-sub)' }}>

            {/* LEFT SIDEBAR: NAVIGATION */}
            <div
                className="shrink-0 flex flex-col border-r relative z-20"
                style={{ width: leftPanelWidth, backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-muted)' }}
            >
                <div className="p-6 border-b border-[#1f1f1f]">
                    <h2 className="text-xs font-bold text-neon-cyan uppercase tracking-widest mb-2">Mission Control</h2>
                    <p className="text-[10px] text-slate-500 font-mono">System Status: ONLINE</p>
                </div>

                <div className="p-4 space-y-2">
                    <div className="text-[9px] text-slate-600 uppercase tracking-widest font-bold mb-2 pl-2">Filters</div>
                    <SmartTooltip source="Exp. Registry" logic="Category Filter" outcome="View Refinement" isVisible={showTooltips} position="right">
                        <div>
                            {['ALL', 'PHYSICS', 'D_ND_CONCEPT', 'FINANCE'].map(cat => (
                                <button
                                    key={cat}
                                    onClick={() => setFilterCategory(cat as any)}
                                    className={`w-full text-left px-3 py-2 rounded text-xs font-mono transition-colors ${filterCategory === cat
                                        ? 'bg-white/10 text-white border-l-2 border-neon-cyan'
                                        : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
                                        }`}
                                >
                                    {cat === 'D_ND_CONCEPT' ? 'D-ND CONCEPT' : cat}
                                </button>
                            ))}
                        </div>
                    </SmartTooltip>
                </div>

                <div className="mt-auto p-4 border-t border-[#1f1f1f]">
                    <div className="text-[9px] text-slate-600 uppercase tracking-widest font-bold mb-2">System Logs</div>
                    <SmartTooltip source="Kernel Daemon" logic="Event Stream" outcome="System Status" isVisible={showTooltips} position="top" alignment="start">
                        <div className="font-mono text-[9px] text-slate-500 space-y-1 opacity-70">
                            <div>[10:42:01] KERNEL_INIT_OK</div>
                            <div>[10:42:05] DATA_STREAM_SYNC</div>
                            <div>[10:42:12] USER_AUTH_VERIFIED</div>
                        </div>
                    </SmartTooltip>
                </div>
            </div>

            {/* RESIZER LEFT */}
            <Resizer onMouseDown={(e) => { e.preventDefault(); setIsResizingLeft(true); }} isVisible={true} />

            {/* CENTER PANEL */}
            {activeView === 'financial_lab' ? renderFinancialLab() : renderMissionControl()}

            {/* RESIZER RIGHT */}
            <Resizer onMouseDown={(e) => { e.preventDefault(); setIsResizingRight(true); }} isVisible={true} />

            {/* RIGHT SIDEBAR: DETAILS / NEWS */}
            <div
                className="shrink-0 flex flex-col border-l relative z-20"
                style={{ width: rightPanelWidth, backgroundColor: 'var(--bg-surface)', borderColor: 'var(--col-muted)' }}
            >
                <div className="p-6 border-b border-[#1f1f1f]">
                    <h2 className="text-xs font-bold text-purple-400 uppercase tracking-widest mb-2">Extropic News</h2>
                    <p className="text-[10px] text-slate-500 font-mono">Latest Updates</p>
                </div>

                <div className="p-6 space-y-6 overflow-y-auto">
                    <div className="space-y-2">
                        <span className="text-[9px] text-slate-500 font-mono border border-slate-700 px-1 rounded">2025-11-29</span>
                        <h3 className="text-sm font-bold text-white">Omega Kernel v0.2 Released</h3>
                        <p className="text-xs text-slate-400 leading-relaxed">
                            Full integration with React frontend complete. Energy Graph visualization restored. New Concept Protocols added for theoretical testing.
                        </p>
                    </div>

                    <div className="w-full h-px bg-[#1f1f1f]"></div>

                    <div className="space-y-2">
                        <span className="text-[9px] text-slate-500 font-mono border border-slate-700 px-1 rounded">UPCOMING</span>
                        <h3 className="text-sm font-bold text-slate-300">OpenRouter Integration</h3>
                        <p className="text-xs text-slate-500 leading-relaxed">
                            Planned integration with OpenRouter to allow dynamic model selection for the Semantic Parser.
                        </p>
                    </div>
                </div>
            </div>

        </div>
    );
};

export default Dashboard;
