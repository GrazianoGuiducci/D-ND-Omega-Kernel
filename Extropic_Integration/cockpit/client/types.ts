
export interface MonthlyData {
    month: string;
    revenue: number;
    costs: number;
    profit: number;
    payroll?: number; // Added for HR demo
}

export interface CashFlowDataPoint {
    month: string;
    cash: number;
    type: 'Past' | 'Forecast';
    upperBound?: number;
    lowerBound?: number;
    range?: number[];
}

export interface RatingData {
    score: number;
    rating: string;
    metrics: {
        label: string;
        value: string | number;
        score: number; // Added numeric score for radar charts
    }[];
}

export interface FinanceOpportunity {
    id: string;
    name: string;
    type: string;
    amount: string;
    deadline: string;
}

export interface UploadedFile {
    name: string;
    type: string;
    data: string; // Base64 string
}

export interface AiResponse {
    analysis: string;
    dashboardUpdates?: {
        managementData?: MonthlyData[];
        cashFlowData?: CashFlowDataPoint[];
        ratingData?: RatingData;
        financeData?: FinanceOpportunity[];
    };
}

// --- WIDGET SYSTEM TYPES ---

export type WidgetType = 'bar' | 'line' | 'area' | 'composed' | 'pie' | 'radar' | 'radial';
export type DataSourceType = 'management' | 'cashflow' | 'rating' | 'hr';
export type WidgetColorTheme = 'cyan' | 'purple' | 'green' | 'orange' | 'magenta';

export interface WidgetConfig {
    id: string;
    title: string;
    description?: string;
    type: WidgetType;
    dataSource: DataSourceType;
    dataKeys: string[]; // Keys to visualize (e.g., 'revenue', 'costs')
    colorTheme: WidgetColorTheme;
    isSystem?: boolean; // If true, cannot be deleted (only hidden)
    isVisible?: boolean; // Controls visibility in dashboard
    colSpan?: 1 | 2; // 1 = 50% width (6/12), 2 = 100% width (12/12)
}

export interface AiWidgetResponse {
    config: WidgetConfig;
    explanation: string;
}

// --- OMEGA KERNEL TYPES (COMBINATORIAL MECHANICS) ---

export interface LatticeNode {
    id: number;
    x: number;
    y: number;
    spin: number; // Float -1.0 to 1.0 (Continuous Spin due to tanh)
    stability: number; // 0.0 to 1.0
    potential: number; // Local NullaTutto Potential
    energy: number; // Local Hamiltonian Energy
    assetTicker?: string; // Optional: Linked Asset for Econophysics
}

export interface OmegaMetrics {
    energy: number; // Hamiltonian
    coherence: number; // Order Parameter
    entropy: number;
    temperature: number;
    step: number;
    curvature: number; // Average Metric Curvature
}

export interface DslStep {
    id: number;
    label: string;
    code: string;
    status: 'pending' | 'active' | 'complete';
    rosetta: string;
}

export type ExperimentCategory = 'PHYSICS' | 'FINANCE' | 'D_ND_CONCEPT';

export interface ExperimentConfig {
    id: string;
    name: string;
    description: string;
    category: ExperimentCategory;
    status?: 'ACTIVE' | 'CONCEPT' | 'COMING_SOON'; // NEW
    intent?: string; // Protocol Intent for Backend Injection
    params: {
        gravity: number; // Warp factor for Metric Tensor
        potentialScale: number; // NullaTutto Scale
        temperature: number; // Noise
    };
}

export interface PortfolioAsset {
    ticker: string;
    name: string;
    sector: string;
}

export interface OmegaState {
    lattice: LatticeNode[];
    metrics: OmegaMetrics;
    phase: 'IDLE' | 'PERTURBATION' | 'ANNEALING' | 'CRYSTALLIZED';
    dslTrace: DslStep[];
    activeExperimentId?: string;
    tensorMatrix?: number[][]; // Visual representation of g_uv
    portfolioOverlay?: boolean; // Active if experiment is FINANCIAL
}

// --- NEW: SCENARIO MANAGEMENT TYPES ---

export interface Scenario {
    id: string;
    name: string;
    timestamp: number;
    data: {
        management: MonthlyData[];
        cashFlow: CashFlowDataPoint[];
        rating: RatingData | null;
        finance: FinanceOpportunity[];
    };
    description?: string;
}

export interface SystemLog {
    id: number;
    timestamp: string;
    source: 'KERNEL' | 'AI' | 'SYSTEM' | 'USER';
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
}
