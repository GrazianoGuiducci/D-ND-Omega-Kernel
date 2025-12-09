
export type Theme = 'cyberpunk' | 'matrix' | 'vaporwave' | 'orbital' | 'deepVoid';

export interface ThemeColors {
    name: string;
    isDark: boolean;
    vars: {
        '--bg-base': string;
        '--bg-base-rgb': string;
        '--bg-surface': string;
        '--bg-surface-rgb': string;
        '--text-main': string;
        '--text-sub': string;
        '--col-primary': string;
        '--col-primary-rgb': string;
        '--col-secondary': string;
        '--col-secondary-rgb': string;
        '--col-muted': string;
        '--col-muted-rgb': string;
        '--grid-color': string;
        // Semantic Vars
        '--col-revenue': string;
        '--col-costs': string;
        '--col-profit': string;
        '--col-cash': string;
        '--col-payroll': string;
        '--col-score': string;
        '--col-headcount': string;
    };
    semanticColors: {
        revenue: string;
        costs: string;
        profit: string;
        cash: string;
        payroll: string;
        score: string;
        headcount: string;
        default: string;
    };
}

export const THEMES: Record<Theme, ThemeColors> = {
    deepVoid: {
        name: 'DEEP VOID',
        isDark: true,
        vars: {
            '--bg-base': '#050505',
            '--bg-base-rgb': '5, 5, 5',
            '--bg-surface': 'rgba(20, 20, 25, 0.6)',
            '--bg-surface-rgb': '20, 20, 25',
            '--text-main': '#ffffff',
            '--text-sub': '#94a3b8',
            '--col-primary': '#00f3ff', // Cyan (Logic)
            '--col-primary-rgb': '0, 243, 255',
            '--col-secondary': '#ff00ff', // Magenta (Entropy)
            '--col-secondary-rgb': '255, 0, 255',
            '--col-muted': '#1f1f1f',
            '--col-muted-rgb': '31, 31, 31',
            '--grid-color': 'rgba(0, 243, 255, 0.05)',
            // Semantic Vars
            '--col-revenue': '#00f3ff',
            '--col-costs': '#ff00ff',
            '--col-profit': '#00f3ff',
            '--col-cash': '#facc15',
            '--col-payroll': '#ff00ff',
            '--col-score': '#00f3ff',
            '--col-headcount': '#94a3b8'
        },
        semanticColors: {
            revenue: '#00f3ff',
            costs: '#ff00ff',
            profit: '#00f3ff',
            cash: '#facc15',
            payroll: '#ff00ff',
            score: '#00f3ff',
            headcount: '#94a3b8',
            default: '#94a3b8'
        }
    },
    cyberpunk: {
        name: 'CYBERPUNK',
        isDark: true,
        vars: {
            '--bg-base': '#050b14',
            '--bg-base-rgb': '5, 11, 20',
            '--bg-surface': 'rgba(15, 23, 42, 0.6)',
            '--bg-surface-rgb': '15, 23, 42',
            '--text-main': '#ffffff',
            '--text-sub': '#94a3b8',
            '--col-primary': '#00f3ff',
            '--col-primary-rgb': '0, 243, 255',
            '--col-secondary': '#bc13fe',
            '--col-secondary-rgb': '188, 19, 254',
            '--col-muted': '#334155',
            '--col-muted-rgb': '51, 65, 85',
            '--grid-color': 'rgba(0, 243, 255, 0.1)',
            // Semantic Vars
            '--col-revenue': '#00f3ff',
            '--col-costs': '#f87171',
            '--col-profit': '#4ade80',
            '--col-cash': '#facc15',
            '--col-payroll': '#e879f9',
            '--col-score': '#bc13fe',
            '--col-headcount': '#2dd4bf'
        },
        semanticColors: {
            revenue: '#00f3ff', // Cyan (Info)
            costs: '#f87171',   // Red (Danger)
            profit: '#4ade80',  // Green (Success)
            cash: '#facc15',    // Gold (Value)
            payroll: '#e879f9', // Pink (Warning/High Cost)
            score: '#bc13fe',   // Purple (Metric)
            headcount: '#2dd4bf', // Teal (Info)
            default: '#94a3b8'
        }
    },
    matrix: {
        name: 'MATRIX',
        isDark: true,
        vars: {
            '--bg-base': '#000000',
            '--bg-base-rgb': '0, 0, 0',
            '--bg-surface': 'rgba(0, 20, 0, 0.8)',
            '--bg-surface-rgb': '0, 20, 0',
            '--text-main': '#00ff41',
            '--text-sub': '#008f11',
            '--col-primary': '#00ff41',
            '--col-primary-rgb': '0, 255, 65',
            '--col-secondary': '#003b00',
            '--col-secondary-rgb': '0, 59, 0',
            '--col-muted': '#001a00',
            '--col-muted-rgb': '0, 26, 0',
            '--grid-color': 'rgba(0, 255, 65, 0.2)',
            // Semantic Vars
            '--col-revenue': '#00ff41',
            '--col-costs': '#005c00',
            '--col-profit': '#ccffcc',
            '--col-cash': '#00ff41',
            '--col-payroll': '#008f11',
            '--col-score': '#00ff41',
            '--col-headcount': '#003b00'
        },
        semanticColors: {
            revenue: '#00ff41', // Bright Green
            costs: '#005c00',   // Dark Green
            profit: '#ccffcc',  // Pale Green
            cash: '#00ff41',
            payroll: '#008f11',
            score: '#00ff41',
            headcount: '#003b00',
            default: '#004d00'
        }
    },
    vaporwave: {
        name: 'VAPORWAVE',
        isDark: true,
        vars: {
            '--bg-base': '#180c2e',
            '--bg-base-rgb': '24, 12, 46',
            '--bg-surface': 'rgba(40, 20, 80, 0.6)',
            '--bg-surface-rgb': '40, 20, 80',
            '--text-main': '#ffd6ff',
            '--text-sub': '#c8b6ff',
            '--col-primary': '#ff00d4',
            '--col-primary-rgb': '255, 0, 212',
            '--col-secondary': '#00ffff',
            '--col-secondary-rgb': '0, 255, 255',
            '--col-muted': '#581c87',
            '--col-muted-rgb': '88, 28, 135',
            '--grid-color': 'rgba(255, 0, 212, 0.15)',
            // Semantic Vars
            '--col-revenue': '#00ffff',
            '--col-costs': '#ff00d4',
            '--col-profit': '#f9a8d4',
            '--col-cash': '#facc15',
            '--col-payroll': '#c084fc',
            '--col-score': '#ff00d4',
            '--col-headcount': '#22d3ee'
        },
        semanticColors: {
            revenue: '#00ffff', // Cyan
            costs: '#ff00d4',   // Magenta
            profit: '#f9a8d4',  // Light Pink
            cash: '#facc15',    // Yellow
            payroll: '#c084fc', // Violet
            score: '#ff00d4',
            headcount: '#22d3ee',
            default: '#c8b6ff'
        }
    },
    orbital: {
        name: 'ORBITAL',
        isDark: true,
        vars: {
            '--bg-base': '#0f172a',
            '--bg-base-rgb': '15, 23, 42',
            '--bg-surface': 'rgba(30, 41, 59, 0.7)',
            '--bg-surface-rgb': '30, 41, 59',
            '--text-main': '#e2e8f0',
            '--text-sub': '#94a3b8',
            '--col-primary': '#38bdf8',
            '--col-primary-rgb': '56, 189, 248',
            '--col-secondary': '#fbbf24',
            '--col-secondary-rgb': '251, 191, 36',
            '--col-muted': '#475569',
            '--col-muted-rgb': '71, 85, 105',
            '--grid-color': 'rgba(56, 189, 248, 0.1)',
            // Semantic Vars
            '--col-revenue': '#38bdf8',
            '--col-costs': '#f43f5e',
            '--col-profit': '#fbbf24',
            '--col-cash': '#fbbf24',
            '--col-payroll': '#818cf8',
            '--col-score': '#38bdf8',
            '--col-headcount': '#94a3b8'
        },
        semanticColors: {
            revenue: '#38bdf8', // Sky (Info)
            costs: '#f43f5e',   // Rose (Danger)
            profit: '#fbbf24',  // Amber (Warning/Success)
            cash: '#fbbf24',
            payroll: '#818cf8', // Indigo
            score: '#38bdf8',
            headcount: '#94a3b8',
            default: '#64748b'
        }
    }
};

export const SEMANTIC_CONFIG: Record<string, { twBg: string; twText: string; twBorder: string, intent: 'success' | 'danger' | 'warning' | 'info' | 'neutral' }> = {
    revenue: { twBg: 'bg-cyan-500/20', twText: 'text-cyan-400', twBorder: 'border-cyan-500', intent: 'info' },
    costs: { twBg: 'bg-red-500/20', twText: 'text-red-400', twBorder: 'border-red-500', intent: 'danger' },
    profit: { twBg: 'bg-green-500/20', twText: 'text-green-400', twBorder: 'border-green-500', intent: 'success' },
    cash: { twBg: 'bg-yellow-500/20', twText: 'text-yellow-400', twBorder: 'border-yellow-500', intent: 'warning' },
    payroll: { twBg: 'bg-purple-500/20', twText: 'text-purple-400', twBorder: 'border-purple-500', intent: 'warning' },
    score: { twBg: 'bg-purple-500/20', twText: 'text-purple-400', twBorder: 'border-purple-500', intent: 'info' },
    headcount: { twBg: 'bg-teal-500/20', twText: 'text-teal-400', twBorder: 'border-teal-500', intent: 'info' },
    default: { twBg: 'bg-slate-700/50', twText: 'text-slate-300', twBorder: 'border-slate-500', intent: 'neutral' }
};

// --- NEW: HOLOGRAM COLOR CONFIG FOR WIDGET CONTAINERS ---
export const HOLOGRAM_CONFIG: Record<string, { hex: string; rgb: string }> = {
    cyan: { hex: '#00f3ff', rgb: '0, 243, 255' },
    purple: { hex: '#bc13fe', rgb: '188, 19, 254' },
    green: { hex: '#4ade80', rgb: '74, 222, 128' },
    orange: { hex: '#f97316', rgb: '249, 115, 22' },
    magenta: { hex: '#ff00ff', rgb: '255, 0, 255' } // Deep Void Secondary
};

// Core logic to identify if a key has semantic meaning or is generic
export const isSemanticKey = (key: string): boolean => {
    const lowerKey = key.toLowerCase();
    return Object.keys(SEMANTIC_CONFIG).includes(lowerKey) && lowerKey !== 'default';
};

export const getSemanticColor = (key: string, theme: Theme): string => {
    const colors = THEMES[theme].semanticColors;
    const lowerKey = key.toLowerCase();
    return (colors as any)[lowerKey] || colors.default;
};

// Helper for Custom Widget Colors (Hologram Themes)
export const getHologramColor = (colorTheme: string): string => {
    return HOLOGRAM_CONFIG[colorTheme]?.hex || HOLOGRAM_CONFIG['cyan'].hex;
};
