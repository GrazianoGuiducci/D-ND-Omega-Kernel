
export type DocSectionId = 'identity' | 'kernel' | 'extropic_guide' | 'forge' | 'scenarios' | 'architecture' | 'ux' | 'troubleshooting';

export interface DocSection {
    id: DocSectionId;
    label: string;
    iconType: 'sparkle' | 'square' | 'swatch' | 'book' | 'alert' | 'clock' | 'cpu';
    color: string;
    title: string;
    subtitle: string;
    content: DocBlock[];
}

export interface DocBlock {
    type: 'text' | 'code' | 'alert' | 'list' | 'step';
    title?: string;
    value: string | string[];
    meta?: string;
}

export const DOCS_DATA: DocSection[] = [
    {
        id: 'identity',
        label: 'System Identity',
        iconType: 'sparkle',
        color: 'text-neon-cyan',
        title: 'E2E Proactive Controller',
        subtitle: 'CFO Sintetico & Architettura Cognitiva',
        content: [
            {
                type: 'text',
                value: "Questo sistema rappresenta un salto evolutivo rispetto alle dashboard tradizionali. Non si limita a visualizzare dati storici (Read-Only), ma possiede capacità di 'State Injection' che gli permettono di modificare la realtà simulata della dashboard in risposta a comandi strategici complessi."
            },
            {
                type: 'alert',
                title: 'Core Directive',
                value: "Perceive (Ingest Data) -> Reason (Gemini 2.0 Flash) -> Act (Mutate State)."
            },
            {
                type: 'list',
                title: 'Funzionalità Attive',
                value: [
                    "Neural Interface: Analisi Multimodale (PDF/Immagini)",
                    "Omega Kernel: Motore Fisico per simulazioni di mercato",
                    "Experimental Forge: Generazione Protocolli Python via AI",
                    "Widget Builder: Costruttore di Dashboard No-Code",
                    "Time Machine: Gestione Scenari Temporali (Snapshot)"
                ]
            }
        ]
    },
    {
        id: 'kernel',
        label: 'Omega Kernel',
        iconType: 'cpu',
        color: 'text-purple-400',
        title: 'Omega Physics Engine',
        subtitle: 'Meccanica Combinatoria & Econofisica',
        content: [
            {
                type: 'text',
                value: "Il Kernel Omega (accessibile dall'Header) sostituisce la logica aziendale lineare con un modello di Ising (Spin Glass). Tratta le entità finanziarie come particelle in un campo di forza."
            },
            {
                type: 'list',
                title: 'Moduli del Cockpit',
                value: [
                    "Control Matrix: Regola Temperatura (Volatilità) e Input Vettoriale.",
                    "Visual Cortex: Visualizza i nodi (Asset/Entità) e le loro interazioni.",
                    "Metric Tensor: Mappa di calore dello stress del sistema (Curvatura).",
                    "Topology Labs: Libreria di esperimenti (es. 'Market Crash', 'Liquidity Annealing')."
                ]
            },
            {
                type: 'code',
                title: 'Parametri Fisici > Finanziari',
                value: "Gravity = Correlazione di Mercato\nPotential = Bias/Trend Macroeconomico\nTemperature = Volatilità (VIX)",
                meta: "Mapping Econofisico in services/omegaPhysics.ts"
            }
        ]
    },
    {
        id: 'extropic_guide',
        label: 'Extropic Guide',
        iconType: 'cpu',
        color: 'text-green-400',
        title: 'Extropic Researcher Guide',
        subtitle: 'Thermodynamic Computing Interface',
        content: [
            {
                type: 'text',
                value: "Welcome to the D-ND Omega Kernel, the bridge between Logic and Thermodynamics. This interface visualizes the 'Metric Tensor' of the cognitive field, where reasoning is a thermodynamic process of annealing."
            },
            {
                type: 'step',
                title: 'Mission Control',
                value: "The host environment managing multiple protocols. Launch 'Active Protocols' (Chaos, Order, Genesis) to initialize the Cockpit with specific physics parameters."
            },
            {
                type: 'list',
                title: 'The Omega Cockpit',
                value: [
                    "Control Matrix (Left): Inject Intent (Perturbation) and adjust Temperature (Noise).",
                    "Visual Cortex (Center): Real-time rendering of the lattice (Spins) and Metric Tensor (Curvature).",
                    "Didactic Layer: Translates thermodynamic events into logic steps (Rosetta Stone)."
                ]
            },
            {
                type: 'code',
                title: 'Concept Mapping',
                value: "Duality -> Dipole / Spin State\nNon-Duality -> Equilibrium\nDialectic Tension -> Potential Gradient\nSynthesis -> Ground State",
                meta: "Theoretical Isomorphism"
            },
            {
                type: 'alert',
                title: 'Operational Workflow',
                value: "1. Select Protocol -> 2. Inject Intent -> 3. Perturbation (Chaos) -> 4. Annealing (Cooling) -> 5. Crystallization (Solution) -> 6. Manifestation (Output)."
            }
        ]
    },
    {
        id: 'forge',
        label: 'The Forge',
        iconType: 'square',
        color: 'text-orange-400',
        title: 'Experimental Forge',
        subtitle: 'Generazione Protocolli AI',
        content: [
            {
                type: 'text',
                value: "La Experimental Forge permette di generare nuovi esperimenti Python per il Kernel usando il linguaggio naturale."
            },
            {
                type: 'step',
                title: 'Architect Mode',
                value: "Descrivi l'esperimento (es. 'Simula un sistema a 3 corpi con alta entropia') e l'AI scriverà il codice Python compatibile con il Kernel."
            },
            {
                type: 'step',
                title: 'Blueprint Mode',
                value: "Visualizza, modifica e inietta il codice generato direttamente nel motore di esecuzione."
            },
            {
                type: 'alert',
                title: 'Widget Builder',
                value: "Per la creazione di grafici UI, usa il 'Widget Forge' accessibile dal Financial Lab."
            }
        ]
    },
    {
        id: 'scenarios',
        label: 'Time Machine',
        iconType: 'clock',
        color: 'text-neon-cyan',
        title: 'Scenario Manager',
        subtitle: 'Navigazione Temporale degli Stati',
        content: [
            {
                type: 'text',
                value: "Il sistema mantiene una cronologia degli stati simulati. La barra 'Time Machine' nella dashboard permette di salvare 'Snapshot' della situazione attuale e confrontarli."
            },
            {
                type: 'list',
                title: 'Workflow Tipico',
                value: [
                    "1. Parti dallo stato 'Live' (Dati reali).",
                    "2. Chiedi all'AI: 'Simula un calo del fatturato del 20%'.",
                    "3. L'AI inietta i nuovi dati.",
                    "4. Salva questo stato come Scenario 'Pessimistico 2025'.",
                    "5. Clicca su 'Live State' per tornare alla realtà e creare un nuovo scenario."
                ]
            }
        ]
    },
    {
        id: 'architecture',
        label: 'Architettura',
        iconType: 'book',
        color: 'text-green-400',
        title: 'Tech Stack & Privacy',
        subtitle: 'Client-Side Processing',
        content: [
            {
                type: 'text',
                value: "Per garantire la privacy finanziaria, il sistema esegue l'ETL (Extract, Transform, Load) direttamente nel browser dell'utente."
            },
            {
                type: 'code',
                title: 'Supported Formats',
                value: "Input: PDF, PNG, JPEG, WEBP\nBlocked: Excel (.xlsx) per sicurezza (richiede server-side parsing sicuro).",
                meta: "FileUploader.tsx"
            },
            {
                type: 'alert',
                title: 'System Logs',
                value: "Il 'System Status Footer' in basso mostra in tempo reale i log di pensiero del Kernel, la latenza dell'AI e l'utilizzo della memoria."
            }
        ]
    },
    {
        id: 'ux',
        label: 'Design System',
        iconType: 'swatch',
        color: 'text-pink-400',
        title: 'Deep Void UI',
        subtitle: 'Estetica Funzionale',
        content: [
            {
                type: 'text',
                value: "L'interfaccia utilizza variabili CSS RGB dinamiche per permettere effetti di 'Glassmorphism' reale senza perdere accessibilità."
            },
            {
                type: 'list',
                title: 'Temi Disponibili',
                value: [
                    "DEEP VOID (Default): Minimalismo assoluto.",
                    "CYBERPUNK: Alto contrasto neon.",
                    "MATRIX: Palette monocromatica verde.",
                    "VAPORWAVE: Estetica retro-futurista.",
                    "ORBITAL: Stile professionale pulito (Blu/Slate)."
                ]
            }
        ]
    },
    {
        id: 'troubleshooting',
        label: 'Troubleshooting',
        iconType: 'alert',
        color: 'text-red-400',
        title: 'Diagnostica',
        subtitle: 'Risoluzione Problemi Comuni',
        content: [
            {
                type: 'alert',
                title: 'THERMAL_RUNAWAY',
                value: "Se il Kernel Omega smette di rispondere durante una simulazione fisica, significa che la 'Temperatura' è troppo alta e il sistema non riesce a convergere (Minimizzare l'Hamiltoniana). Riduci la temperatura nel Control Matrix."
            },
            {
                type: 'alert',
                title: 'AI_RATE_LIMIT',
                value: "Se la Neural Interface risponde con errori, verifica la quota API di OpenRouter/Gemini. Il sistema usa modelli Flash/Pro che possono avere limiti di frequenza."
            }
        ]
    }
];
