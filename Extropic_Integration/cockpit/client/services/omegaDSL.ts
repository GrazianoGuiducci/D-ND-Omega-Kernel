
import { DslStep } from '../types';

export const generateDslTrace = (intent: string): DslStep[] => {
    if (!intent) return [];

    // Simulate parsing the user intent into logical execution steps
    const trace: DslStep[] = [
        {
            id: 1,
            label: 'SEMANTIC_PARSING',
            code: `PARSE("${intent.substring(0, 20)}${intent.length > 20 ? '...' : ''}")`,
            status: 'pending',
            rosetta: 'Analisi semantica del vettore di input (Natural Language Processing).'
        },
        {
            id: 2,
            label: 'INTENT_EXTRACTION',
            code: 'EXTRACT_ENTITIES(source=USER_INPUT)',
            status: 'pending',
            rosetta: 'Identificazione delle variabili di stato e dei vincoli del sistema.'
        },
        {
            id: 3,
            label: 'METRIC_MAPPING',
            code: 'MAP_TO_TENSOR(h_i, J_ij)',
            status: 'pending',
            rosetta: 'Conversione dei concetti in parametri fisici: Bias (Gravità) e Coupling (Correlazione).'
        },
        {
            id: 4,
            label: 'ANNEALING_INIT',
            code: 'INIT_SYSTEM(temp=3.0, steps=1000)',
            status: 'pending',
            rosetta: 'Iniezione di energia termica (Rumore) per avviare la ricerca della soluzione ottima.'
        },
        {
            id: 5,
            label: 'CONVERGENCE_CHECK',
            code: 'MINIMIZE_HAMILTONIAN(H)',
            status: 'pending',
            rosetta: 'Il sistema si raffredda e cristallizza nella configurazione a minima energia (Stato Stabile).'
        }
    ];

    return trace;
};
