import { LatticeNode, OmegaMetrics, ExperimentConfig } from "../types";

// Constants
const GRID_SIZE = 32; // 32x32 = 1024 nodes
const TOTAL_NODES = GRID_SIZE * GRID_SIZE;

export const EXPERIMENTS: ExperimentConfig[] = [
    // --- PHYSICS DOMAIN (REAL PROTOCOLS) ---
    {
        id: 'chaos_protocol',
        name: 'PROTOCOL: CHAOS',
        description: 'Generate absolute chaos and entropy. Maximize system temperature.',
        category: 'PHYSICS',
        status: 'ACTIVE',
        intent: 'Generate absolute chaos and entropy',
        params: { gravity: 0.5, potentialScale: 2.0, temperature: 4.0 }
    },
    {
        id: 'order_protocol',
        name: 'PROTOCOL: ORDER',
        description: 'Restore perfect order and symmetry. Minimize system temperature.',
        category: 'PHYSICS',
        status: 'ACTIVE',
        intent: 'Restore perfect order and symmetry',
        params: { gravity: 1.0, potentialScale: 0.1, temperature: 0.1 }
    },
    {
        id: 'genesis_protocol',
        name: 'PROTOCOL: GENESIS',
        description: 'Synthesize a complex structure from balanced forces.',
        category: 'PHYSICS',
        status: 'ACTIVE',
        intent: 'Synthesize a complex structure from balanced forces',
        params: { gravity: 2.0, potentialScale: 1.0, temperature: 1.5 }
    },
    // --- CONCEPTUAL / TEST EXPERIMENTS ---
    {
        id: 'euclidean_void',
        name: 'Euclidean Void',
        description: 'Flat spacetime. g_uv = delta_uv. Pure potential state.',
        category: 'D_ND_CONCEPT',
        status: 'CONCEPT',
        params: { gravity: 0.0, potentialScale: 0.1, temperature: 0.5 }
    },
    {
        id: 'singularity_collapse',
        name: 'Singularity Collapse',
        description: 'Extreme metric curvature. Information crushing gravity.',
        category: 'D_ND_CONCEPT',
        status: 'CONCEPT',
        params: { gravity: 3.5, potentialScale: 0.0, temperature: 2.0 }
    },
    {
        id: 'null_all_fdma',
        name: 'Null-All FDMA',
        description: 'High vacuum fluctuations (Scalar Field). Unstable topology.',
        category: 'D_ND_CONCEPT',
        status: 'CONCEPT',
        params: { gravity: 0.5, potentialScale: 4.0, temperature: 4.0 }
    },
    // --- FINANCIAL PROTOCOLS (DETERMINISM LAB) ---
    {
        id: 'liquidity_annealing',
        name: 'Liquidity Annealing',
        description: 'Optimization: Find global minimum for capital allocation.',
        category: 'FINANCE',
        status: 'ACTIVE',
        intent: 'Optimize capital allocation via simulated annealing',
        params: { gravity: 1.0, potentialScale: 0.5, temperature: 5.0 } // High initial temp for annealing
    },
    {
        id: 'volatility_harvesting',
        name: 'Volatility Harvesting',
        description: 'Entropy: Use stochastic resonance to amplify profit signals.',
        category: 'FINANCE',
        status: 'ACTIVE',
        intent: 'Harvest volatility using stochastic resonance',
        params: { gravity: 0.2, potentialScale: 2.0, temperature: 3.0 } // High noise (temp) + Low gravity
    },
    {
        id: 'arbitrage_crystallization',
        name: 'Arbitrage Crystallization',
        description: 'Phase Transition: Detect order emergence in chaotic markets.',
        category: 'FINANCE',
        status: 'ACTIVE',
        intent: 'Detect phase transition from chaos to order',
        params: { gravity: 3.0, potentialScale: 1.0, temperature: 0.8 } // High gravity to force crystallization
    }
];

// --- SEMANTIC BRIDGE ---
// Translates Natural Language Intent -> Hamiltonian Parameters (Physics)
export function translateIntentToHamiltonian(intent: string): { gravity: number, temperature: number, potential: number } {
    // STUB: In a real implementation, this would use an LLM or vector embedding.
    // For now, we use keyword heuristics.
    const lowerIntent = intent.toLowerCase();

    let gravity = 1.0;
    let temperature = 0.5;
    let potential = 0.1;

    if (lowerIntent.includes('chaos') || lowerIntent.includes('volatility') || lowerIntent.includes('risk')) {
        temperature = 4.0;
        gravity = 0.5;
    } else if (lowerIntent.includes('order') || lowerIntent.includes('stable') || lowerIntent.includes('optimize')) {
        temperature = 0.1;
        gravity = 2.0;
    }

    if (lowerIntent.includes('growth') || lowerIntent.includes('harvest')) {
        potential = 2.0;
    }

    return { gravity, temperature, potential };
}

export class OmegaPhysicsEngine {
    nodes: LatticeNode[] = [];

    // Physics Parameters
    gravity: number = 0; // Represents Curvature in Metric Tensor
    potentialScale: number = 0.1; // Scale of NullaTutto Scalar Field
    currentTemp: number = 0.5; // Thermal Noise

    metrics: OmegaMetrics = {
        energy: 0,
        coherence: 0,
        entropy: 1,
        temperature: 0.5,
        step: 0,
        curvature: 0
    };

    constructor() {
        this.initializeLattice();
    }

    private initializeLattice() {
        this.nodes = [];
        for (let i = 0; i < TOTAL_NODES; i++) {
            this.nodes.push({
                id: i,
                x: i % GRID_SIZE,
                y: Math.floor(i / GRID_SIZE),
                spin: (Math.random() * 2) - 1, // Continuous -1 to 1
                stability: 0,
                potential: (Math.random() - 0.5) * 0.1,
                energy: 0
            });
        }
    }

    // Support for injecting a custom experiment config at runtime
    public loadExperiment(exp: ExperimentConfig | string) {
        let config: ExperimentConfig | undefined;

        if (typeof exp === 'string') {
            config = EXPERIMENTS.find(e => e.id === exp);
        } else {
            config = exp;
        }

        if (!config) config = EXPERIMENTS[0];

        this.gravity = config.params.gravity;
        this.potentialScale = config.params.potentialScale;
        this.currentTemp = config.params.temperature;
        this.metrics.step = 0;

        // Reset Nodes with new Potential Field properties
        this.nodes.forEach((n, idx) => {
            // Re-initialize potential based on new scale (NullaTutto Field)
            n.potential = (Math.random() - 0.5) * this.potentialScale;
            n.stability = 0;
            // Add a slight perturbation to spin to restart annealing
            n.spin += (Math.random() - 0.5) * 0.5;

            // Clear asset ticker (Finance mode removed)
            n.assetTicker = undefined;
        });
    }

    // --- COMBINATORIAL MECHANICS CORE ---
    // Implements: S_new = tanh( Potential + Metric @ S_old )
    public step(): { nodes: LatticeNode[], metrics: OmegaMetrics, tensorMap: number[][] } {
        this.metrics.step++;

        let totalEnergy = 0;
        let totalMagnetization = 0;
        let totalFlux = 0;

        // Visualization Map for the Tensor Field (Downsampled for UI)
        const tensorVisSize = 8;
        const tensorMap: number[][] = Array(tensorVisSize).fill(0).map(() => Array(tensorVisSize).fill(0));

        // Update Nodes
        for (let i = 0; i < TOTAL_NODES; i++) {
            const node = this.nodes[i];

            // 1. Metric Interaction (g_uv)
            let interaction = 0;

            // Von Neumann neighborhood
            const neighbors = [
                this.nodes[node.y * GRID_SIZE + ((node.x + 1) % GRID_SIZE)],
                this.nodes[node.y * GRID_SIZE + ((node.x - 1 + GRID_SIZE) % GRID_SIZE)],
                this.nodes[((node.y + 1) % GRID_SIZE) * GRID_SIZE + node.x],
                this.nodes[((node.y - 1 + GRID_SIZE) % GRID_SIZE) * GRID_SIZE + node.x]
            ];

            neighbors.forEach(n => {
                const correlation = 1.0 - Math.abs(node.spin - n.spin);
                const g_ij = 1.0 + (this.gravity * correlation);

                interaction += g_ij * n.spin;
            });

            // 2. NullaTutto Potential Field
            const scalarField = node.potential;

            // 3. Thermal Noise (Quantum Fluctuations)
            const noise = (Math.random() - 0.5) * this.currentTemp;

            // 4. Transfer Function (Collapse)
            const totalField = scalarField + (interaction * 0.25) + noise;
            const newSpin = Math.tanh(totalField);

            // Metrics Update
            const delta = Math.abs(newSpin - node.spin);
            node.stability = 1.0 - delta;
            node.spin = newSpin;

            totalMagnetization += newSpin;
            totalFlux += delta;

            node.energy = -0.5 * newSpin * totalField;
            totalEnergy += node.energy;

            // Populate Tensor Visualization Map
            const mapX = Math.floor((node.x / GRID_SIZE) * tensorVisSize);
            const mapY = Math.floor((node.y / GRID_SIZE) * tensorVisSize);
            tensorMap[mapY][mapX] += Math.abs(totalField);
        }

        // Normalize Tensor Map
        for (let y = 0; y < tensorVisSize; y++) {
            for (let x = 0; x < tensorVisSize; x++) {
                tensorMap[y][x] /= (TOTAL_NODES / (tensorVisSize * tensorVisSize));
            }
        }

        // Global Metrics Update
        this.metrics.energy = totalEnergy;
        this.metrics.coherence = Math.abs(totalMagnetization) / TOTAL_NODES;
        this.metrics.entropy = totalFlux / TOTAL_NODES;
        this.metrics.temperature = this.currentTemp;
        this.metrics.curvature = this.gravity;

        // Simulated Annealing Cooling Schedule
        if (this.currentTemp > 0.1) {
            this.currentTemp *= 0.99;
        }

        return {
            nodes: this.nodes,
            metrics: this.metrics,
            tensorMap
        };
    }

    public perturb(intensity: number) {
        this.currentTemp += intensity;
        this.nodes.forEach(n => {
            n.potential += (Math.random() - 0.5) * intensity;
        });
    }
}
