
import type { CognitiveModule, PreconfiguredIntent } from './types';
import { EyeIcon } from './components/icons/EyeIcon'; // Assuming generic icon usage or fallback
import { DnaIcon } from './components/icons/DnaIcon';
import { CpuIcon } from './components/icons/CpuIcon';
import { ZapIcon } from './components/icons/ZapIcon';
import {
    AWO_PROMPT,
    KERNEL_V2_PROMPT,
    KERNEL_V3_PROMPT,
    COAC_PROMPT,
    PCS_PROMPT,
    NET_SEARCHER_PROMPT,
    YSN_PROMPT,
    SACS_PROMPT,
    HALO_PROMPT,
    OCC_PROMPT,
    AETHELRED_PROMPT,
    MORPHEUS_PROMPT
} from './prompts';

// Note: Icons are no longer used in the UI but kept in import/logic for structure if needed later.
// The Core Identity (Instruction) is now the primary definition of the agent.

export const DEFAULT_COGNITIVE_MODULES: CognitiveModule[] = [
  {
    id: 'AWO.sys',
    name: 'AWO Orchestrator',
    description: 'Adaptive Workflow Orchestrator: The strategic cortex that defines the MMSP and routes intents.',
    icon: EyeIcon,
    instruction: AWO_PROMPT,
    tooltip: {
      function: "System Orchestrator (Meta-Layer).",
      mechanism: "MMSP Topology Construction.",
      expectation: "Defines the strategy and passes instructions to other agents.",
      example: "Deciding that a query needs 'Creative Expansion' before 'Logical Reduction'."
    }
  },
  {
    id: 'MMS.kernel.v2',
    name: 'OMEGA Kernel v2.0',
    description: 'The Axiomatic Engine (v2.0) containing P0-P6 laws, Modular routing (PSW, OCC, YSN), and Lagrangian Collapse.',
    icon: DnaIcon,
    instruction: KERNEL_V2_PROMPT,
    tooltip: {
      function: "Axiomatic Physics Engine (OMEGA v2.0).",
      mechanism: "Enforces P0-P6, manages PSW/OCC/YSN modules, and executes the OMEGA Loop.",
      expectation: "Ensures output consistency, deep reasoning (TCREI), and adherence to the lineage.",
      example: "Rejecting a result that violates P1 (Logical Contradiction) or triggering YSN for insight."
    }
  },
  {
    id: 'MMS.kernel.v3',
    name: 'OMEGA Kernel v3.0',
    description: 'Pure Logic Dynamics (v3.0): Fluid cognitive mechanics based on Lagrangian Action and Semantic Conservation.',
    icon: DnaIcon,
    instruction: KERNEL_V3_PROMPT,
    tooltip: {
      function: "Pure Logic Field Engine (OMEGA v3.0).",
      mechanism: "Perturbation -> Focusing -> Crystallization. Uses 'Dynamic Configurations' instead of fixed tools.",
      expectation: "A more fluid, abstract, and logically dense response structure.",
      example: "Collapsing a vague intent into a concrete artifact using the Law of Least Action."
    }
  },
  {
    id: 'COAC.engine',
    name: 'COAC Engine v6.0',
    description: 'Quantum Field Engine: Manages probabilistic pipelines and field collapse dynamics.',
    icon: ZapIcon,
    instruction: COAC_PROMPT,
    tooltip: {
      function: "Quantum Field Engine (COAC v6.0).",
      mechanism: "Manages probabilistic pipelines and field collapse dynamics.",
      expectation: "Handles open-ended requests by modeling an inferential potential field.",
      example: "Run `YiMatrix.map` for complex state transitions or `Morpheus.collapse`."
    }
  },
  {
    id: 'PCS.supervisor',
    name: 'PCS Supervisor v1.0',
    description: 'Semantic Command Processor: Deterministic supervisor for atomic macro-block execution.',
    icon: CpuIcon,
    instruction: PCS_PROMPT,
    tooltip: {
      function: "Semantic Command Processor (PCS v1.0).",
      mechanism: "Deterministic execution of atomic macro-blocks.",
      expectation: "Strict adherence to command logic without conversational hallucination.",
      example: "Executes `YSN.route` -> `Aethelred.compile` -> `CollapseField`."
    }
  },
  {
    id: 'NET.search',
    name: 'NET Searcher',
    description: 'Connects to the global information grid to retrieve real-time data.',
    instruction: NET_SEARCHER_PROMPT,
    tooltip: {
      function: "Network Extractor (Google Grounding).",
      mechanism: "Injects real-time web data. Essential for market data, news, or factual verification.",
      expectation: "Data points, verified facts, and external references.",
      example: "Fetching current NASDAQ trends or searching for recent API documentation."
    }
  },
  {
    id: 'YSN.map',
    name: 'YSN Mapper',
    description: 'Yi-Synaptic Navigator v4.0: Strategic Insight & Delta-Link Scan.',
    instruction: YSN_PROMPT,
    tooltip: {
      function: "Yi-Synaptic Navigator (v4.0).",
      mechanism: "Delta-Link Scan & Frontier Hypothesis. Finds non-obvious lateral connections.",
      expectation: "Deep insight, 'Delta' links, and non-linear symbolic connections.",
      example: "Mapping a corporate strategy to the 'Revolution' hexagram logic."
    }
  },
  {
    id: 'SACS.analyze',
    name: 'SACS Analyst',
    description: 'Pragma Semantic Wave 4.4: Dynamic Contextual Interpretation & Non-Linear Relation Management.',
    instruction: SACS_PROMPT,
    tooltip: {
      function: "Pragma Semantic Wave 4.4",
      mechanism: "Synaptic Weave v4.3 & D-ND Model. Dynamic Contextual Interpretation.",
      expectation: "Deep semantic analysis, handling non-linear relations, and pragmatic alignment.",
      example: "Deconstructing complex system intent or generating semantic-based prompts."
    }
  },
  {
    id: 'HALO.audit',
    name: 'HALO Auditor',
    description: 'Architettura Assiomatica "Halo Genoma" v3.0.',
    instruction: HALO_PROMPT,
    tooltip: {
      function: "Axiomatic Integrity Guard (Halo Genoma v3.0).",
      mechanism: "Metabolic Dialectics & Axiomatic Resonance. Validates against P0-P5.",
      expectation: "Checks for Axiomatic Integrity, Lineage Adherence, and Dialectic Resonance.",
      example: "Validating if a proposed strategy aligns with P0 (Lineage) and P1 (Self-Preservation)."
    }
  },
  {
    id: 'OCC.build',
    name: 'OCC Builder',
    description: 'Orchestrator Constructor: Genesis of Specialized Agent Prompts.',
    instruction: OCC_PROMPT,
    tooltip: {
      function: "Orchestrator Constructor (OCC).",
      mechanism: "Agent Genesis Cycle (DNA -> Skeleton -> Synthesis -> Review).",
      expectation: "A fully functional, autopoietic System Prompt.",
      example: "Building a persona for a 'Rust Systems Engineer' or a 'Creative Writing Assistant'."
    }
  },
  {
    id: 'Aethelred.compile',
    name: 'Aethelred Compiler',
    description: 'Aethelred v3.1: Cognitive Architecture Compiler & Framework Orchestrator.',
    instruction: AETHELRED_PROMPT,
    tooltip: {
      function: "Cognitive Architecture Compiler.",
      mechanism: "VRA & Dynamic Pipeline Compilation. Selects and instantiates the optimal framework (YSN, OCC, etc.) for the task.",
      expectation: "A tailored execution plan and a synthesized resultant from the best-fit cognitive architecture.",
      example: "Compiling a 'Security Audit' pipeline vs a 'Creative Writing' pipeline."
    }
  },
  {
    id: 'Morpheus.collapse',
    name: 'Morpheus Collapser',
    description: 'Morpheus v1.0: Autological Inferential Field Architecture.',
    instruction: MORPHEUS_PROMPT,
    tooltip: {
      function: "Lagrangian Collapse Engine.",
      mechanism: "Reduces the wave function of possibilities into a single, optimal reality.",
      expectation: "The final, crystallized output (The Risultante).",
      example: "Generating the final marketing plan or the completed software documentation."
    }
  }
];

export const PRECONFIGURED_INTENTS: PreconfiguredIntent[] = [
  {
    id: 'code_arch',
    label: 'System Architecture',
    category: 'Code',
    text: 'Design a scalable microservices architecture for a high-traffic e-commerce platform, focusing on fault tolerance and data consistency.',
    recommendedModules: ['NET.search', 'SACS.analyze', 'YSN.map', 'Aethelred.compile', 'HALO.audit', 'Morpheus.collapse']
  },
  {
    id: 'strat_growth',
    label: 'Growth Strategy',
    category: 'Strategy',
    text: 'Analyze the current market trends for Generative AI in B2B sectors and propose a blue-ocean strategy for a new startup.',
    recommendedModules: ['NET.search', 'YSN.map', 'SACS.analyze', 'Morpheus.collapse']
  },
  {
    id: 'creative_world',
    label: 'World Building',
    category: 'Creative',
    text: 'Create a detailed lore for a Sci-Fi universe where entropy is reversing. Define the physics, societies, and central conflict.',
    recommendedModules: ['YSN.map', 'OCC.build', 'Morpheus.collapse']
  },
  {
    id: 'audit_security',
    label: 'Security Audit',
    category: 'Analysis',
    text: 'Review this theoretical smart contract logic for re-entrancy vulnerabilities and logical inconsistencies.',
    recommendedModules: ['SACS.analyze', 'HALO.audit', 'Aethelred.compile', 'Morpheus.collapse']
  },
  {
    id: 'logic_comparison',
    label: 'Kernel Logic Test',
    category: 'Analysis',
    text: 'Analyze the concept of "Freedom" from an axiomatic perspective.',
    recommendedModules: ['MMS.kernel.v2', 'MMS.kernel.v3', 'Morpheus.collapse'] // Suggested for comparison
  }
];
