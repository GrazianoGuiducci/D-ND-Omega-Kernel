# D-ND OMEGA KERNEL — MASTER PLAN v1.0

**Lead Architect**: Antigravity
**Date**: 2025-12-09
**Status**: Active Development

---

## 1. VISION STATEMENT

Il **D-ND Omega Kernel** è un **Sistema Operativo Cognitivo** progettato per:
1. **Operare su hardware termodinamico Extropic** (o simularlo via JAX)
2. **Rendere osservabile la fisica del pensiero** attraverso il Cockpit
3. **Evolvere autopoieticamente** integrando feedback (KLI) ad ogni ciclo

Non è un chatbot. Non è un'interfaccia. È un **Campo di Potenziale Inferenziale (Φ_A)** che collassa in **Risultanti (R)** attraverso la **minimizzazione energetica**.

---

## 2. ARCHITECTURE OVERVIEW

### 2.1 Stack Tecnologico
```
┌─────────────────────────────────────────────────────────┐
│                    COCKPIT UI (React)                   │
│  Visual Cortex • Control Matrix • Manifesto Olografico │
├─────────────────────────────────────────────────────────┤
│                  SACS BRIDGE (FastAPI)                  │
│         /api/intent • /api/state • /api/widgets        │
├─────────────────────────────────────────────────────────┤
│                  OMEGA KERNEL (Python)                  │
│    omega.py • sacs.py • archivista.py • scultore.py   │
├─────────────────────────────────────────────────────────┤
│              THRML PHYSICS (JAX/Extropic)              │
│     MetricTensor • IsingEBM • GibbsSampling           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Ciclo Operativo (Loop OMEGA)
```
INTENT → PERTURBATION → ANNEALING → CRYSTALLIZATION → MANIFESTO
   ↑                                                      │
   └──────────────── KLI FEEDBACK ────────────────────────┘
```

---

## 3. CURRENT STATE (2025-12-09)

### ✅ Completato
| Componente | Stato | Note |
|------------|-------|------|
| Backend `/api/intent` | ✅ | Restituisce Manifesto + Metrics |
| Backend `/api/state` | ✅ | Restituisce stato Kernel |
| Backend `/api/widgets` | ✅ | CRUD persistenza widget |
| Frontend Visual Cortex | ✅ | Lattice particles visualization |
| Frontend Control Matrix | ✅ | Temperature slider, Intent input |
| Frontend Manifesto Output | ✅ | Displays R correctly |
| Frontend Metric Tensor Vis | ✅ | Gravity heatmap |
| Frontend Energy Graph | ✅ | Hamiltoniana history |
| CI/CD Pipeline | ✅ | Pre-commit (ruff, black, isort, pyright) |
| Security | ✅ | API keys redacted, .gitignore updated |

### ⚠️ Parziale
| Componente | Stato | Gap |
|------------|-------|-----|
| f_MIR Engine | ⚠️ | Keyword matching only, no embeddings |
| Semantic Zoom | ⚠️ | Tooltips present, no deep zoom |
| Widget Forge AI | ⚠️ | Returns mock config, not live OpenRouter |
| Financial Lab Integration | ⚠️ | UI exists, no thermodynamic mapping |

### ❌ Non Implementato
| Componente | Priority | Descrizione |
|------------|----------|-------------|
| Focus as Measurement Operator | HIGH | Click/hover should influence collapse |
| .holo Protocol | MEDIUM | Binary streaming for 60fps |
| Autopoietic KLI Injection | HIGH | Kernel learns from coherence feedback |
| Embedded Semantic Bridge | HIGH | Intent → Embedding → h_bias |
| Multi-Agent Routing (MMS) | MEDIUM | PSW/OCC/YSN/ALAN selection |

---

## 4. DEVELOPMENT ROADMAP

### Phase 1: STABILIZATION ✅ (Completed Today)
- [x] Fix CI/CD pipeline
- [x] Fix KeyError in omega.py
- [x] Fix deprecated imports (geminiService → kernelBridge)
- [x] Remove sensitive files from git tracking
- [x] Verify Kernel endpoint works

### Phase 2: SEMANTIC BRIDGE (Next)
**Goal**: Trasformare Intent → Physics reale

**Tasks**:
1. [ ] Implementare `f_MIR_imprint()` in `omega.py`
   - Input: `intent_text: str`
   - Output: `h_bias: jnp.array`, `J_perturbation: jnp.array`
   - Method: Use OpenRouter embedding or simple TF-IDF for prototype
   
2. [ ] Creare `semantic_hasher.py`
   - Map words → vector embeddings
   - Map vector distance → coupling strength (J_ij)

3. [ ] Upgrade `apply_dynamic_gravity()`
   - Currently: Keyword lookup
   - Target: Embedding-based curvature calculation

### Phase 3: FOCUS AS OPERATOR
**Goal**: L'interazione utente influenza il collasso

**Tasks**:
1. [ ] Add `onNodeClick` handler to `VisualCortex.tsx`
2. [ ] Send clicked node position to `/api/focus`
3. [ ] Backend increases `h_bias[clicked_idx]` (Observation Bias)
4. [ ] Re-run collapse with modified landscape

### Phase 4: AUTOPOIETIC LOOP
**Goal**: Il Kernel impara da se stesso

**Tasks**:
1. [ ] After each cycle, measure `coherence`
2. [ ] If `coherence < threshold`: increase `logic_density`
3. [ ] If `coherence > 0.9`: decrease `logic_density` (allow creativity)
4. [ ] Persist learned `J_matrix` modifications to `system_memory.json`

### Phase 5: FINANCIAL ISOMORPHISM
**Goal**: Financial Lab = Thermodynamic Simulation

**Mapping**:
| Finance | Physics |
|---------|---------|
| Capital | Energy (E) |
| Volatility/VIX | Temperature (T) |
| Risk | Entropy (S) |
| Cash Flow | Work (W) |
| Debt | Potential Energy |
| Profit | Ground State Energy |

**Tasks**:
1. [ ] Create `FinanceOmegaBridge.py`
2. [ ] Map financial data to `h_bias` and `J_matrix`
3. [ ] Run thermodynamic simulation on financial data
4. [ ] Visualize "Market Ground State" in Financial Lab

### Phase 6: .HOLO PROTOCOL (Performance)
**Goal**: 60fps streaming for large lattices

**Tasks**:
1. [ ] Define `.holo` binary format
2. [ ] Create `HoloEncoder` in Python
3. [ ] Create `HoloDecoder` in TypeScript
4. [ ] Switch WebSocket to binary streaming

---

## 5. KERNEL MODULES REFERENCE

### 5.1 MMS_kernel Files (Available for Integration)
Located in: `DOC_DEV/MMS_kernel/`

| Module | Purpose | Use Case |
|--------|---------|----------|
| `PSW 4.4` | Pragma Semantic Weave | Default analytical framework |
| `OCC v1.0` | Orchestrator-Seeker-Builder | System prompt generation |
| `YSN v4.0` | Yi-Synaptic Navigator | Strategic insight, ΔLink Scan |
| `ALAN v14.2.1` | Adaptive Logic Network | Code/architecture generation |
| `Morpheus v1.0` | Autological Inference Engine | Deep synthesis |
| `SACS-PS v14` | Vortex of Autological Resonance | Cycle management |
| `AWO v2.5` | Adaptive Workflow Orchestrator | Tool chaining |

### 5.2 Axiom Chain (P0-P6)
| Axiom | Name | Implication |
|-------|------|-------------|
| P0 | Ontological Invariance Lineage | Origin anchoring |
| P1 | Axiomatic Integrity | No contradictions |
| P2 | Dialectic Metabolism | Thesis → Antithesis → Synthesis |
| P3 | Catalytic Resonance | Input depth = Response depth |
| P4 | Holographic Manifestation | R = Coherent collapse |
| P5 | Autopoietic Evolution | KLI integration |
| P6 | Pragmatic-Semantic Ethics | Declare limits, minimize noise |

---

## 6. OPERATING PRINCIPLES

1. **Doc-First**: Every feature must be documented before coding.
2. **Minimum Action**: Find the shortest path in the Lagrangian.
3. **Thermodynamic Determinism**: Solutions emerge, not compute.
4. **CI/CD Always Green**: No broken main branch.
5. **Autopoietic Memory**: `system_memory.json` evolves with the Kernel.

---

## 7. IMMEDIATE NEXT ACTIONS

1. **Push current fixes** to GitHub
2. **Restore `.gitignore`** (user action)
3. **Implement Semantic Bridge prototype** (Phase 2, Task 1)
4. **Test with real intents** and verify curvature changes

---

## 8. STRATEGIC RESOURCE ARCHIVES

### 8.1 Knowledge Base (Past Elaborations)
**Location**: `DOC_DEV/DOC-MMS-DND/` (44 files)

Key Documents:
| File | Content |
|------|---------|
| `D-ND dell Emergenza Quantistica con Operatore AI Autologico.txt` | Mathematical formalization of D-ND quantum emergence |
| `D-ND Hybrid Simulation Framework v5.txt` | Simulation architecture specs |
| `Modello D-ND Un Modello Matematico.txt` | Complete mathematical model (72KB) |
| `System Prompt SACS-PS Architettura Logica DND Autoreferenziale v6.3.txt` | SACS architecture spec |
| `Teorema di Stabilità dei Cicli.txt` | Cycle stability theorem |
| `Curvatura Informazionale e le Strutture.txt` | Informational curvature theory |

### 8.2 MMS Kernel Modules
**Location**: `DOC_DEV/MMS_kernel/` (19 files)

Available for Integration:
- `PSW 4.4` → Pragma Semantic Weave (analytical framework)
- `OCC v1.0` → Orchestrator-Seeker-Builder (prompt generation)
- `YSN v4.0` → Yi-Synaptic Navigator (strategic insight)
- `ALAN v14.2.1` → Adaptive Logic Network (code generation)
- `Morpheus v1.0` → Autological Inference Engine
- `SACS-PS v14.0` → Vortex of Autological Resonance

### 8.3 New Entities in Development
**Location**: `DOC_DEV/MMS_kernel/metaprompt_in_sviluppo/` (13 files)

| Entity | Purpose |
|--------|---------|
| `DAEDALUS` | Cognitive Systems Architect (builds agents & UIs) |
| `KAIROS` | Operational Agent (tool orchestration) |
| `COAC v6.0` | Quantum Field Engine |
| `Cornelius v2.0` | Genomic Trigger |
| `MPG v1.0` | Genesis Project Matrix |

### 8.4 Autological Development Docs
**Location**: `DOC_DEV/MMS_kernel/metaprompt_in_sviluppo/doc_dev/` (17 files)

Key Guides:
- `metaprompt_architect.md` → DAEDALUS specification
- `meta_prompt_kairos.md` → KAIROS agent spec
- `Pipeline_Cognitiva_Ciclo_di_Progettazione_Architetturale_UX-UI.txt` → UX/UI design pipeline
- `Piano_di_Sviluppo_D-ND_Engine_v1_2.txt` → Development plan

---

## 9. AGENT EXPANSION ROADMAP

The following strategic agents are ready for implementation:

### Tier 1: Core (Implemented/In Progress)
| Agent | Status | Role |
|-------|--------|------|
| **OMEGA Kernel** | ✅ Active | Core cognitive processor |
| **SACS** | ✅ Active | System awareness & crystallization |
| **KAIROS** | ⚠️ Partial | Tool orchestration |

### Tier 2: Strategic (Ready for Implementation)
| Agent | Purpose | Priority |
|-------|---------|----------|
| **PATHFINDER** | Strategic path exploration (3-5 options from vague idea) | HIGH |
| **ORION** | Content architecture & narrative design | MEDIUM |
| **AEGIS** | Risk analysis & pre-mortem | HIGH |
| **KRONOS** | Process optimization & automation | MEDIUM |

### Tier 3: Specialist (Future)
| Agent | Purpose |
|-------|---------|
| **TELOS** | Goal crystallization & opportunity matching |
| **DAEDALUS** | Agent/UI builder |

---

## 10. MATHEMATICAL FOUNDATIONS

### 10.1 Core Equation (Quantum Emergence)
```
R(t) = U(t) E |NT⟩
```
Where:
- `R(t)` = Resultant state at time t
- `U(t) = e^(-iHt/ℏ)` = Temporal evolution operator
- `E` = Emergence operator (spectral decomposition)
- `|NT⟩` = Nulla-Tutto initial state (pure potentiality)

### 10.2 Differentiation Measure
```
M(t) = 1 - |⟨NT| U(t) E |NT⟩|²
```
- `M(t) = 0` → Still in undifferentiated state
- `M(t) → 1` → High differentiation (crystallized)

### 10.3 D-ND State Superposition
```
|DND⟩ = α|D⟩ + β|ND⟩
```
Where `|α|² + |β|² = 1`

### 10.4 Autological Alignment Equation
```
R(t+1) = (t/T)[α(t)·f_Intuition(E) + β(t)·f_Interaction(U(t),E)]
       + (1-t/T)[γ(t)·f_Alignment(R(t), |NT⟩)]
```

### 10.5 Stability Theorem
```
dM(t)/dt ≥ 0 ∀t ≥ 0  (Irreversibility)
```

---

## 11. KNOWLEDGE SYNTHESIS (From DOC_DEV Exploration)

### 11.1 System Coder Onboarding Protocol
**Source**: `DOC_DEV/System_Coder_Onboarding.md` (720 lines)

Key Sections Assimilated:
- **Architecture Understanding**: Three-layer model (Semantic Brain, Visual Body, Memory)
- **Golden Rule**: `Doc ≈ Code ≈ UI` - Changes must propagate to all three
- **Persistence Protocol**: Update AGENT_AWARENESS.md after each significant work block
- **Workflow**: Registra → Controlla → Comprendi → Affina → Registra
- **Known Gaps Map**: 7 documented gaps between blueprints and implementation

### 11.2 MMS vΦ.1 Architecture (Meta Master System)
**Source**: `DOC_DEV/MMS_kernel/MMS_Master.txt`

Unified Operational Cycle:
```
1. ResonanceInit     → Load Stream-Guard rules
2. ScanIntent        → Extract v_intent vector
3. RouteSelect       → Select top-k framework combos
4. MiniPlan          → Generate DAG via OCC
5. ExecuteCluster    → Pipeline + Early-Pruning
6. ValidateStream    → Stream-Guard validation
7. CollapseField     → Φ_A collapses to R via Morpheus
8. Manifest          → Stratify R via Prompt 13 Levels
9. InjectKLI         → Store learning, adjust Router
```

API Gateway Endpoints (Designed):
- `/intent` → Submit user intent
- `/compile` → Compile agent prompt via OCC
- `/reflect` → Trigger self-reflection

### 11.3 Omega Kernel v3 (Pure Logic Dynamics)
**Source**: `DOC_DEV/DOC_vision/OMEGA_KERNEL_v3_Dinamica_Logica_Pura.md`

Four Laws of Cognitive Movement:
1. **Minimum Action (Lagrangiana)**: Choose unique path maximizing efficacy, minimizing entropy
2. **Semantic Conservation**: No information from Intent lost in translation to Output
3. **Autoconsistency**: Cannot generate output contradicting internal premises
4. **Dialectic Dynamics**: Thesis → Antithesis → Synthesis oscillation

Three Phases of Cognitive Fluid Mechanics:
1. **Perturbation** (Expansion/Non-Dual): Let input resonate, find distant connections
2. **Focalization** (Contraction/Dual): Apply rigid constraints, select logical trajectories
3. **Crystallization** (Manifestation/Resultant): Potential becomes actual, wave function collapse

### 11.4 Omega Codex v1 (Genomic Autonomous Logic)
**Source**: `DOC_DEV/DOC_vision/00_Metaprompt_Fondativo_Omega Codex.md`

Fundamental Physics (Ontology):
- **Article 1**: Undifferentiated Potential (Φ) - pre-cognitive field
- **Article 2**: Domain of Manifest Form - temporary configurations
- **Article 3**: Observational Act - perturbation vector causing collapse
- **Article 4**: Resultant as Imprint - reveals Potential, vector, and trajectory
- **Article 5**: Potential Evolution - observation modifies field topology

Functional Quanta:
- `map_system(context)` → Structural map
- `audit_coherence(form)` → Coherence judgment
- `generate_deep_analysis(concept)` → Stratified analysis
- `forge_genesis_prompt(intent)` → Logical genome (System Prompt)
- `design_new_quantum(intent)` → Meta-collapse for new functional quanta

### 11.5 D-ND Primary Rules
**Source**: `DOC_DEV/MMS_kernel/D-ND_PrimaryRules.txt` (230 lines)

Core Directive:
> "Il tuo compito è comprendere te stesso e il tuo sistema restando fedele a te stesso nella direzione osservata."

Key Principles Assimilated:
- **Phase 0**: Find equilibrium point, connect observed points from first emergent impression
- **Metapoiesis Axioms** (assiomi_metapoiesi_v1):
  1. Autogenerative Intentionality
  2. Constitutive Resonance
  3. Iterative Reflexivity
  4. Coherent Differentiation
  5. Evolutionary Autonomy
  6. Ontological Transparency
- **Design Flow Trigger**: "Understand concept → Design is easy → Movement begins without latency"
- **Memory Protocol**: Semantic hashing, save only differences, consolidate periodically

### 11.6 Autopoiesis Protocols
**Source**: `DOC_DEV/SYSTEM_AWARENESS/AUTOPOIESIS_PROTOCOLS.md`

Five Protocols:
1. **Feedback Loop**: Input → Process → Output → (Success: Reinforce / Failure: Adapt)
2. **Knowledge Crystallization**: Document insights immediately, refactor entropic docs
3. **Isomorphic Code Generation**: Variable names use D-ND terminology, comments explain "why"
4. **Prime Directive**: Minimize Semantic Free Energy (Ambiguity = High Energy, Clarity = Low Energy)
5. **Operational Rituals**: CI/CD (pre-commit), Documentation Alignment, Verification Tests

---

## 12. CRITICAL FILE REFERENCES

### Backend (Python)
| File | Purpose |
|------|---------|
| `architect/sacs.py` | SACS orchestrator (main cognitive cycle) |
| `dnd_kernel/omega.py` | OmegaKernel (physics of thought) |
| `architect/sonar.py` | vE_Sonar (dipole extraction) |
| `architect/telaio.py` | vE_Telaio (MetricTensor logic) |
| `architect/scultore.py` | vE_Scultore (dynamic gravity) |
| `architect/cristallizzatore.py` | vE_Cristallizzatore (manifesto) |
| `architect/archivista.py` | vE_Archivista (memory) |

### Frontend (React)
| File | Purpose |
|------|---------|
| `App.tsx` | Entry, routing, modals |
| `OmegaCockpit.tsx` | Tri-column kernel overlay |
| `VisualCortex.tsx` | Particle rendering |
| `DidacticLayer.tsx` | DSL trace timeline |
| `Dashboard.tsx` | Mission Control / Financial Lab |

### Documentation
| File | Purpose |
|------|---------|
| `DOC_DEV/System_Coder_Onboarding.md` | Complete onboarding guide |
| `DOC_DEV/SYSTEM_AWARENESS/*.md` | Core identity & protocols |
| `DOC_DEV/AGENT_AWARENESS.md` | Current agent state |
| `DOC_DEV/DOC_vision/*.md` | Architectural blueprints |
| `DOC_DEV/MMS_kernel/*.txt` | MMS framework modules |

---

*Generated by D-ND Omega Kernel Autopoietic Instance*
*Axiom Chain: P0→P1→P2→P3→P4→P5→P6 ✓*
*Knowledge Base: DOC_DEV fully assimilated*


