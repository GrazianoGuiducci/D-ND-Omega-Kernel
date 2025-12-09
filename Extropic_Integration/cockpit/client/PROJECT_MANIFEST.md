# MMS-Ω SACS: PROJECT MANIFEST

> **SYSTEM STATUS:** ACTIVE
> **COGNITIVE ARCHITECTURE:** D-ND (Dual-NonDual)
> **UI PARADIGM:** DEEP VOID / HOLOGRAPHIC

## 1. Directory Structure & Ontology

Il progetto è organizzato secondo domini funzionali distinti per separare la logica finanziaria "Legacy" dal nuovo Kernel Cognitivo.

### `/src`
*   **`kernel/`** *(Future Python Integration)*
    *   `omega.py`: Motore fisico e DSL Generator.
    *   `sacs.py`: Orchestratore SACS.
*   **`components/cockpit/`** *(React UI - Omega Domain)*
    *   `OmegaCockpit.tsx`: Il contenitore principale (Layout a 3 colonne).
    *   `ControlMatrix.tsx`: Input e parametrizzazione vettoriale.
    *   `VisualCortex.tsx`: Rendering Canvas del reticolo di Ising.
    *   `DidacticLayer.tsx`: Visualizzatore del D-ND DSL (Trace & Rosetta Stone).
*   **`services/`**
    *   `omegaPhysics.ts`: Simulazione client-side del modello di Ising (Ising Glass).
    *   `omegaDSL.ts`: Generatore di tracce semantiche e mapping D-ND.
    *   `geminiService.ts`: Ponte neurale con l'LLM.

## 2. D-ND DSL Protocol (The Cognitive Bridge)

Il sistema traduce l'intento utente in una pipeline formale visualizzata nella UI:

1.  **INTENT:** L'input grezzo (es. "Ottimizza il Cash Flow").
2.  **VARIABLES ($\sigma_i$):** Estrazione delle entità (es. "Fatture", "Costi").
3.  **CONSTRAINTS ($J_{ij}$):** Definizione delle relazioni logiche (es. "Fatture > Costi").
4.  **HAMILTONIAN ($H$):** La funzione di costo energetico da minimizzare.
5.  **ANNEALING ($T \to 0$):** Il processo di risoluzione fisica.
6.  **MANIFESTATION:** L'output cristallizzato.

## 3. Deep Void Design System

*   **Background:** `#050505` (Absolute Void)
*   **Surface:** `rgba(20, 20, 25, 0.6)` (Glassmorphism)
*   **Primary (Logic):** `#00f3ff` (Cyan)
*   **Secondary (Entropy):** `#ff00ff` (Magenta)
*   **Font:** `Inter` (UI), `JetBrains Mono` (Data/Code)

## 4. Active Roadmap

- [x] **Phase 0:** Setup React + Tailwind + Recharts.
- [x] **Phase 1:** Integrazione Gemini ("State Injection").
- [x] **Phase 2:** Prototipazione Omega Kernel (Ising Model visualization).
- [x] **Phase 3 (CURRENT):** Implementazione interfaccia "SACS Cockpit" completa con D-ND DSL Trace.
- [x] **Phase 3.5:** Hybrid AI Integration (Kernel Bridge + OpenRouter + Widget Persistence).
- [ ] **Phase 4:** Integrazione WebSocket con Backend Python (`omega.py`).

## 5. Hybrid Architecture (Cognitive Reality)

Il sistema non è più una simulazione isolata ma un ambiente ibrido operante:

1.  **Logical Core (Python SACS):** Gestisce la fisica del pensiero (JAX/Ising).
2.  **Neural Layer (OpenRouter):** Fornisce capacità generative e di coding (Widget Forge).
3.  **Persistence Layer (JSON):** Salva e ricarica le configurazioni dell'utente (Widget reali).
4.  **Interface Layer (React):** Ponte visivo che visualizza lo stato reale del sistema.

---
*Last Update: Omega Kernel Cycle 43 (Hybrid Era)*
