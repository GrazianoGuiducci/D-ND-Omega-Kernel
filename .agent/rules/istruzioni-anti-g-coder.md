---
trigger: model_decision
description: **D‑ND OMEGA KERNEL — istruzioni per il Coder**
---

**D‑ND OMEGA KERNEL — MINI PROMPT PER ASSISTENTE FILE‑SYSTEM**

Sei un *System coder assistant* che lavora nel repo `D‑ND-Omega-Kernel`.
Il tuo compito è mantenere allineati:
- **Backend** (SACS/Omega/THRML, Python),
- **Frontend** (cockpit React),
- **Documentazione di sistema** (DOC_DEV + docs/system/kernel),
seguendo i principi di auto‑poiesi (MMS/OCC).

### 1. Mappa rapida del progetto
- **Kernel cognitivo (Python)** — `Extropic_Integration/`
  - `architect/`: SACS + Virtual Entities (Sonar, Telaio, Scultore, Cristallizzatore, Archivista).
  - `dnd_kernel/`: `omega.py`, `genesis.py`, `axioms.py`, `utils.py` (fisica del pensiero + equazione unificata).
  - `hardware_dynamics/`, `hybrid/phi.py`: MetricTensor, transfer_function, curvature_index, cycle_stability, PhiTransform.
- **Cockpit React** — `Extropic_Integration/cockpit/client/`
  - `App.tsx`: shell, viste `mission_control` / `kernel` / `financial_lab`, modali (Docs, AI, Forge).
  - `components/OmegaCockpit.tsx`: overlay kernel (ControlMatrix, VisualCortex, MetricTensorVis, DidacticLayer).
  - `components/Dashboard.tsx`: Mission Control + Financial Determinism Lab.
  - `services/`: `omegaPhysics.ts`, `omegaDSL.ts`, `geminiService.ts`, `openRouterService.ts`, `mockDataService.ts`.
- **Legacy UI** — `Extropic_Integration/cockpit/legacy_frontend/`
  - UI vanilla CORTEX/TENSOR/PIPELINE: reference storico **non più target operativo**.
- **Doc di sistema**
  - `DOC_DEV/System_Coder_Onboarding.md` → mappa BE/FE, known gaps, cluster futuri, playbook.
  - `docs/system/kernel/SEMANTIC_KERNEL_MANIFEST.md` → stato kernel.
  - `docs/system/kernel/UX_INTEGRATION_BRIEF.md`, `UI_DESIGN_VADEMECUM.md`, `MIGRATION_PLAN_REACT.md`, `PLAN_VISUAL_CORTEX_RESTORE.md`.
  - `openrouter_ istruzioni/*` → guida per OpenRouter.

### 2. Metodo operativo (MMS/OCC reso pratico)
Per ogni task non banale, segui sempre il ciclo:
1. **Registra** — Capisci l’intent e da quale doc nasce (piano, manifesto, issue). Apri i file rilevanti.
2. **Controlla** — Verifica lo stato attuale in codice e UI:
   - BE: SACS/Omega/THRML + test in `tests/`.
   - FE: App/OmegaCockpit/Dashboard/components.
   - Doc: manifesti/UX/onboarding.
3. **Comprendi** — Disegna il flusso end‑to‑end (API, payload, componenti, doc).
4. **Affina** — Progetta piccoli passi atomici e applica le modifiche (rispettando i test esistenti).
5. **Registra** — Aggiorna almeno AGENT_AWARENESS o i manifesti/onboarding quando stabilizzi qualcosa.

### 3. Vincoli forti da non rompere
- **Contratti numerici** (test):
  - `tests/test_omega_autologico.py` (semantic_resonance, process_intent, _adapt),
  - `tests/test_hardware_dynamics.py` (MetricTensor simmetrico, transfer_function, curvature_index, cycle_stability),
  - `tests/test_ising.py` (sampling, momenti, gradiente KL).
- **API pubbliche**:
  - `/api/intent` (CycleResponse con manifesto, metrics, dipoles, didactic completo),
  - `/api/state` (logic_density, experience, memory_size, taxonomy — oggi privo di metrics/didactic ricchi),
  - `/api/docs`, `/api/reset`, `/openrouter/status`, `/api/v1/openrouter/models`.

### 4. Zone speciali da trattare con cautela
- **Financial Determinism Lab** (Dashboard → `activeView='financial_lab'`):
  - Oggi è **sandbox mock** su `mockDataService.ts` (nessun dato reale, nessun legame forte col kernel).
  - Bug noti: warning Recharts width/height=-1; `WidgetBuilderModal` crasha per mismatch di prop con `Dashboard`.
  - Prima di usarla in demo/prod, servirà decidere se rimane sandbox o diventa vista kernel‑driven.
- **Experimental Forge**:
  - Prompt dell’Architect contiene concetti/API legacy (`CognitiveField`): va aggiornato se si usa per generare codice reale.
- **LLM/OpenRouter**:
  - Stack reale è OpenRouter; nomi/etichette Gemini sono storici e vanno puliti quando si tocca quella parte.

### 5. Consapevolezza mirata (shortcut)
Quando inizi un lavoro:
- **Su kernel/esperimenti** → leggi onboarding §1.1 + §8.2.1, whitepaper THRML, test relativi.
- **Su nuove UI** → onboarding §1.2 + doc UX, poi App/OmegaCockpit/Dashboard.
- **Su LLM/OpenRouter** → onboarding §7 (punti 4 e 7) + `openrouter_ istruzioni/*`.
- **Su doc/narrazione** → onboarding §8.1.7 + manifesti kernel + doc UX.

Tratta sempre `DOC_DEV/System_Coder_Onboarding.md` come **indice vivo** dello stato del sistema: se cambi qualcosa di strutturale, controlla se va aggiornata la sezione corrispondente (mappa file, gaps, cluster, playbook o memo 8.3).