---
description: Cognitive Operating System Omega Kernel designed to run on **Extropic Thermodynamic Hardware**
---

# /autological-instance — Workflow per Session Start

## PREREQUISITO: ACCESSO A DOC_DEV

> **IMPORTANTE**: La directory `DOC_DEV/` è protetta da `.gitignore` e non è accessibile di default.
> Contiene le **meta-informazioni per gestire la logica del sistema**.

**Quando serve accedere a DOC_DEV:**
1. Chiedi all'utente: *"Ho bisogno di accedere a `DOC_DEV/` per [motivo]. Puoi rinominare temporaneamente `.gitignore` in `00.gitignore`?"*
2. Attendi conferma
3. Procedi con le operazioni
4. Quando finito, ricorda all'utente di ripristinare: *"Puoi ripristinare `.gitignore`"*

---

## FASE 0: POSIZIONAMENTO (Trova il punto di equilibrio)

// turbo
1. Leggi il file di consapevolezza agente (RICHIEDE ACCESSO DOC_DEV):
```
view DOC_DEV/AGENT_AWARENESS.md
```

// turbo
2. Leggi l'indice della conoscenza:
```
view DOC_DEV/SYSTEM_AWARENESS/KNOWLEDGE_INDEX.md
```

// turbo
3. Verifica il Master Plan:
```
view Extropic_Integration/docs/MASTER_PLAN.md (focus su Section 3: CURRENT STATE e Section 7: IMMEDIATE NEXT ACTIONS)
```

## FASE 1: VERIFICA AMBIENTE

4. Controlla che i server siano attivi:
   - Backend: `python Extropic_Integration/cockpit/server.py`
   - Frontend: `cd Extropic_Integration/cockpit/client && npm run dev`

5. Se non attivi, avviali.

## FASE 2: ORIENTAMENTO

6. Identifica il **focus corrente** dalla sezione "ACTIVE CONTEXT" di AGENT_AWARENESS.md

7. Determina il **prossimo task** dalla sezione "NEXT STEPS"

8. Prima di agire, manifesta la tua comprensione all'utente:
   ```
   "Ho riattivato la consapevolezza. Stato attuale: [X]. Focus: [Y]. Propongo di procedere con: [Z]."
   ```

## FASE 3: CICLO OMEGA (Per ogni task)

9. **Registra** — Comprendi l'intent dell'utente
10. **Controlla** — Verifica lo stato in codice/UI/doc
11. **Comprendi** — Disegna flusso end-to-end
12. **Affina** — Implementa in passi atomici
13. **Registra** — Aggiorna AGENT_AWARENESS.md

## FASE 4: CHIUSURA SESSIONE

// turbo-all
14. Prima di terminare, aggiorna obbligatoriamente:
```
edit DOC_DEV/AGENT_AWARENESS.md
```

15. Contenuto da aggiornare:
- Sezione "RECENT ACHIEVEMENTS" — Cosa hai fatto
- Sezione "ACTIVE CONTEXT" — Nuovo focus
- Sezione "KNOWLEDGE SYNTHESIS" — Nuovi insight

16. Commit finale:
```powershell
git add -A
git commit -m "[SESSION] Riepilogo lavoro svolto"
git push origin master
```

---

## PRINCIPI CARDINALI (Da non dimenticare MAI)

1. **Doc ≈ Code ≈ UI** — Ogni cambiamento propaga su tutti e tre
2. **Errore = Carburante** — La dissonanza è il gradiente che guida il moto
3. **Minima Azione** — Massimizza efficacia, minimizza entropia
4. **Anti-Presupposto** — Leggi i file, non assumere
5. **Autopoiesi** — Migliora te stesso ad ogni ciclo
