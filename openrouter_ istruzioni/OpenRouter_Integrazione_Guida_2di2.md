Seconda parte
---

Azioni principali:

```ts
interface IInteractionActions {
  setSelectedEngine: (engine: EngineVersion) => void;
  setSelectedModel: (modelId: string) => void;
  // ...
}
```

Uso nel componente Chat:

```ts
const { selectedModel, selectedEngine } = useInteractionState();
const { setSelectedModel } = useInteractionActions();

// In header:
<Select
  value={selectedModel}
  onChange={(e) => setSelectedModel(e.target.value)}
>
  {getHeaderChatModels().map((m) => (
    <MenuItem key={m.id} value={m.id}>{m.name}</MenuItem>
  ))}
</Select>
```

### 7.2. Chiamata all’engine

Funzioni FE → BE (es. `invokeLLM_V2`):

```ts
import type { Message } from './api-contracts';

export async function invokeLLM_V2(
  prompt: string,
  domainId: number | null,
  selectedFiles: string[],
  model: string,
  pinnedMessages: Message[],
  chatHistory: Message[] = [],
  maxFullFiles?: number,
  forceIncludeAllSelected?: boolean,
  forceAllUnbounded?: boolean,
  signal?: AbortSignal,
): Promise<any> {
  if (!domainId) throw new Error("Serve un domainId valido.");
  const api = getApiInstance();
  const requestPayload = {
    prompt,
    domain_id: domainId,
    selected_files: selectedFiles,
    model_name: model,     // <--- stringa OpenRouter
    task_type: 'chat',
    engine_version: 'v2',
    // extra opzionali...
    pinned_messages: pinnedMessages,
    chat_history: chatHistory,
  };
  const response = await api.post('invoke', requestPayload, { signal });
  return response.data;
}
```

In `handleSendMessage`:

```ts
const { selectedModel, selectedEngine } = useInteractionState();

const model = selectedModel;
if (selectedEngine === 'v2') {
  await invokeLLM_V2(prompt, domainId, selectedFiles, model, pinned, history, ...);
} else if (selectedEngine === 'v1') {
  await invokeLLM_V1(prompt, domainId, selectedFiles, model, pinned, history, ...);
} else {
  await invokeLLM_V3(prompt, domainId, selectedFiles, model, pinned, history, ...);
}
```

---

## 8. Cost HUD (uso + costi) – schema

### 8.1. Backend

Lo stream LLM (`stream_llm_response`) emette eventi `usage` quando disponibili:

```json
{
  "event": "usage",
  "data": {
    "usage": {
      "total_tokens": 1234,
      "prompt_tokens": 456,
      "completion_tokens": 778,
      "cache_read": 0,
      "cache_write": 0
    },
    "model": "openai/gpt-5-mini",
    "provider": "OpenAI",
    "id": "cmpl-..."
  }
}
```

### 8.2. Frontend

1. Il consumer SSE (es. in `useSendMessage` / `HomePage_v2`) ascolta:

   - `message_chunk` → aggiorna testo
   - `final_response` → chiude lo stream
   - `usage` → aggiorna uno store `cost-store`:

   ```ts
   type Usage = { prompt: number; completion: number; total: number; model: string; };
   ```

2. Il **Model Catalog** (modale) o `or_header_models` contiene `priceIn` / `priceOut` (per 1M token o 1k, come preferisci).

3. Il Cost HUD calcola:

   ```ts
   const costIn  = usage.prompt    / 1_000_000 * parseFloat(priceIn || '0');
   const costOut = usage.completion / 1_000_000 * parseFloat(priceOut || '0');
   const lastCost = costIn + costOut;
   ```

4. UI:
   - Barra sopra il composer con:
     - `IN / OUT / TOT` token
     - `$lastCost` per risposta
     - `Σ session` per conversazione (somma delle risposte).

Se non ti serve il Cost HUD, puoi **ignorare gli eventi `usage`** e implementare solo `message_chunk` + `final_response`.

---

## 9. Adattamento a un’app senza sistema utenti

Nell’altra app (“dove c’è Gemini che aspetta”):

1. Puoi usare **esattamente lo stesso backend** OpenRouter, ma:

   - Eliminare dipendenze da user_id, JWT, RBAC
   - Usare solo:
     - `OPENROUTER_API_KEY` / `OPENROUTER_API_KEYS`
     - BYOK *opzionale* da `X-OpenRouter-Key` se vuoi permettere override dalla UI.

2. Frontend:

   - Niente auth-store: puoi rimuovere `Authorization` e tenere solo BYOK/lingua.
   - Il selettore chat e il catalogo modelli funzionano nello stesso modo (stato locale + localStorage).

3. Nodi AI:

   - Se non hai ancora un sistema di “nodi”, puoi cominciare **solo con il selettore chat** (un drop‑down globale).
   - In un secondo momento puoi introdurre un file JSON / tabella DB con nodi e modelli, e un modale di configurazione “per‑nodo” simile alla `SettingsModal` esistente.

---

## 10. Riassunto operativo per il porting

1. **Backend**
   - Aggiungi `llm_inference_openrouter.py` con `_resolve_api_key`, `_get_client`, funzioni `invoke_llm_for_raw_json` / `get_llm_text_response` / `stream_llm_response`.
   - Configura `.env` con `OPENROUTER_API_KEY` / `OPENROUTER_API_KEYS`.
   - (Opz.) Aggiungi `GET /openrouter/status` per la UI.

2. **Frontend**
   - Crea un client Axios con interceptor che inietta `X-OpenRouter-Key` da `localStorage.openrouter_api_key`.
   - Crea una UI minima per BYOK (campo input + `localStorage`).
   - Implementa `AVAILABLE_MODELS` e, se vuoi, un registry `or_header_models` per il catalogo.

3. **Chat**
   - Aggiungi uno store con `selectedModel` e `selectedEngine`.
   - Modifica la chiamata all’engine per passare `model_name: selectedModel`.

4. **Per‑nodo (facoltativo)**
   - Definisci un JSON/tabella con nodi e `model_id`.
   - Crea un modale “Configurazione Nodi” per cambiare il modello dei vari task.
   - Fai sì che i tuoi agenti leggano il modello dal registry invece di hard‑codare stringhe.

5. **Cost HUD (facoltativo)**
   - In streaming, inoltra gli eventi `usage` al frontend.
   - Mantieni uno store per `usage` + prezzi (dal catalogo) e mostra un HUD sopra il composer.

Questa guida è pensata come **base di porting**: puoi copiare gli snippet così come sono o usarli come specifica tecnica per l’altra app (anche se usa ancora “Gemini”). Il punto centrale è che l’LLM diventa **solo un modello OpenRouter selezionabile** via stringa, non codificato nel codice dell’agente.

---

## 11. Chiave di sistema, badge di stato e modal BYOK quando il credito è esaurito

Questa sezione descrive **esattamente il flusso** che hai in mente:

- L’admin (tu) mette una **chiave di sistema** nel `.env`.
- Gli utenti possono lavorare con la chiave di sistema finché ci sono crediti.
- Quando la chiave di sistema è **esaurita** (o va in errore), la UI:
  - Mostra uno stato di errore/badge.
  - Apre un **modal BYOK** che chiede di inserire una chiave OpenRouter personale, con guida e link:
    - https://openrouter.ai/
    - https://openrouter.ai/docs/faq

### 11.1. Chiave di sistema nel `.env` (default globale)

Lato backend:

```env
# Chiave di sistema (default globale)
OPENROUTER_API_KEY=sk-or-AAA

# (Opzionale) Pool di chiavi server per bilanciare il carico
OPENROUTER_API_KEYS=["sk-or-AAA","sk-or-BBB"]
```

Questa chiave è il **fallback** per tutti gli utenti che:

- non hanno ancora inserito una propria chiave nel browser (BYOK),
- oppure per l’app “senza utenti” dove non esiste login.

La precedenza rimane:

1. `X-OpenRouter-Key` (BYOK da client)
2. `OPENROUTER_API_KEYS` (pool server)
3. `OPENROUTER_API_KEY` (singola server)

### 11.2. Badge di stato chiave in UI (server vs utente)

Usa l’endpoint `/openrouter/status` descritto sopra per mostrare un **badge** nella UI (es. in header o nel tab “Account / Chiavi API”):

```ts
type OpenRouterStatus = {
  key_source: 'server' | 'user' | 'none';
  has_server_key: boolean;
  has_user_key: boolean;
  cache_age_s: number;
  cache_size: number;
};

async function fetchOpenRouterStatus(): Promise<OpenRouterStatus> {
  const api = getApiInstance();
  const res = await api.get('/openrouter/status');
  return res.data;
}
```

Esempio di mapping in UI:

- `key_source === 'user'` → badge verde: **“Chiave personale attiva (BYOK)”**
- `key_source === 'server'` → badge blu: **“Chiave di sistema (admin)”**
- `key_source === 'none'` → badge rosso: **“Nessuna chiave: configura BYOK”**

Questo già ti permette di:

- Capire, visivamente, da dove viene la chiave usata.
- Invitare l’utente alla scheda BYOK quando `key_source === 'server'` e vuoi spingerlo a portarsi i costi a casa sua.

### 11.3. Rilevare esaurimento crediti/limiti della chiave di sistema

Quando l’API OpenRouter va in errore per **credito esaurito / limiti**, di solito:

- Restituisce un codice HTTP (es. `402` o `429`)
- Oppure un messaggio di errore nel JSON (`"insufficient funds"`, `"payment required"`, ecc.)

La strategia consigliata:

1. **Cattura l’eccezione** in `llm_inference.py` (sia in chiamate sincrone che streaming).
2. Se riconosci che è un errore di **chiave di sistema esaurita**, mappa ad un errore controllato lato backend, per esempio:

   ```python
   from fastapi import HTTPException

   def _map_openrouter_error(e: Exception) -> None:
       text = str(e).lower()
       if 'insufficient funds' in text or 'payment required' in text:
           # Codice simbolico per la UI
           raise HTTPException(
               status_code=503,
               detail="OPENROUTER_SYSTEM_KEY_EXHAUSTED"
           )
       # altri casi → rilancia o logga
   ```

3. Nelle funzioni `invoke_llm_for_raw_json`, `get_llm_text_response`, `stream_llm_response`, fai:

   ```python
   except Exception as e:
       _map_openrouter_error(e)
       # se non è stato mappato, logga e rilancia o ritorna errore generico
   ```

Per lo **streaming**, oltre a sollevare l’eccezione, puoi anche emettere un evento SSE dedicato:

```python
yield {
    "event": "error",
    "data": {
        "code": "OPENROUTER_SYSTEM_KEY_EXHAUSTED",
        "message": "La chiave di sistema OpenRouter è esaurita.",
    },
}
return
```

### 11.4. Apertura del modal BYOK sul frontend

#### 11.4.1. Struttura del modal BYOK

Crea un piccolo store/UI per un modal, ad es.:

```ts
type ByokModalReason = 'manual' | 'systemKeyExhausted';

type ByokModalState = {
  isOpen: boolean;
  reason: ByokModalReason | null;
};

const useByokModalStore = create<{
  state: ByokModalState;
  open: (reason: ByokModalReason) => void;
  close: () => void;
}>((set) => ({
  state: { isOpen: false, reason: null },
  open: (reason) => set({ state: { isOpen: true, reason } }),
  close: () => set({ state: { isOpen: false, reason: null } }),
}));
```

Il modal mostra:

- Titolo diverso a seconda del `reason`:
  - `systemKeyExhausted` → “La chiave di sistema è esaurita”
  - `manual` → “Imposta la tua chiave OpenRouter (BYOK)”
- Contenuto:

  ```md
  1. Vai su [openrouter.ai](https://openrouter.ai/) e registrati con la tua email.
  2. Dopo il login, apri la sezione **API Keys** e crea una nuova chiave.
  3. Copia la chiave (inizia con `sk-or-...`) e incollala qui sotto.
  4. Premi “Salva chiave”: da ora in poi le chiamate LLM useranno la tua chiave.
  ```

- Campi:
  - Input `OpenRouter API Key`
  - Bottone “Salva chiave” che scrive in `localStorage.openrouter_api_key`.
- Link:

  ```tsx
  <a href="https://openrouter.ai/" target="_blank" rel="noreferrer">Registrati su OpenRouter</a>
  <a href="https://openrouter.ai/docs/faq" target="_blank" rel="noreferrer">FAQ OpenRouter</a>
  ```

#### 11.4.2. Quando aprire il modal (errore system key)

Nel codice che gestisce le risposte API (non 401 di auth, ma errori LLM), controlla il `detail`:

```ts
import { AxiosError } from 'axios';

function handleLlmError(error: unknown) {
  const maybeAxios = error as AxiosError<any>;
  const detail = maybeAxios?.response?.data?.detail;

  if (detail === 'OPENROUTER_SYSTEM_KEY_EXHAUSTED') {
    useByokModalStore.getState().open('systemKeyExhausted');
    return;
  }

  // altri errori → toast generico, log, ecc.
}
```

Per lo **streaming SSE**, quando consumi lo stream:

```ts
if (eventName === 'error' && data?.code === 'OPENROUTER_SYSTEM_KEY_EXHAUSTED') {
  useByokModalStore.getState().open('systemKeyExhausted');
  // ferma lo stream / aggiorna stato chat
}
```

In questo modo:

- Se la chiave di sistema funziona → tutto va liscio, gli utenti non vedono nulla.
- Quando si esaurisce:
  - La chiamata fallisce con `detail: "OPENROUTER_SYSTEM_KEY_EXHAUSTED"` o evento SSE con `code`.
  - La UI apre **automaticamente** il modal BYOK con guida e link.

### 11.5. Badge + modal → esperienza completa

Flusso finale:

1. Admin configura `OPENROUTER_API_KEY` in `.env` → backend pronto.
2. UI mostra badge **“Chiave di sistema (admin)”** in header / tab Account.
3. Quando la chiave di sistema è esaurita:
   - Backend mappa l’errore ad un codice simbolico (`OPENROUTER_SYSTEM_KEY_EXHAUSTED`).
   - Frontend cattura il codice ed apre il modal BYOK con istruzioni:
     - Registrati → https://openrouter.ai/
     - Guida → https://openrouter.ai/docs/faq
   - L’utente incolla la propria chiave nel form → salvata in `localStorage`.
4. Dalla chiamata successiva:
   - L’interceptor Axios invia `X-OpenRouter-Key`.
   - `_resolve_api_key()` dà precedenza alla chiave utente.
   - Il badge cambia in **“Chiave personale attiva (BYOK)”**.

In questo modo hai:

- Una **chiave di sistema** che rende l’app subito utilizzabile.
- Un flusso automatico che **invita l’utente ad aggiungere la propria chiave** quando la chiave di sistema non è più sufficiente.
---
openrouter apikey per D-ND OMEGA KERNEL
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
