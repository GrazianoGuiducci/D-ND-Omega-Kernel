# Guida di Porting – Integrazione OpenRouter (Backend, Frontend, UI)

Questa guida riassume **come è implementato il sistema OpenRouter** in questa app (MMS / Cockpit) e come **portare lo stesso pattern** in un’altra applicazione (es. quella “con Gemini che aspetta”), anche **senza sistema utenti** completo.

Focus:

1. Gateway LLM unico via **OpenRouter** (backend)
2. **Gestione chiavi**: chiave di sistema + BYOK (Bring Your Own Key)
3. **Selettore modello** (chat + per‑nodo) e supporto a un **modale/catalogo modelli**
4. **Cost HUD / usage** (facoltativo, ma integrato nel design)
5. Come adattare il tutto a un’app **senza utenti**.

---

## 1. Architettura ad alto livello

### 1.1. Backend

- Tutte le chiamate LLM passano da **un solo servizio** `llm_inference` che usa la libreria `openai` puntata a `https://openrouter.ai/api/v1`.
- La scelta del modello è **solo una stringa** (`model_name`) passata dal resto del sistema.
- Le chiavi sono risolte con precedenza:
  1. **BYOK per‑request** (header `X-OpenRouter-Key`)
  2. **Pool server** (`OPENROUTER_API_KEYS=["sk-or-AAA", "sk-or-BBB"]`)
  3. **Chiave singola server** (`OPENROUTER_API_KEY=...`)
- Il servizio espone funzioni:
  - `invoke_llm_for_raw_json` → output JSON strutturato
  - `get_llm_text_response` → risposta testuale sincrona
  - `stream_llm_response` → streaming con eventi `message_chunk` / `final_response` / `usage`

### 1.2. Frontend

- Un client Axios centralizzato:
  - Inietta `Authorization: Bearer <jwt>` se presente
  - Inietta **BYOK lato client** da `localStorage.openrouter_api_key` come `X-OpenRouter-Key`
  - Imposta `Accept-Language` per IT/EN.
- Uno **Zustand store** (`interaction-store`) mantiene, per utente+dominio:
  - `selectedModel: string` (es. `"openai/gpt-5-mini"`)
  - `selectedEngine: 'v1' | 'v2' | 'v3'`
- La UI (header + Settings) usa:
  - Una lista statica `AVAILABLE_MODELS`
  - Un **registry dinamico** `or_header_models` in `localStorage` per il selettore chat (pinned models, metadati, prezzi).

---

## 2. Backend – Gateway OpenRouter

### 2.1. Dipendenze

```bash
pip install openai aiohttp
```

### 2.2. Configurazione `.env`

Esempio `.env` lato backend:

```env
# Chiavi OpenRouter
OPENROUTER_API_KEY=sk-or-AAA
OPENROUTER_API_KEYS=["sk-or-AAA","sk-or-BBB"]  # opzionale pool JSON

# Metadati app (facoltativo, per OpenRouter dashboard)
APP_URL=https://example.com
APP_TITLE=My AI App
```

> Nota: `OPENROUTER_API_KEYS` deve essere JSON valido (array di stringhe).

### 2.3. Precedenza chiavi (`_resolve_api_key`)

Schema usato in `backend/app/llm_inference.py`:

```python
from threading import Lock
from app.config import settings
from app.core.request_context import get_current_openrouter_key  # opzionale

_rr_index: int = 0
_rr_lock: Lock = Lock()

def _resolve_api_key() -> str:
    """
    Precedenza:
      1) Chiave BYOK per-request (header X-OpenRouter-Key)
      2) Lista server OPENROUTER_API_KEYS (round-robin)
      3) Chiave singola OPENROUTER_API_KEY
    """
    # 1) BYOK utente (dal context var)
    user_key = (get_current_openrouter_key() or "").strip()
    if user_key:
        return user_key

    # 2) Lista multipla server
    try:
        keys = getattr(settings, "OPENROUTER_API_KEYS", None)
        if keys and isinstance(keys, list):
            cleaned = [str(k).strip() for k in keys if str(k).strip()]
            if cleaned:
                global _rr_index
                with _rr_lock:
                    key = cleaned[_rr_index % len(cleaned)]
                    _rr_index += 1
                return key
    except Exception:
        pass  # fallback

    # 3) Fallback singola
    env_key = (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip()
    if not env_key:
        raise RuntimeError("OPENROUTER_API_KEY non configurata e nessuna BYOK fornita.")
    return env_key
```

Se **non hai un sistema utenti**, puoi semplificare eliminando `get_current_openrouter_key()` e guardando direttamente l’header sulla `Request` (vedi §2.5).

### 2.4. Client OpenAI compatibile OpenRouter

```python
from openai import OpenAI

def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_resolve_api_key(),
    )
```

#### 2.4.1. JSON strutturato

```python
import json
from app.schemas import InvokeRequestForJson

async def invoke_llm_for_raw_json(request: InvokeRequestForJson) -> dict | None:
    """
    Chiamata JSON-only via OpenRouter / OpenAI.
    """
    try:
        client = _get_client()
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

        response = client.chat.completions.create(
            model=request.model_name,                 # es. "openai/gpt-5-mini"
            messages=messages,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content:
            return json.loads(content)
    except Exception as e:
        print(f"[invoke_llm_for_raw_json] ERROR: {e}")
    return None
```

#### 2.4.2. Risposta testuale semplice

```python
async def get_llm_text_response(model_name: str, system_prompt: str, user_prompt: str) -> str:
    try:
        client = _get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[get_llm_text_response] ERROR: {e}")
        return f"ERRORE LLM: {e}"
```

#### 2.4.3. Streaming + usage (per Cost HUD)

La versione in MMS usa due percorsi:

1. **SSE raw verso OpenRouter** (via `aiohttp`) per ricevere `usage` inline
2. Fallback al client OpenAI streaming

Interfaccia logica (semplificata):

```python
from typing import AsyncGenerator, Dict, Any

async def stream_llm_response(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Yield:
      - {"event": "message_chunk", "data": {"chunk": str}}
      - {"event": "final_response", "data": {"response": str}}
      - {"event": "usage", "data": {"usage": dict, "model": str, "provider": str, "id": str}}
      - {"event": "error", "data": {"error": str}}
    """
    ...
```

Nella tua app esterna puoi:

- Implementare **solo** `message_chunk` + `final_response` se non ti serve il Cost HUD
- O replicare la stessa struttura `usage` per mostrare costi in UI.

### 2.5. BYOK per‑request (senza sistema utenti)

Se non hai utenti autenticati, puoi prendere la chiave direttamente dall’header della request.

Esempio FastAPI:

```python
from fastapi import Request

async def get_current_openrouter_key_from_request(request: Request) -> str | None:
    """
    Recupera la chiave BYOK dal client, se presente.
    """
    key = request.headers.get("X-OpenRouter-Key") or ""
    return key.strip() or None
```

Poi modifica `_resolve_api_key()` per usare questa funzione (o passa la chiave come argomento).

---

## 3. Endpoint diagnostico `/openrouter/status` (facoltativo ma utile)

L’app corrente espone `GET /openrouter/status` per mostrare nella UI:

- `key_source`: `"user" | "server" | "none"`
- `has_server_key`: bool
- `has_user_key`: bool
- `cache_age_s`, `cache_size`: metadati lato FE per il Model Catalog

Schema di massima:

```python
from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/openrouter/status")
async def openrouter_status(request: Request):
    user_key = (request.headers.get("X-OpenRouter-Key") or "").strip()
    has_user_key = bool(user_key)
    has_server_single = bool((getattr(settings, "OPENROUTER_API_KEY", "") or "").strip())
    has_server_pool = bool(getattr(settings, "OPENROUTER_API_KEYS", None))

    if has_user_key:
        key_source = "user"
    elif has_server_single or has_server_pool:
        key_source = "server"
    else:
        key_source = "none"

    return {
        "key_source": key_source,
        "has_server_key": bool(has_server_single or has_server_pool),
        "has_user_key": has_user_key,
        "cache_age_s": 0,
        "cache_size": 0,
    }
```

Il frontend può chiamare `/openrouter/status` per mostrare badge tipo:

- “Stai usando la chiave personale OpenRouter”
- “Stai usando la chiave di sistema”
- “Nessuna chiave disponibile”

---

## 4. Registry dei Nodi di Sistema (per-nodo Model Selector)

In MMS, i diversi “nodi cognitivi” (selector, condenser, pinner, ecc.) leggono il **modello dalle configurazioni di sistema**, non hard‑coded.

File JSON di base (es. `backend/config/system_node_config.json`):

```json
[
  {
    "task_name": "relevance_selector_agent",
    "description": "Seleziona i documenti più pertinenti.",
    "model_id": "openai/gpt-5-mini"
  },
  {
    "task_name": "excess_condenser_agent",
    "description": "Condensa le informazioni in eccesso.",
    "model_id": "openai/gpt-5-mini"
  },
  {
    "task_name": "pinner_agent",
    "description": "Estrae Atomi di Conoscenza dalle conversazioni.",
    "model_id": "openai/gpt-5-mini"
  },
  {
    "task_name": "metapinner_agent",
    "description": "Consolida Atomi in documenti/metaprompt.",
    "model_id": "openai/gpt-5"
  },
  {
    "task_name": "domain_genesis_agent",
    "description": "Scrive il Metaprompt fondativo di un nuovo Dominio.",
    "model_id": "openai/gpt-5"
  },
  {
    "task_name": "chroma_agent",
    "description": "Genera palette cromatiche per la UI.",
    "model_id": "openai/gpt-5-mini"
  }
]
```

Pattern:

1. Ogni agente (es. `domain_genesis_agent`) chiede al registry `get_system_nodes()` → trova `model_id` per `task_name`.
2. Passa `model_id` al servizio `llm_inference`.

Per replicare nella tua app:

- Definisci un **elenco di nodi** (anche solo per type di richiesta o componente)
- Offri un endpoint REST:

  - `GET /system/nodes` → lista nodi
  - `PATCH /system/nodes/{task_name}` → aggiorna `model_id` (da UI)

- In UI, crea un **modale “Configurazione Nodi”** con:

  - Una card per nodo (nome, descrizione)
  - Un `Select` che usa la lista modelli OpenRouter (vedi §5)

---

## 5. Frontend – Axios + BYOK + OpenRouter Status

### 5.1. Client Axios

Pattern usato in `frontend/src/api/api.ts` (semplificato):

```ts
import axios, { AxiosInstance } from 'axios';

let axiosInstance: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api/v1',
});

let interceptorsConfigured = false;

function setupInterceptors() {
  if (interceptorsConfigured) return;

  axiosInstance.interceptors.request.use((config) => {
    try {
      // 1) Authorization se hai login/JWT
      // const token = authStore.getState().accessToken;
      // if (token) { config.headers.Authorization = `Bearer ${token}`; }

      // 2) BYOK OpenRouter da localStorage
      const or = typeof window !== 'undefined'
        ? window.localStorage.getItem('openrouter_api_key')
        : null;
      if (or && or.trim().length > 0) {
        (config.headers as any)['X-OpenRouter-Key'] = or.trim();
      }

      // 3) BYOK Tavily (se usi ricerca web)
      const tv = typeof window !== 'undefined'
        ? window.localStorage.getItem('tavily_api_key')
        : null;
      if (tv && tv.trim().length > 0) {
        (config.headers as any)['X-Tavily-Key'] = tv.trim();
      }

      // 4) Lingua UI
      const lang = typeof window !== 'undefined'
        ? (localStorage.getItem('app_lang') || '')
        : '';
      (config.headers as any)['Accept-Language'] = (lang === 'en' ? 'en' : 'it');
    } catch {}
    return config;
  });

  interceptorsConfigured = true;
}

const getApiInstance = (): AxiosInstance => {
  if (!interceptorsConfigured) setupInterceptors();
  return axiosInstance;
};
```

Se non hai autenticazione, puoi **rimuovere la parte di JWT** e tenere solo BYOK + lingua.

### 5.2. UI per BYOK (lato client, senza utenti)

Minimo indispensabile:

```ts
// Esempio hook/azione per salvare la chiave personale
function saveOpenRouterKeyToLocalStorage(key: string) {
  if (typeof window === 'undefined') return;
  localStorage.setItem('openrouter_api_key', key.trim());
}
```

Poi in un tab “Account / Chiavi API”:

- Campo `OpenRouter API Key`
- Bottone “Salva” → chiama `saveOpenRouterKeyToLocalStorage`
- Nessun salvataggio su backend = nessun rischio per chiavi.

---

## 6. Selettore modello (Chat) e Catalogo

### 6.1. Lista base di modelli (fallback)

Estratto da `interaction-store.ts`:

```ts
export const AVAILABLE_MODELS = [
  { id: 'google/gemini-2.5-flash',       name: 'Gemini 2.5 Flash' },
  { id: 'google/gemini-2.5-pro',         name: 'Gemini 2.5 Pro' },
  { id: 'openai/gpt-5-mini',             name: 'Openai GPT-5 Mini' },
  { id: 'openai/gpt-5',                  name: 'Openai GPT-5' },
  { id: 'qwen/qwen3-235b-a22b-thinking-2507', name: 'Qwn3-235B Thinking' },
];

export const AVAILABLE_CHAT_MODELS = AVAILABLE_MODELS; // o con filtro
```

Per il porting:

- Usa un array `AVAILABLE_MODELS` con gli ID OpenRouter che vuoi supportare.
- L’`id` dev’essere **esattamente** il valore passato a `model` in `chat.completions.create()`.

### 6.2. Registry dinamico + modale Catalogo

MMS usa un registry per il selettore chat (header):

```ts
export type HeaderChatModel = {
  id: string;
  name: string;
  provider?: string;
  ctx?: number;           // context window
  supported?: string[];   // es. ["json", "vision"]
  outs?: string[];        // tipi di output supportati
  zdr?: boolean;          // Zero Data Retention
  priceIn?: string | null;
  priceOut?: string | null;
  note?: string;
};

export const HEADER_MODELS_STORAGE_KEY = 'or_header_models';

export function getHeaderChatModels(): HeaderChatModel[] {
  try {
    const raw = localStorage.getItem(HEADER_MODELS_STORAGE_KEY);
    if (!raw) return AVAILABLE_CHAT_MODELS;
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return AVAILABLE_CHAT_MODELS;
    const cleaned = arr
      .map((x: any) => ({
        id: String(x?.id || ''),
        name: String(x?.name || x?.id || ''),
        provider: x?.provider ?? undefined,
        ctx: typeof x?.ctx === 'number' ? x.ctx : undefined,
        supported: Array.isArray(x?.supported) ? x.supported : undefined,
        outs: Array.isArray(x?.outs) ? x.outs : undefined,
        zdr: typeof x?.zdr === 'boolean' ? x.zdr : undefined,
        priceIn: x?.priceIn ?? undefined,
        priceOut: x?.priceOut ?? undefined,
        note: typeof x?.note === 'string' ? x.note : undefined,
      }))
      .filter((x: any) => x.id);
    return cleaned.length ? cleaned : AVAILABLE_CHAT_MODELS;
  } catch {
    return AVAILABLE_CHAT_MODELS;
  }
}

export function setHeaderChatModels(models: HeaderChatModel[]) {
  try {
    const cleaned = (Array.isArray(models) ? models : []).map((x) => ({
      id: String(x?.id || ''),
      name: String(x?.name || x?.id || ''),
      provider: x?.provider ?? undefined,
      ctx: typeof x?.ctx === 'number' ? x.ctx : undefined,
      supported: Array.isArray(x?.supported) ? x.supported : undefined,
      outs: Array.isArray(x?.outs) ? x.outs : undefined,
      zdr: typeof x?.zdr === 'boolean' ? x.zdr : undefined,
      priceIn: x?.priceIn ?? undefined,
      priceOut: x?.priceOut ?? undefined,
      note: typeof x?.note === 'string' ? x.note : undefined,
    })).filter((x) => x.id);

    localStorage.setItem(HEADER_MODELS_STORAGE_KEY, JSON.stringify(cleaned));
    window.dispatchEvent(new CustomEvent('or:header-models-updated', { detail: cleaned }));
  } catch {}
}
```

Pattern di porting:

1. **Modale Catalogo Modelli**:
   - Chiama OpenRouter `/models` o `/openrouter/status` + cache.
   - Permette di **pinnare** alcuni modelli per il selettore Chat.
   - Salva la scelta in `localStorage['or_header_models']` via `setHeaderChatModels`.

2. **Selettore in Header**:
   - Legge `getHeaderChatModels()`.
   - Mostra `Select` + info (ctx, ZDR, prezzo).
   - Aggiorna `selectedModel` in uno store globale (vedi sotto).

---

## 7. Store di interazione (Chat) – `selectedModel` + invocazione

### 7.1. Stato per dominio (Zustand)

Pattern di base:

```ts
export type EngineVersion = 'v1' | 'v2' | 'v3';

interface ScopedInteractionState {
  currentInteraction: any | null;
  botIsTyping: boolean;
  selectedModel: string;
  selectedEngine: EngineVersion;
  maxFullFiles?: number | null;
  forceAllSelected?: boolean | null;
}

const defaultScopedState: ScopedInteractionState = {
  currentInteraction: null,
  botIsTyping: false,
  selectedModel: getHeaderChatModels()[0]?.id || AVAILABLE_MODELS[0].id,
  selectedEngine: 'v3',
  maxFullFiles: null,
  forceAllSelected: null,
};
```

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
