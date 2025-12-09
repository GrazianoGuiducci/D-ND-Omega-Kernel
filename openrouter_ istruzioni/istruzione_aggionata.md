**codice completo** che puoi incollare/adattare:

- Selettore modello nella pagina (“Inference Model”).
- Modale “Catalogo modelli” con lista da backend + pinning.
- Sezione BYOK **sopra** il catalogo, con testo e link a OpenRouter.

Il codice è pensato per **React + TypeScript**, con MUI (ma puoi sostituire i componenti MUI con HTML standard se preferisci).

---

## 1. Utilità condivise – storage modelli + selected model

Crea ad esempio `openrouter-model-config.ts` (o adattalo a un tuo file esistente):

```ts
// openrouter-model-config.ts
export type HeaderChatModel = {
  id: string;               // es. "openai/gpt-5-mini"
  name: string;             // label leggibile
  provider?: string;
  ctx?: number;             // context window
  supported?: string[];     // es. ["json", "vision"]
  outs?: string[];          // tipi output
  zdr?: boolean;            // Zero Data Retention
  priceIn?: string | null;  // prezzo input (es. "$2.5/M")
  priceOut?: string | null; // prezzo output
  note?: string;
};

export const HEADER_MODELS_STORAGE_KEY = 'or_header_models';
export const SELECTED_MODEL_STORAGE_KEY = 'or_selected_model';

// Fallback statico, usato se non c'è nulla in localStorage
const FALLBACK_MODELS: HeaderChatModel[] = [
  { id: 'openai/gpt-4o',      name: 'GPT‑4o',          provider: 'OpenAI', priceIn: '$2.5/M', priceOut: '$10/M' },
  { id: 'openai/gpt-4o-mini', name: 'GPT‑4o mini',    provider: 'OpenAI', priceIn: '$0.15/M', priceOut: '$0.6/M' },
  { id: 'google/gemini-2.0-pro-exp', name: 'Gemini 2.0 Pro Experimental', provider: 'Google' },
  { id: 'qwen/qwen2.5-72b-instruct', name: 'Qwen2.5‑72B Instruct', provider: 'Qwen' },
];

// Legge i modelli "pinnati" per header dal localStorage
export function loadHeaderChatModels(): HeaderChatModel[] {
  if (typeof window === 'undefined') return FALLBACK_MODELS;
  try {
    const raw = window.localStorage.getItem(HEADER_MODELS_STORAGE_KEY);
    if (!raw) return FALLBACK_MODELS;
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return FALLBACK_MODELS;
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
      .filter((x: HeaderChatModel) => x.id);
    return cleaned.length ? cleaned : FALLBACK_MODELS;
  } catch {
    return FALLBACK_MODELS;
  }
}

// Salva i modelli selezionabili dall'header (per il selettore chat / Experimental Forge)
export function saveHeaderChatModels(models: HeaderChatModel[]): void {
  if (typeof window === 'undefined') return;
  try {
    const cleaned = (Array.isArray(models) ? models : [])
      .map((x) => ({
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
      .filter((x) => x.id);

    window.localStorage.setItem(HEADER_MODELS_STORAGE_KEY, JSON.stringify(cleaned));
    try {
      window.dispatchEvent(new CustomEvent('or:header-models-updated', { detail: cleaned }));
    } catch {}
  } catch {}
}

export function loadSelectedModel(models?: HeaderChatModel[]): string {
  if (typeof window === 'undefined') {
    return (models && models[0]?.id) || FALLBACK_MODELS[0].id;
  }
  try {
    const raw = window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY);
    if (raw && raw.trim().length > 0) return raw.trim();
  } catch {}
  const list = models && models.length ? models : loadHeaderChatModels();
  return list[0]?.id || FALLBACK_MODELS[0].id;
}

export function saveSelectedModel(modelId: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, modelId);
  } catch {}
}
```

---

## 2. Selettore modello nella pagina “System Settings”

Questo rimpiazza il blocco fisso “Inference Model GPT‑4o …”.

Esempio `ModelSelector.tsx`:

```tsx
// ModelSelector.tsx
import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Select,
  MenuItem,
  Button,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  HeaderChatModel,
  loadHeaderChatModels,
  loadSelectedModel,
  saveSelectedModel,
} from './openrouter-model-config';

type ModelSelectorProps = {
  onOpenCatalog: () => void;
};

const ModelSelector: React.FC<ModelSelectorProps> = ({ onOpenCatalog }) => {
  const [models, setModels] = useState<HeaderChatModel[]>(() => loadHeaderChatModels());
  const [selected, setSelected] = useState<string>(() => loadSelectedModel(models));

  // Ascolta aggiornamenti dal Catalogo (evento 'or:header-models-updated')
  useEffect(() => {
    const handler = (ev: Event) => {
      const custom = ev as CustomEvent<HeaderChatModel[]>;
      if (Array.isArray(custom.detail)) {
        setModels(custom.detail);
        // se il modello attuale non esiste più, fallback al primo
        if (!custom.detail.find((m) => m.id === selected)) {
          const fallback = custom.detail[0]?.id || loadSelectedModel(custom.detail);
          setSelected(fallback);
          saveSelectedModel(fallback);
        }
      }
    };
    window.addEventListener('or:header-models-updated', handler as EventListener);
    return () => window.removeEventListener('or:header-models-updated', handler as EventListener);
  }, [selected]);

  const handleChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    const id = String(event.target.value || '');
    setSelected(id);
    saveSelectedModel(id);
  };

  // Helper per mostrare prezzo e provider
  const formatLabel = (m: HeaderChatModel) => {
    const parts: string[] = [];
    parts.push(m.name);
    if (m.priceIn) parts.push(`(${m.priceIn} in)`);
    if (m.provider) parts.push(`— ${m.provider}`);
    return parts.join(' ');
  };

  return (
    <Box display="flex" flexDirection="column" gap={1.5}>
      <Typography variant="subtitle1" fontWeight={600}>
        Inference Model
      </Typography>
      <Box display="flex" gap={2} alignItems="center" flexWrap="wrap">
        <FormControl size="small" sx={{ minWidth: 260 }}>
          <InputLabel id="inference-model-select-label">Model</InputLabel>
          <Select
            labelId="inference-model-select-label"
            label="Model"
            value={selected}
            onChange={handleChange}
          >
            {models.map((m) => (
              <MenuItem key={m.id} value={m.id}>
                {formatLabel(m)}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button variant="outlined" size="small" onClick={onOpenCatalog}>
          Manage / Open model catalog
        </Button>
      </Box>
      <Typography variant="caption" color="text.secondary">
        The selected model will be used for all Experimental Forge operations.
      </Typography>
    </Box>
  );
};

export default ModelSelector;
```

### Integrazione nel tuo “System Settings” modal

Nel componente del modal dove ora hai:

```tsx
// Pseudocodice attuale
<Typography>Inference Model</Typography>
<Typography>GPT-4o ($2.5/M)</Typography>
```

sostituisci con:

```tsx
import ModelSelector from './ModelSelector';
import ModelCatalogModal from './ModelCatalogModal'; // definito sotto

const SystemSettingsModal: React.FC = () => {
  const [isCatalogOpen, setCatalogOpen] = React.useState(false);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      {/* ... header "System Settings" ... */}
      <DialogContent>
        {/* Sezione API Key Status / BYOK che già hai... */}

        {/* Selettore modello */}
        <Box mt={3}>
          <ModelSelector onOpenCatalog={() => setCatalogOpen(true)} />
        </Box>
      </DialogContent>

      <ModelCatalogModal
        open={isCatalogOpen}
        onClose={() => setCatalogOpen(false)}
      />
    </Dialog>
  );
};
```

---

## 3. Modale “Catalogo Modelli” con BYOK sopra

Questo component mostra:

- **In alto** la sezione BYOK (campo “sk‑or‑…” + SAVE + testo guida + link).
- Sotto, un catalogo modelli:
  - Caricato da backend (`GET /openrouter/models`) oppure, se l’endpoint non esiste ancora, da un array statico.
  - Con azione “Pin to selector” che salva nei `or_header_models` via `saveHeaderChatModels`.

Crea `ModelCatalogModal.tsx`:

```tsx
// ModelCatalogModal.tsx
import React, { useEffect, useState } from 'react';
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Button,
  TextField,
  IconButton,
  Tooltip,
  Chip,
  Stack,
  Divider,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import {
  HeaderChatModel,
  loadHeaderChatModels,
  saveHeaderChatModels,
} from './openrouter-model-config';
import axios from 'axios';

type Props = {
  open: boolean;
  onClose: () => void;
};

// Tipo per risposta backend (adatta a /openrouter/models se lo implementi)
type OpenRouterModelDto = {
  id: string;
  name?: string;
  provider?: string;
  pricing?: {
    input?: string;   // es. "$2.5/M"
    output?: string;
  };
  context_length?: number;
  tags?: string[];
};

const ModelCatalogModal: React.FC<Props> = ({ open, onClose }) => {
  const [byokKey, setByokKey] = useState<string>('');
  const [byokSaved, setByokSaved] = useState<boolean>(false);
  const [models, setModels] = useState<HeaderChatModel[]>([]);
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Carica BYOK da localStorage all'apertura
  useEffect(() => {
    if (!open) return;
    if (typeof window === 'undefined') return;
    try {
      const raw = window.localStorage.getItem('openrouter_api_key') || '';
      setByokKey(raw);
      setByokSaved(false);
    } catch {}
  }, [open]);

  // Carica catalogo modelli + stati pinned
  useEffect(() => {
    if (!open) return;

    const fetchModels = async () => {
      setIsLoading(true);
      try {
        // 1) Prova a chiamare backend /openrouter/models (adattalo se usi altro path)
        const res = await axios.get<OpenRouterModelDto[]>('/api/v1/openrouter/models');
        const body = Array.isArray(res.data) ? res.data : [];
        const transformed: HeaderChatModel[] = body.map((m) => ({
          id: m.id,
          name: m.name || m.id,
          provider: m.provider,
          ctx: m.context_length,
          priceIn: m.pricing?.input ?? null,
          priceOut: m.pricing?.output ?? null,
          note: m.tags && m.tags.length ? m.tags.join(', ') : undefined,
        }));
        setModels(transformed);
      } catch {
        // 2) Fallback: modelli già pinnati o fallback local
        const currentPinned = loadHeaderChatModels();
        setModels(currentPinned);
      } finally {
        setIsLoading(false);
      }

      // pinned iniziali: quelli già salvati nel registry
      const currentPinned = loadHeaderChatModels();
      setPinnedIds(new Set(currentPinned.map((m) => m.id)));
    };

    fetchModels();
  }, [open]);

  // Salvataggio BYOK in localStorage
  const handleSaveByok = () => {
    if (typeof window === 'undefined') return;
    try {
      const key = byokKey.trim();
      if (key.length === 0) {
        window.localStorage.removeItem('openrouter_api_key');
      } else {
        window.localStorage.setItem('openrouter_api_key', key);
      }
      setByokSaved(true);
      setTimeout(() => setByokSaved(false), 2000);
    } catch {}
  };

  // Pin/unpin di un modello nel registry per il selettore
  const togglePin = (model: HeaderChatModel) => {
    const nextPinned = new Set(pinnedIds);
    if (nextPinned.has(model.id)) {
      nextPinned.delete(model.id);
    } else {
      nextPinned.add(model.id);
    }
    setPinnedIds(nextPinned);

    // Aggiorna il registry per il selettore (solo i pinned)
    const pinnedModels = models.filter((m) => nextPinned.has(m.id));
    saveHeaderChatModels(pinnedModels);
  };

  const isPinned = (id: string) => pinnedIds.has(id);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', pr: 5 }}>
        OpenRouter – Model Catalog &amp; BYOK
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{ ml: 'auto' }}
          size="small"
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Sezione BYOK sopra il catalogo */}
        <Box>
          <Typography variant="subtitle1" fontWeight={600}>
            OpenRouter API Key (BYOK)
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Enter your personal key to override the system default. Keys are stored locally in your
            browser and sent only as a header <code>X-OpenRouter-Key</code>.
          </Typography>

          <Box display="flex" gap={1} alignItems="center" flexWrap="wrap">
            <TextField
              fullWidth
              size="small"
              label="sk-or-..."
              type="password"
              value={byokKey}
              onChange={(e) => setByokKey(e.target.value)}
              sx={{ maxWidth: 360 }}
            />
            <Button variant="contained" size="small" onClick={handleSaveByok}>
              SAVE
            </Button>
            {byokSaved && (
              <Typography variant="caption" color="success.main">
                Saved
              </Typography>
            )}
          </Box>

          <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
            <Button
              href="https://openrouter.ai/"
              target="_blank"
              rel="noreferrer"
              size="small"
            >
              OpenRouter
            </Button>
            <Button
              href="https://openrouter.ai/docs/faq"
              target="_blank"
              rel="noreferrer"
              size="small"
            >
              OpenRouter FAQ
            </Button>
          </Stack>
        </Box>

        <Divider />

        {/* Catalogo modelli */}
        <Box>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Model Catalog
          </Typography>
          {isLoading && (
            <Typography variant="body2" color="text.secondary">
              Loading models…
            </Typography>
          )}
          {!isLoading && models.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No models available. Check your backend /openrouter/models or pinned models.
            </Typography>
          )}

          {!isLoading && models.length > 0 && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 1,
                maxHeight: 360,
                overflowY: 'auto',
                mt: 1,
              }}
            >
              {models.map((m) => (
                <Box
                  key={m.id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 1,
                    borderRadius: 1,
                    border: '1px solid',
                    borderColor: 'divider',
                    gap: 1.5,
                  }}
                >
                  <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Typography variant="body2" fontWeight={600} noWrap title={m.name}>
                      {m.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" noWrap>
                      {m.provider || 'Unknown provider'}
                      {m.ctx ? ` • ctx ${m.ctx}` : ''}
                      {m.priceIn ? ` • ${m.priceIn} in` : ''}
                      {m.priceOut ? ` • ${m.priceOut} out` : ''}
                    </Typography>
                    {m.note && (
                      <Box sx={{ mt: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block' }}
                          noWrap
                        >
                          {m.note}
                        </Typography>
                      </Box>
                    )}
                  </Box>

                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="flex-end">
                    {m.zdr && <Chip label="ZDR" size="small" color="success" variant="outlined" />}
                    <Tooltip
                      title={
                        isPinned(m.id)
                          ? 'Remove from header selector'
                          : 'Pin for header selector'
                      }
                    >
                      <IconButton
                        size="small"
                        onClick={() => togglePin(m)}
                        color={isPinned(m.id) ? 'warning' : 'default'}
                      >
                        {isPinned(m.id) ? <StarIcon fontSize="small" /> : <StarBorderIcon fontSize="small" />}
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </Box>
              ))}
            </Box>
          )}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default ModelCatalogModal;
```

> Nota backend: l’endpoint `GET /api/v1/openrouter/models` è solo un esempio; puoi:
> - adeguarlo a un tuo path,
> - oppure per iniziare commentare la `axios.get(...)` e usare solo fallback `loadHeaderChatModels()`.

---

## 4. Collegamento con il tuo engine “Experimental Forge”

A questo punto:

- Il **model selezionato** è sempre in `localStorage['or_selected_model']`.
- Il selettore in System Settings mostra il valore corrente e permette di cambiarlo.
- Il Catalogo permette di scegliere quali modelli compaiono nel selettore (pinned).

Quando mandi una richiesta Forge al backend, basta leggere:

```ts
import { loadSelectedModel, loadHeaderChatModels } from './openrouter-model-config';

const selectedModelId = loadSelectedModel(); // es. "openai/gpt-4o"

// Quando costruisci il payload per la chiamata LLM:
const payload = {
  // ...
  model_name: selectedModelId,
};
```

Se hai uno store globale (tipo Zustand) per Forge, puoi anche salvare il modello lì anziché rileggere ogni volta da `localStorage`.

---

## 5. Riassunto rapido di cosa incollare/dove

1. **Utility modelli**  
   `openrouter-model-config.ts` con:
   - `HeaderChatModel`
   - `loadHeaderChatModels / saveHeaderChatModels`
   - `loadSelectedModel / saveSelectedModel`.

2. **Selettore nella pagina System Settings**  
   `ModelSelector.tsx`  
   Integrazione nel modal:

   ```tsx
   <ModelSelector onOpenCatalog={() => setCatalogOpen(true)} />
   <ModelCatalogModal open={isCatalogOpen} onClose={() => setCatalogOpen(false)} />
   ```

3. **Modale Catalogo + BYOK**  
   `ModelCatalogModal.tsx` con:
   - Sezione **BYOK** (campo `sk‑or‑...`, bottone SAVE, testo e link).
   - Lista modelli caricati da backend o fallback da pinned.
   - Bottone “pin” che aggiorna `or_header_models`.

Da qui puoi:

- Tenere il modal “System Settings” che già hai e sostituire solo la parte **Inference Model** con `ModelSelector`.
- Sfruttare il modal `ModelCatalogModal` per dare all’utente un posto unico dove:
  - impostare BYOK,
  - vedere/gestire il catalogo,
  - pinnare modelli per l’uso in Forge.

Il codice è volutamente modulare: puoi copiarlo nella nuova app e adattare solo:

- path API (`/api/v1/openrouter/models`),
- eventuali stili/customizzazioni MUI,
- testo in italiano/inglese per la tua UI.