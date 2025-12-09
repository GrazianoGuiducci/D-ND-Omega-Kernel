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

// Fallback statico vuoto
const FALLBACK_MODELS: HeaderChatModel[] = [];

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
        return cleaned;
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
        } catch { }
    } catch { }
}

export function loadSelectedModel(models?: HeaderChatModel[]): string {
    if (typeof window === 'undefined') {
        return (models && models[0]?.id) || '';
    }
    try {
        const raw = window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY);
        if (raw && raw.trim().length > 0) return raw.trim();
    } catch { }
    const list = models && models.length ? models : loadHeaderChatModels();
    return list[0]?.id || '';
}

export function saveSelectedModel(modelId: string): void {
    if (typeof window === 'undefined') return;
    try {
        window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, modelId);
    } catch { }
}
