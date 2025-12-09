import { AiResponse, AiWidgetResponse, ExperimentConfig } from "../types";

// --- TYPES ---

export interface OpenRouterStatus {
    key_source: 'server' | 'user' | 'none';
    has_server_key: boolean;
    has_user_key: boolean;
}

export interface LlmInvokeRequest {
    model: string;
    system_prompt: string;
    user_prompt: string;
}

export interface ChatModel {
    id: string;
    name: string;
    contextWindow: number;
    pricing: {
        prompt: string; // per 1M tokens
        completion: string; // per 1M tokens
    };
}

// --- CONSTANTS ---

export const AVAILABLE_MODELS: ChatModel[] = [
    {
        id: 'google/gemini-2.0-flash-exp:free',
        name: 'Gemini 2.0 Flash (Free)',
        contextWindow: 1048576,
        pricing: { prompt: '0', completion: '0' }
    },
    {
        id: 'google/gemini-exp-1206:free',
        name: 'Gemini Exp 1206 (Free)',
        contextWindow: 2097152,
        pricing: { prompt: '0', completion: '0' }
    },
    {
        id: 'meta-llama/llama-3.3-70b-instruct',
        name: 'Llama 3.3 70B',
        contextWindow: 128000,
        pricing: { prompt: '0.13', completion: '0.4' }
    },
    {
        id: 'anthropic/claude-3.5-sonnet',
        name: 'Claude 3.5 Sonnet',
        contextWindow: 200000,
        pricing: { prompt: '3.0', completion: '15.0' }
    },
    {
        id: 'openai/gpt-4o',
        name: 'GPT-4o',
        contextWindow: 128000,
        pricing: { prompt: '2.5', completion: '10.0' }
    }
];

const STORAGE_KEY_API_KEY = 'openrouter_api_key';
const STORAGE_KEY_MODEL = 'openrouter_selected_model';

// --- SERVICE ---

export const openRouterService = {

    // --- KEY MANAGEMENT ---

    saveUserKey: (key: string) => {
        if (!key.trim()) {
            localStorage.removeItem(STORAGE_KEY_API_KEY);
        } else {
            localStorage.setItem(STORAGE_KEY_API_KEY, key.trim());
        }
    },

    getUserKey: (): string | null => {
        return localStorage.getItem(STORAGE_KEY_API_KEY);
    },

    // --- MODEL MANAGEMENT ---

    getSelectedModelId: (): string => {
        return localStorage.getItem(STORAGE_KEY_MODEL) || AVAILABLE_MODELS[0].id;
    },

    setSelectedModelId: (modelId: string) => {
        localStorage.setItem(STORAGE_KEY_MODEL, modelId);
    },

    getModelDetails: (modelId: string): ChatModel | undefined => {
        return AVAILABLE_MODELS.find(m => m.id === modelId);
    },

    // --- API CALLS ---

    getStatus: async (): Promise<OpenRouterStatus> => {
        const userKey = openRouterService.getUserKey();
        const headers: HeadersInit = { 'Content-Type': 'application/json' };
        if (userKey) headers['X-OpenRouter-Key'] = userKey;

        const res = await fetch('/openrouter/status', { headers });
        if (!res.ok) throw new Error('Failed to fetch status');
        return res.json();
    },

    invokeLlm: async (systemPrompt: string, userPrompt: string): Promise<any> => {
        const userKey = openRouterService.getUserKey();
        const model = openRouterService.getSelectedModelId();

        const headers: HeadersInit = { 'Content-Type': 'application/json' };
        if (userKey) headers['X-OpenRouter-Key'] = userKey;

        const payload: LlmInvokeRequest = {
            model,
            system_prompt: systemPrompt,
            user_prompt: userPrompt
        };

        const res = await fetch('/api/llm/invoke', {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || `API Error: ${res.statusText}`);
        }

        return res.json();
    }
};
