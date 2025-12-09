import React, { useEffect, useState } from 'react';
import {
    HeaderChatModel,
    loadHeaderChatModels,
    loadSelectedModel,
    saveSelectedModel,
} from '../services/openrouter-model-config';

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

    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
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
        <div className="flex flex-col gap-2">
            <label className="text-xs text-slate-300 font-bold uppercase tracking-wide">
                Inference Model
            </label>
            <div className="flex gap-2 items-center flex-wrap">
                <div className="flex-1 min-w-[200px]">
                    <select
                        value={selected}
                        onChange={handleChange}
                        className="w-full bg-black border border-white/20 rounded px-3 py-2 text-xs text-white font-mono focus:border-neon-cyan focus:outline-none"
                    >
                        {models.map((m) => (
                            <option key={m.id} value={m.id}>
                                {formatLabel(m)}
                            </option>
                        ))}
                    </select>
                </div>

                <button
                    onClick={onOpenCatalog}
                    className="bg-white/10 hover:bg-white/20 text-white text-xs px-3 py-2 rounded border border-white/10 transition-colors whitespace-nowrap"
                >
                    Manage / Catalog
                </button>
            </div>
            <p className="text-[10px] text-slate-500">
                The selected model will be used for all Experimental Forge operations.
            </p>
        </div>
    );
};

export default ModelSelector;
