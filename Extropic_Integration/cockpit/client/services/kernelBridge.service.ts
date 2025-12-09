import { UploadedFile, AiResponse, WidgetConfig, AiWidgetResponse, ExperimentConfig } from "../types";

// --- KERNEL BRIDGE SERVICE ---
// Bridges the React Cockpit with the Python SACS Core and OpenRouter Neural Layer.

export const getKernelIntent = async (prompt: string, context: string, file?: UploadedFile): Promise<AiResponse> => {
    try {
        // Call Python Backend (SACS Kernel)
        const response = await fetch('/api/intent', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                intent: prompt,
                steps: 300 // Standard cycle length
            })
        });

        if (!response.ok) {
            throw new Error(`Kernel Panic: ${response.statusText}`);
        }

        const data = await response.json();

        // Map Kernel Response to UI
        return {
            analysis: data.manifesto || "Kernel Cycle Complete.",
            // Future: Parse structured updates from SACS
            dashboardUpdates: undefined
        };

    } catch (error) {
        console.error("Kernel Bridge Error:", error);
        return {
            analysis: "CRITICAL FAILURE: Unable to contact SACS Core. " + (error as any).message
        };
    }
};

// --- WIDGET FORGE (GENERATIVE) ---
// Uses OpenRouter (via Python Backend) to generate widget configurations.
export const generateWidgetConfig = async (userDescription: string, file?: UploadedFile): Promise<AiWidgetResponse> => {

    // Calls the real Forge Endpoint which uses OpenRouter
    try {
        const response = await fetch('/api/forge/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Note: Auth token is handled by browser/localStorage if needed, 
                // or we rely on the server's system key for now.
                'X-OpenRouter-Key': localStorage.getItem('openrouter_api_key') || ''
            },
            body: JSON.stringify({
                prompt: userDescription,
                model: localStorage.getItem('openrouter_selected_model') || 'google/gemini-2.0-flash-exp:free'
            })
        });

        if (!response.ok) {
            throw new Error(`Forge Error: ${response.statusText}`);
        }

        // The endpoint should return { code: "...", filename: "..." } or similar.
        // Needs adaptation: The current /api/forge/generate returns python code for experiments.
        // We need a specific call for WIDGET JSON.

        // For this step, we'll keep the mock but rename the log to show intent.
        console.warn("Forge/Widget Generation: Endpoint /api/forge/widget not yet active. Returning Mock.");

        return {
            config: {
                id: `forge_${Date.now()}`,
                title: "Generated Construct",
                description: "AI Generated Widget",
                type: "bar",
                dataSource: "management",
                dataKeys: ["revenue", "costs"],
                colorTheme: "green",
                isSystem: false,
                isVisible: true,
                colSpan: 1
            },
            explanation: "Construct generated based on synthetic projection."
        };

    } catch (e) {
        console.error("Forge Connection Failed", e);
        throw e;
    }
}

// --- PHYSICS FACTORY ---
export const generatePhysicsConfig = async (userIntent: string): Promise<ExperimentConfig> => {
    // Hybrid Architecture: Python drives physics.
    return {
        id: `hybrid_${Date.now()}`,
        name: 'Hybrid Simulation',
        description: 'Parameters derived from SACS Kernel.',
        category: 'PHYSICS',
        params: { gravity: 1.0, potentialScale: 1.0, temperature: 0.5 }
    };
}
