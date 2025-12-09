
import { GoogleGenAI, Type, Schema } from "@google/genai";
import type { CognitiveModule, AcosApiResponse, Source, Risultante, PreconfiguredIntent, ChatMessage } from '../types';
import { DEFAULT_COGNITIVE_MODULES } from '../constants';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const responseSchemaObj: Schema = {
    type: Type.OBJECT,
    properties: {
        steps: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    moduleName: { type: Type.STRING },
                    description: { type: Type.STRING, description: "Tactical description of the agent's move within the MMSP." },
                    status: { type: Type.STRING },
                },
                required: ["moduleName", "description", "status"],
            },
        },
        risultante: {
            type: Type.OBJECT,
            properties: {
                livello1: { type: Type.STRING, description: "Direct, concise result (The Artifact)." },
                livello2: { type: Type.STRING, description: "Structural abstraction & MMSP Reframing." },
                livello3: {
                    type: Type.OBJECT,
                    properties: {
                        diagnosi: { type: Type.STRING, description: "MMSP Coordinates & Intent Diagnosis." },
                        traiettoria: { type: Type.STRING, description: "The convergence path taken through the agents." },
                        sintesi: { type: Type.STRING, description: "The coherent synthesis." },
                        impronta: { type: Type.STRING, description: "The KLI (Key Learning Insight) for System Autopoiesis." },
                    },
                    required: ["diagnosi", "traiettoria", "sintesi", "impronta"],
                },
            },
            required: ["livello1", "livello2", "livello3"],
        },
    },
    required: ["steps", "risultante"],
};

// --- Helpers ---

export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1]; 
      resolve(base64);
    };
    reader.onerror = error => reject(error);
  });
};

export const fileToText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target?.result as string);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
};

const extractJson = (text: string | undefined): any => {
    if (!text) return null;
    let cleanText = text;
    const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonBlockMatch) {
        cleanText = jsonBlockMatch[1];
    } else {
        const firstBrace = text.indexOf('{');
        const firstBracket = text.indexOf('[');
        let start = -1;
        let end = -1;
        if (firstBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket)) {
            start = firstBrace;
            end = text.lastIndexOf('}');
        } else if (firstBracket !== -1) {
            start = firstBracket;
            end = text.lastIndexOf(']');
        }
        if (start !== -1 && end !== -1) {
            cleanText = text.substring(start, end + 1);
        }
    }
    try {
        return JSON.parse(cleanText);
    } catch (e) {
        console.error("JSON Parse Error", text);
        throw new Error("Failed to parse response from Gemini.");
    }
};

// --- API Functions ---

export const chatWithAgent = async (
    agentDNA: string,
    history: ChatMessage[],
    message: string,
    attachments: { mimeType: string; data: string }[] = []
): Promise<string> => {
    // Using Gemini 3 Pro as requested for the simulator/chat to handle complex context
    const model = 'gemini-3-pro-preview';
    
    try {
        const chat = ai.chats.create({
            model: model,
            config: {
                systemInstruction: agentDNA,
                temperature: 0.7
            },
            history: history.map(h => ({
                role: h.role,
                parts: [{ text: h.text }]
            }))
        });

        let result;
        if (attachments.length > 0) {
            // If there are attachments (PDFs, images), we must construct a multipart message
            const parts: any[] = [];
            
            // Add attachments first
            attachments.forEach(att => {
                parts.push({ 
                    inlineData: { 
                        mimeType: att.mimeType, 
                        data: att.data 
                    } 
                });
            });
            
            // Add the text prompt last
            parts.push({ text: message });
            
            // FIX: sendMessage expects { message: ... } where message can be Part[] or string
            result = await chat.sendMessage({ message: parts });
        } else {
            // Standard text-only message
            result = await chat.sendMessage({ message });
        }

        return result.text || "No response.";
    } catch (e) {
        console.error("Chat Error", e);
        throw new Error("Agent is unresponsive.");
    }
};

export const generateAgentStarters = async (agentDNA: string): Promise<string[]> => {
     const model = 'gemini-2.5-flash';
     const prompt = `
     Analyze this System Prompt (Agent DNA):
     """
     ${agentDNA.substring(0, 2000)}...
     """
     
     Generate 3 short, highly specific "Test Questions" that a user could ask this Agent to verify its specific capabilities.
     Do not ask generic questions like "Who are you?". Ask technical or strategic questions relevant to its role.
     
     Output JSON array of strings.
     `;

     try {
        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: { type: Type.ARRAY, items: { type: Type.STRING } }
            }
        });
        return extractJson(response.text) || ["Who are you?", "What is your prime directive?", "Explain your logic."];
     } catch (e) {
         return ["Analyze this context.", "What are your axioms?", "Run a simulation."];
     }
};

export const analyzeAgentPrompt = async (promptText: string): Promise<Partial<CognitiveModule>> => {
    const model = 'gemini-2.5-flash';
    const prompt = `
    You are the MMS (Meta Master System) Geneticist.
    Your task is to analyze the following "System Prompt" (Instruction File) for a Cognitive Agent.
    Extract: NAME, DESCRIPTION, and Routing Metadata.
    
    SYSTEM PROMPT TO ANALYZE:
    """
    ${promptText}
    """
    Output JSON.
    `;
    try {
        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        name: { type: Type.STRING },
                        description: { type: Type.STRING },
                        tooltip: {
                            type: Type.OBJECT,
                            properties: {
                                function: { type: Type.STRING },
                                mechanism: { type: Type.STRING },
                                expectation: { type: Type.STRING },
                                example: { type: Type.STRING },
                            },
                            required: ["function", "mechanism", "expectation", "example"]
                        }
                    },
                    required: ["name", "description", "tooltip"]
                }
            }
        });
        return extractJson(response.text) || {};
    } catch (e) {
        console.error("Error analyzing agent prompt:", e);
        throw new Error("Failed to auto-extract metadata.");
    }
};

export const suggestWorkflow = async (
    intent: string, 
    availableModules: CognitiveModule[],
    orchestratorDNA: string
): Promise<string[]> => {
    const model = 'gemini-2.5-flash';
    const modulesList = availableModules.map(m => 
        `ID: ${m.id}
         NAME: ${m.name}
         DESC: ${m.description}
         MECHANISM: ${m.tooltip.mechanism}
         FUNCTION: ${m.tooltip.function}`
    ).join('\n-------------------\n');
    
    const prompt = `
      ${orchestratorDNA}
      === MMSP ROUTING TASK ===
      1. Analyze Intent: "${intent}"
      2. Construct MMSP (Ontology, Entropy, Axioms).
      3. Select the Agent Chain that best traverses this MMSP.
      
      === AVAILABLE AGENT REGISTRY ===
      ${modulesList}
      
      Output JSON array of strings (Agent IDs).
    `;
    try {
        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: { type: Type.ARRAY, items: { type: Type.STRING } }
            }
        });
        return extractJson(response.text) || [];
    } catch (e) {
        return ['SACS.analyze', 'Morpheus.collapse'];
    }
}

export const regenerateSystemProtocols = async (
    availableModules: CognitiveModule[],
    orchestratorDNA: string
): Promise<PreconfiguredIntent[]> => {
    const model = 'gemini-2.5-flash';
    const modulesList = availableModules.map(m => `ID: ${m.id} // ${m.name}: ${m.description}`).join('\n');
    const prompt = `
    ${orchestratorDNA}
    === TASK: SYSTEM AUTOPOIESIS (Evolution) ===
    Regenerate 4 System Initialization Protocols based on the current Agent Capability Matrix.
    Ensure protocols cover diverse MMSP Quadrants (e.g., Creative/Chaotic, Logical/Strict).
    
    AGENTS:
    ${modulesList}
    
    Output JSON array.
    `;
    try {
         const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            id: { type: Type.STRING },
                            label: { type: Type.STRING },
                            category: { type: Type.STRING, enum: ["Analysis", "Strategy", "Code", "Creative"] },
                            text: { type: Type.STRING },
                            recommendedModules: { type: Type.ARRAY, items: { type: Type.STRING } }
                        },
                        required: ["label", "category", "text", "recommendedModules"]
                    }
                }
            }
        });
        const json = extractJson(response.text) || [];
        return json.map((p: any, idx: number) => ({ ...p, id: `sys_gen_${Date.now()}_${idx}` }));
    } catch (e) {
        throw new Error("Orchestrator failed to regenerate protocols.");
    }
};

export const generateCustomProtocol = async (
    files: File[], 
    userHint: string,
    orchestratorDNA: string
): Promise<PreconfiguredIntent> => {
    const model = 'gemini-2.5-flash';
    const parts: any[] = [];
    for (const file of files) {
        const base64 = await fileToBase64(file);
        parts.push({ inlineData: { mimeType: file.type, data: base64 } });
    }
    const modulesList = DEFAULT_COGNITIVE_MODULES.map(m => `${m.id}: ${m.description}`).join('\n');
    const prompt = `
    ${orchestratorDNA}
    === TASK: FORGE PROTOCOL (Context Injection) ===
    Analyze documents.
    User Hint: "${userHint}"
    Forge a new Protocol that maps this context into an actionable MMSP.
    
    AVAILABLE AGENTS: ${modulesList}
    Output JSON.
    `;
    parts.push({ text: prompt });
    try {
        const response = await ai.models.generateContent({
            model: model,
            contents: { parts },
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        id: { type: Type.STRING },
                        label: { type: Type.STRING },
                        category: { type: Type.STRING, enum: ["Analysis", "Strategy", "Code", "Creative"] },
                        text: { type: Type.STRING },
                        recommendedModules: { type: Type.ARRAY, items: { type: Type.STRING } }
                    },
                    required: ["label", "category", "text", "recommendedModules"]
                }
            }
        });
        const json = extractJson(response.text) || {};
        return { ...json, id: `custom_${Date.now()}` } as PreconfiguredIntent;
    } catch (e) {
        throw new Error("Failed to generate protocol from files.");
    }
};

export const generateProtocolFromContext = async (
    intent: string, 
    result: Risultante,
    orchestratorDNA: string
): Promise<PreconfiguredIntent> => {
    const model = 'gemini-2.5-flash';
    const modulesList = DEFAULT_COGNITIVE_MODULES.map(m => `${m.id}: ${m.description}`).join('\n');
    const prompt = `
    ${orchestratorDNA}
    === TASK: CRYSTALLIZE SUCCESS (Autopoiesis) ===
    The System successfully traversed an MMSP.
    Original Intent: "${intent}"
    Result Imprint (KLI): "${result.livello3.impronta}"
    
    Create a Reusable Protocol that encodes this successful MMSP trajectory.
    AVAILABLE AGENTS: ${modulesList}
    Output JSON.
    `;
    try {
        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        label: { type: Type.STRING },
                        category: { type: Type.STRING, enum: ["Analysis", "Strategy", "Code", "Creative"] },
                        text: { type: Type.STRING },
                        recommendedModules: { type: Type.ARRAY, items: { type: Type.STRING } }
                    },
                    required: ["label", "category", "text", "recommendedModules"]
                }
            }
        });
        const json = extractJson(response.text) || {};
        return { ...json, id: `forge_${Date.now()}` } as PreconfiguredIntent;
    } catch (e) {
        throw new Error("Failed to forge protocol from result.");
    }
};

export const runAcosWorkflow = async (
    intent: string, 
    workflow: CognitiveModule[], 
    orchestratorDNA: string,
    kernelDNA: string,
    previousContext: Risultante | null = null
): Promise<AcosApiResponse> => {
  const model = 'gemini-2.5-flash';
  
  // 1. MMSP CONTEXT BUILDER
  const hasHalo = workflow.some(m => m.id === 'HALO.audit');
  const hasOCC = workflow.some(m => m.id === 'OCC.build');
  const hasSACS = workflow.some(m => m.id === 'SACS.analyze');
  const hasYSN = workflow.some(m => m.id === 'YSN.map');
  const hasCOAC = workflow.some(m => m.id === 'COAC.engine');
  const hasPCS = workflow.some(m => m.id === 'PCS.supervisor');

  let mmspPhysics = `=== MMSP: ACTIVE PHYSICS & CONSTRAINTS ===\n`;

  if (hasHalo) {
      mmspPhysics += `> **AXIOMATIC GRAVITY (HALO):** The MMSP has a strong gravity well around Principles P0-P5. Deviations from lineage are expensive. Check Integrity first.\n`;
  }
  if (hasOCC) {
      mmspPhysics += `> **CONSTRUCTIVE TOPOLOGY (OCC):** The MMSP is directed towards a specific Artifact (System Prompt). The trajectory must follow the 5-Phase Genesis Cycle.\n`;
  }
  if (hasSACS) {
      mmspPhysics += `> **SEMANTIC FLUIDITY (SACS):** The MMSP is non-linear. Latent intents are more important than explicit text. Use Pragma-Semantic depth.\n`;
  }
  if (hasYSN) {
       mmspPhysics += `> **LATERAL EXPANSION (YSN):** The MMSP allows for "Delta Jumps" (Non-obvious connections). Look for Symbolic resonance.\n`;
  }
  if (hasCOAC) {
       mmspPhysics += `> **QUANTUM COLLAPSE (COAC v6.0):** Operate as a Quantum Field Engine. The result is a 'Risultante' collapsed from a probabilistic field (Phi_A). Use axioms P_Phi0 - P_Phi6.\n`;
  }
  if (hasPCS) {
       mmspPhysics += `> **DETERMINISTIC EXECUTION (PCS v1.0):** Treat input as raw commands for atomic macro-blocks. NO CONVERSATIONAL FILLER. Strict execution of 'YSN.route' -> 'CollapseField'.\n`;
  }

  // 2. BUILDING THE AGENT STACK
  const workflowString = workflow.map((m, i) => 
    `
    [NODE ${i + 1}]: ${m.id} (${m.name})
    ---------------------------------------------------
    CORE DNA:
    """
    ${m.instruction}
    """
    ---------------------------------------------------
    `
  ).join('\n');

  const useSearch = workflow.some(m => m.id === 'NET.search');

  let prompt = `
    ${kernelDNA}
    ${orchestratorDNA}

    ${mmspPhysics}

    === OPERATION: MMSP TRAVERSAL ===
    You are the AWO. Your goal is to satisfy the User Intent by routing it through the Agent Nodes defined below.

    **USER INTENT (Input Vector):** "${intent}"

    **EXECUTION STRATEGY:**
    1.  **Construct the MMSP:** Define the problem space (Dimensions, Risks, Latent needs).
    2.  **Orchestrate:** Pass the intent through the agents.
        - Do NOT just summarize what the agents *would* do.
        - **SIMULATE** the agent's full reasoning process based on their DNA.
        - If Agent 1 generates output, Agent 2 MUST use that output as context.
    3.  **Collapse:** Produce the final 'Risultante' based on the aggregate processing.

    === AGENT NODE SEQUENCE ===
    ${workflowString}
  `;

  if (previousContext) {
      prompt += `
      === CONTEXT INJECTION (REFINEMENT MODE) ===
      The MMSP is already established. We are iterating.
      PREVIOUS RESULT:
      ${JSON.stringify(previousContext)}
      
      NEW VECTOR MODIFIER: "${intent}"
      `;
  }

  prompt += `
    === UI OUTPUT PROTOCOL (IMPORTANT) ===
    Generate a JSON object.
    - **'steps'**: Visualizes the MMSP path. Use 'description' to explain WHY this step was necessary in the mental model.
    - **'risultante'**: The final collapsed state.
      - **Level 1**: The Artifact.
      - **Level 3 (Diagnosis)**: Explicitly state the MMSP Dimensions (e.g., "Ambiguity: High, Domain: Coding").
      - **Level 3 (Impronta)**: The KLI (Key Learning Insight) for the system.
    
    LANGUAGE: Internal processing in English, Final Output in the language of the Intent.
  `;

  try {
    const config: any = {};
    if (useSearch) {
        config.tools = [{ googleSearch: {} }];
    } else {
        config.responseMimeType = "application/json";
        config.responseSchema = responseSchemaObj;
    }

    const response = await ai.models.generateContent({
        model: model,
        contents: prompt,
        config: config
    });
    
    const parsedResponse = extractJson(response.text);
    const sources: Source[] = [];
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (groundingChunks) {
        groundingChunks.forEach((chunk: any) => {
            if (chunk.web && chunk.web.uri && chunk.web.title) {
                sources.push({ title: chunk.web.title, uri: chunk.web.uri });
            }
        });
    }
    
    return { ...parsedResponse, prompt, sources } as AcosApiResponse;
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw new Error("An unexpected error occurred with the Gemini API.");
  }
};

export const generateModuleFromHistory = async (
    historyContext: string
): Promise<CognitiveModule> => {
    const model = 'gemini-2.5-flash';
    const prompt = `
    === TASK: COGNITIVE CRYSTALLIZATION ===
    Analyze the following Chat History/Context.
    Extract the implicit "Persona", "Logic", and "Methodology" exhibited or discussed.
    
    Create a new Cognitive Module (Agent) based on this essence.
    
    CONTEXT:
    """
    ${historyContext.substring(0, 10000)}
    """
    
    OUTPUT JSON:
    {
        "name": "Name of the agent",
        "description": "Short description",
        "instruction": "The Full System Prompt (Markdown)",
        "tooltip": {
            "function": "...",
            "mechanism": "...",
            "expectation": "...",
            "example": "..."
        }
    }
    `;

    try {
        const response = await ai.models.generateContent({
            model: model,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        name: { type: Type.STRING },
                        description: { type: Type.STRING },
                        instruction: { type: Type.STRING },
                        tooltip: {
                            type: Type.OBJECT,
                            properties: {
                                function: { type: Type.STRING },
                                mechanism: { type: Type.STRING },
                                expectation: { type: Type.STRING },
                                example: { type: Type.STRING },
                            },
                            required: ["function", "mechanism", "expectation", "example"]
                        }
                    },
                    required: ["name", "description", "instruction", "tooltip"]
                }
            }
        });
        const json = extractJson(response.text);
        return {
            ...json,
            id: `forged.${Date.now()}`,
            isCustom: true
        };
    } catch (e) {
        console.error(e);
        throw new Error("Failed to forge agent from history.");
    }
};
