
import type React from 'react';

export interface TooltipContent {
  function: string;
  mechanism: string;
  expectation: string;
  example: string;
}

export interface CognitiveModule {
  id: string;
  name: string;
  description: string;
  // Icon removed from UI, keeping type optional for backward compatibility if needed, 
  // but strictly we won't render it.
  icon?: React.FC<React.SVGProps<SVGSVGElement>>; 
  tooltip: TooltipContent;
  
  // THE CORE: The actual System Prompt / File Content of the Agent
  instruction: string; 
  
  isCustom?: boolean; // If true, it's a user-created module
  isSystemModified?: boolean; // If true, it's a system module with overrides
}

export interface PreconfiguredIntent {
  id: string;
  label: string;
  category: 'Code' | 'Strategy' | 'Creative' | 'Analysis';
  text: string;
  recommendedModules: string[]; // Array of Module IDs in order
}

export type ApiState = 'idle' | 'loading' | 'success' | 'error';

export interface Risultante {
  livello1: string;
  livello2: string;
  livello3: {
    diagnosi: string;
    traiettoria: string;
    sintesi: string;
    impronta: string;
  };
}

export interface WorkflowStep {
  moduleName: string;
  description: string;
  status: 'completed';
}

export interface Source {
  title: string;
  uri: string;
}

export interface AcosApiResponse {
    steps: WorkflowStep[];
    risultante: Risultante;
    prompt: string;
    sources?: Source[];
}

export interface ChatMessage {
    role: 'user' | 'model';
    text: string;
}
