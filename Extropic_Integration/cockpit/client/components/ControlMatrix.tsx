
import React from 'react';
import QuantumTooltip from './QuantumTooltip';

interface ControlMatrixProps {
  temp: number;
  setTemp: (t: number) => void;
  prompt: string;
  setPrompt: (t: string) => void;
  onInject: () => void;
  isProcessing: boolean;
}

const ControlMatrix: React.FC<ControlMatrixProps> = ({ temp, setTemp, prompt, setPrompt, onInject, isProcessing }) => {
  return (
    <aside className="p-6 h-full flex flex-col gap-5 relative overflow-hidden" style={{ borderColor: 'var(--col-muted)', backgroundColor: 'var(--bg-base)' }}>
      {/* Background Grid */}
      <div className="absolute inset-0 bg-[length:20px_20px] pointer-events-none" style={{ backgroundImage: 'linear-gradient(var(--grid-color) 1px, transparent 1px), linear-gradient(90deg, var(--grid-color) 1px, transparent 1px)' }}></div>

      {/* HEADER */}
      <div className="font-mono text-[10px] tracking-[0.2em] relative z-10 flex justify-between items-center" style={{ color: 'var(--col-secondary)' }}>
        <span>:: CONTROL_MATRIX v15.0</span>
        <div className="w-2 h-2 rounded-full animate-pulse shadow-[0_0_5px_currentColor]" style={{ backgroundColor: 'var(--col-primary)' }}></div>
      </div>

      {/* INPUT VECTOR */}
      <div className="flex flex-col gap-2 relative z-10">
        <QuantumTooltip
          title="Perturbation Vector (h_i)"
          desc="Definisci l'intento semantico che guiderà la simulazione."
          mechanism="L'input testuale viene convertito in embeddings (vettori) che agiscono come campo magnetico locale (Bias h_i) sul reticolo di Ising."
          expectation="I nodi del reticolo inizieranno a polarizzarsi verso lo stato target definito dal testo."
          useCase="Definire l'obiettivo della simulazione (es. 'Ottimizza Portafoglio' o 'Simula Crash')."
          type="info"
          position="bottom"
        >
          <label className="text-[10px] font-bold uppercase tracking-widest mb-2 block cursor-help border-b border-dashed w-max" style={{ color: 'var(--text-sub)', borderColor: 'var(--col-muted)' }}>Inject Vector</label>
        </QuantumTooltip>
        <textarea
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value);
            e.target.style.height = 'auto';
            e.target.style.height = e.target.scrollHeight + 'px';
          }}
          rows={1}
          className="border p-3 text-xs font-mono outline-none resize-none transition-colors focus:shadow-[0_0_15px_rgba(var(--col-secondary-rgb),0.1)] overflow-hidden min-h-[40px]"
          style={{
            backgroundColor: 'rgba(var(--bg-surface-rgb), 0.5)',
            borderColor: 'var(--col-muted)',
            color: 'var(--text-main)',
            '--tw-placeholder-opacity': '0.5'
          } as React.CSSProperties}
          placeholder="Definisci lo stato target..."
        />
      </div>

      {/* TEMPERATURE SLIDER */}
      <div className="flex flex-col gap-4 relative z-10">
        <QuantumTooltip
          title="Thermal Noise (Beta)"
          desc="Controlla l'entropia del sistema."
          mechanism="Regola la variabile 'T' nella distribuzione di Boltzmann. T alta = Random Walk. T bassa = Discesa del Gradiente."
          expectation="Alta T: Movimento caotico rapido (rosso). Bassa T: Cristallizzazione lenta (blu)."
          useCase="Usa T>2.0 per uscire da minimi locali (Creatività). Usa T<0.5 per raffinare la soluzione (Precisione)."
          type="energy"
          position="bottom"
        >
          <div className="flex justify-between items-end mb-2 cursor-help">
            <label className="text-[10px] font-bold uppercase tracking-widest border-b border-dashed" style={{ color: 'var(--text-sub)', borderColor: 'var(--col-muted)' }}>Temperature</label>
            <span className={`font-mono text-sm ${temp > 2 ? 'text-red-500' : 'text-cyan-400'}`}>{temp.toFixed(2)}K</span>
          </div>
        </QuantumTooltip>
        <div className="relative h-6 w-full flex items-center">
          <input
            type="range" min="0.1" max="5.0" step="0.1"
            value={temp} onChange={(e) => setTemp(parseFloat(e.target.value))}
            className="w-full h-1 rounded-lg appearance-none cursor-pointer relative z-20"
            style={{ backgroundColor: 'var(--col-muted)' }}
          />
          {/* Heat Gradient Background for Slider */}
          <div className="absolute top-1/2 left-0 right-0 h-0.5 -translate-y-1/2 bg-gradient-to-r from-cyan-900 via-purple-900 to-red-900 z-10"></div>
        </div>
        <div className="flex justify-between text-[9px] font-mono uppercase" style={{ color: 'var(--text-sub)' }}>
          <span>Order</span>
          <span>Chaos</span>
        </div>
      </div>

      {/* MODE SELECTOR */}
      <div className="flex flex-col gap-2 relative z-10">
        <QuantumTooltip
          title="Observation Filter"
          desc="Cambia il modo in cui il sistema interpreta i nodi."
          mechanism="Cambia il mapping semantico dei nodi del grafo (Ontologia)."
          expectation="Nessun cambio visivo immediato, ma cambia il modo in cui l'AI interpreta i risultati."
          type="info"
        >
          <label className="text-[10px] font-bold uppercase tracking-widest mb-1 cursor-help border-b border-dashed w-max" style={{ color: 'var(--text-sub)', borderColor: 'var(--col-muted)' }}>Output Logic</label>
        </QuantumTooltip>
        <div className="flex gap-1 p-1 border rounded" style={{ backgroundColor: 'rgba(var(--bg-surface-rgb), 0.5)', borderColor: 'var(--col-muted)' }}>
          <button className="flex-1 py-1 text-[9px] border rounded font-mono" style={{ color: 'var(--col-primary)', backgroundColor: 'rgba(var(--col-primary-rgb), 0.1)', borderColor: 'rgba(var(--col-primary-rgb), 0.3)' }}>ISIN</button>
          <button className="flex-1 py-1 text-[9px] font-mono hover:text-white transition-colors" style={{ color: 'var(--text-sub)' }}>CODE</button>
          <button className="flex-1 py-1 text-[9px] font-mono hover:text-white transition-colors" style={{ color: 'var(--text-sub)' }}>GEN</button>
        </div>
      </div>

      {/* ACTION BUTTON */}
      <div className="mt-auto relative z-10">
        <QuantumTooltip
          title="System Initialization"
          desc="Avvia il processo di Annealing."
          mechanism="Resetta lo stato quantico, inietta il bias (vettore) e avvia il loop termodinamico di minimizzazione dell'energia."
          expectation="Il reticolo 'esploderà' in attività (Alta energia) per poi raffreddarsi verso una soluzione stabile."
          type="physics"
        >
          <button
            onClick={onInject}
            disabled={isProcessing}
            className={`
                    w-full font-bold py-4 px-4 border-b-2 text-xs font-mono tracking-widest uppercase transition-all
                    ${isProcessing
                ? 'bg-red-900/20 text-red-500 border-red-900 cursor-not-allowed animate-pulse'
                : 'hover:text-white hover:shadow-[0_0_20px_rgba(var(--col-secondary-rgb),0.4)]'
              }
                `}
            style={!isProcessing ? {
              backgroundColor: 'rgba(var(--col-secondary-rgb), 0.1)',
              color: 'var(--col-secondary)',
              borderColor: 'var(--col-secondary)'
            } : {}}
          >
            {isProcessing ? 'ANNEALING...' : 'INITIALIZE [FASE 0]'}
          </button>
        </QuantumTooltip>
        <div className="text-[9px] text-center mt-2 font-mono" style={{ color: 'var(--text-sub)' }}>
          {isProcessing ? 'System Entropy High' : 'Ready for Perturbation'}
        </div>
      </div>
    </aside>
  );
};

export default ControlMatrix;
