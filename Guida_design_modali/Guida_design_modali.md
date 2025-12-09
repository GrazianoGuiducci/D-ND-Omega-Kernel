# Guida Design Modali – MMS Omega Kernel Tester

Questa guida documenta il **pattern di design dei modali** usato in `mms-omega-kernel-tester`, con particolare attenzione a:

- Layout fullscreen/desktop con **resize del modale** (drag dall’angolo in basso a destra)
- **Sidebar resizabile** con snapping a tab verticale (collapse a 48px) e ri‑espansione
- Pattern di **gestione globale del drag** con `requestAnimationFrame`
- Componente `Resizer` riusabile

I due esempi principali sono:

- `AgentSimulatorModal.tsx` – Modale di simulazione agente + tab "FORGE AGENT"
- `ModuleEditorModal.tsx` – Modale editor/registry di moduli cognitivi

---

## 1. Anatomia di un Modal Resizabile a Schermo Intero

Pattern comune ai due modali:

```tsx
return (
  <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-50 flex items-center justify-center p-0 md:p-6">
    <div
      className="bg-gray-800 shadow-2xl border-gray-700 flex flex-col overflow-hidden relative
                  md:rounded-2xl transition-all duration-200 ease-out"
      style={{ width: isMobile ? '100%' : `${modalSize.width}px`, height: isMobile ? '100%' : `${modalSize.height}px` }}
    >
      {/* ...contenuto... */}

      {!isMobile && (
        <div
          className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 flex items-end justify-end p-1 group"
          onMouseDown={startResizingModal}
        >
          <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-cyan-400 transition-colors"></div>
        </div>
      )}
    </div>
  </div>
);
```

### 1.1. Overlay + container modale

- **Overlay**: `fixed inset-0`, sfondo scuro + blur, flex centering.
- **Container modale**:
  - Desktop: `md:rounded-2xl`, dimensione controllata da stato `modalSize` (`{ width, height }`).
  - Mobile: `w-full h-full`, bordi non arrotondati.
  - Animazioni: `transition-all duration-200 ease-out`, disattivate durante il drag (`isResizing`).

```ts
const [modalSize, setModalSize] = useState({ width: 1100, height: 750 });
const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
const [isResizing, setIsResizing] = useState(false);
```

Responsività:

```ts
useEffect(() => {
  const handleWindowResize = () => setIsMobile(window.innerWidth < 768);
  window.addEventListener('resize', handleWindowResize);
  return () => window.removeEventListener('resize', handleWindowResize);
}, []);
```

---

## 2. Resize del Modale con Drag Angolare

Entrambi i modali usano lo stesso schema:

- Refs booleani per sapere se stai trascinando **modale** o **sidebar**
- `dragStartRef` con coordinate iniziali e dimensioni
- `requestAnimationFrame` per aggiornamenti fluidi
- Event handler globali su `document` per `mousemove` e `mouseup`

### 2.1. Stato e ref comuni

```ts
const isResizingSidebarRef = useRef(false);
const isResizingModalRef = useRef(false);
const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
const rafRef = useRef<number | null>(null);
const [isResizing, setIsResizing] = useState(false);
```

### 2.2. Avvio del resize del modale

```ts
const startResizingModal = useCallback((e: React.MouseEvent) => {
  e.preventDefault();
  e.stopPropagation();
  isResizingModalRef.current = true;
  dragStartRef.current = { x: e.clientX, y: e.clientY, w: modalSize.width, h: modalSize.height };
  setIsResizing(true);
}, [modalSize]);
```

Questo handler è collegato al "corner resizer":

```tsx
<div
  className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 flex items-end justify-end p-1 group"
  onMouseDown={startResizingModal}
>
  <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-cyan-400 transition-colors" />
</div>
```

### 2.3. Gestione global mousemove/mouseup

```ts
useEffect(() => {
  const handleMouseMove = (e: MouseEvent) => {
    if (!isResizingSidebarRef.current && !isResizingModalRef.current) return;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    rafRef.current = requestAnimationFrame(() => {
      const deltaX = e.clientX - dragStartRef.current.x;
      const deltaY = e.clientY - dragStartRef.current.y;

      if (isResizingModalRef.current) {
        const newWidth = Math.max(700, Math.min(window.innerWidth - 20, dragStartRef.current.w + deltaX));
        const newHeight = Math.max(500, Math.min(window.innerHeight - 20, dragStartRef.current.h + deltaY));
        setModalSize({ width: newWidth, height: newHeight });
      }

      if (isResizingSidebarRef.current) {
        // gestito al §3
      }
    });
  };

  const handleMouseUp = () => {
    if (isResizingSidebarRef.current || isResizingModalRef.current) {
      isResizingSidebarRef.current = false;
      isResizingModalRef.current = false;
      setIsResizing(false);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    }
  };

  document.addEventListener('mousemove', handleMouseMove);
  document.addEventListener('mouseup', handleMouseUp);

  return () => {
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  };
}, []);
```

**Note chiave:**

- `requestAnimationFrame` → evita aggiornamenti troppo frequenti e rende il drag fluido.
- `Math.max` / `Math.min` → evitano che il modale esca dal viewport.
- `isResizing` → usato per togliere le transition CSS durante il drag.

---

## 3. Sidebar Resizabile con Snap e Tab Verticale

Sia `AgentSimulatorModal` che `ModuleEditorModal` hanno una **sidebar sinistra** che può essere ridimensionata o collassata in una "barra verticale" (48px) con testo ruotato.

### 3.1. Stato e costanti

```ts
const SNAP_THRESHOLD = 250; // sotto questa larghezza, collassa
const [sidebarWidth, setSidebarWidth] = useState(280); // o 350 a seconda del modal
```

### 3.2. Avvio resize sidebar

```ts
const startResizingSidebar = useCallback((e: React.MouseEvent) => {
  e.preventDefault();
  e.stopPropagation();
  isResizingSidebarRef.current = true;
  dragStartRef.current = { x: e.clientX, y: e.clientY, w: sidebarWidth || 280, h: 0 };
  setIsResizing(true);
}, [sidebarWidth]);
```

Collegato al componente `Resizer`:

```tsx
{!isMobile && sidebarWidth > 0 && (
  <div className="hidden md:block h-full flex-shrink-0 relative z-10 -ml-1">
    <Resizer onMouseDown={startResizingSidebar} isVisible={true} />
  </div>
)}
```

### 3.3. Logica di snap in `handleMouseMove`

```ts
if (isResizingSidebarRef.current) {
  const rawWidth = dragStartRef.current.w + deltaX;

  if (rawWidth < SNAP_THRESHOLD) {
    // collassa la sidebar
    setSidebarWidth(0);
  } else {
    const newWidth = Math.max(SNAP_THRESHOLD, Math.min(500 /* o 600 */, rawWidth));
    setSidebarWidth(newWidth);
  }
}
```

### 3.4. Render sidebar: stato collassato vs espanso

**Stato collassato** – width 48px, bottone con testo verticale:

```tsx
<div
  className="flex-shrink-0 h-full transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] relative z-10"
  style={{ width: isMobile ? '100%' : (sidebarWidth === 0 ? '48px' : `${sidebarWidth}px`) }}
>
  {sidebarWidth === 0 && !isMobile ? (
    <div className="w-full h-full">
      <button
        onClick={() => setSidebarWidth(280)} // o 350
        className="h-full w-full bg-gray-900/95 border-r border-gray-800 rounded-r-2xl flex flex-col
                   items-center justify-between py-8 hover:bg-gray-800 hover:border-cyan-500/60
                   transition-all duration-300 group cursor-pointer relative z-10"
      >
        {/* Icona */}
        <div className="p-2 rounded-lg bg-gray-800/50 text-cyan-500 group-hover:bg-cyan-500 group-hover:text-white transition-all mb-4">
          <ChevronDownIcon className="w-5 h-5 -rotate-90" />
        </div>

        {/* Testo verticale */}
        <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
          <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-cyan-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
            <span>Registry</span>
            <span className="w-px h-8 bg-gray-700 group-hover:bg-cyan-500/50 transition-colors"></span>
          </div>
        </div>

        {/* Dot */}
        <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-cyan-400 transition-colors mt-4"></div>
      </button>
    </div>
  ) : (
    // Contenuto sidebar espansa
    <div className="bg-gray-900/60 h-full border-b md:border-b-0 md:border-r border-gray-700 flex flex-col w-full overflow-hidden">
      {/* header + lista / DNA ecc. */}
    </div>
  )}
</div>
```

Questo pattern è **riusabile** per qualsiasi sidebar che vuoi far collassare in un tab verticale:

- `SNAP_THRESHOLD` controlla quando collassare.
- `48px` è la larghezza del tab verticale (puoi cambiarla).
- Un semplice `setSidebarWidth(defaultWidth)` ri‑espande la sidebar.

---

## 4. Componente `Resizer` (Gutter centrale)

`Resizer.tsx` è un componente minimale ma riutilizzabile per la "maniglia" tra sidebar e area principale.

```tsx
interface ResizerProps {
  onMouseDown: (e: React.MouseEvent) => void;
  isVisible: boolean;
}

export const Resizer: React.FC<ResizerProps> = ({ onMouseDown, isVisible }) => {
  if (!isVisible) return null;

  return (
    <div
      className="group h-full w-4 cursor-col-resize flex-shrink-0 flex flex-col justify-center items-center relative z-20 hover:bg-cyan-900/20 transition-colors duration-200"
      onMouseDown={onMouseDown}
    >
      <div className="h-full w-[2px] bg-gray-600 group-hover:bg-cyan-400 group-active:bg-cyan-300 transition-colors duration-150 shadow-[0_0_5px_rgba(0,0,0,0.5)] group-hover:shadow-[0_0_10px_rgba(6,182,212,0.8)]"></div>
      <div className="absolute top-1/2 -translate-y-1/2 w-6 h-10 rounded-full bg-gray-800 border border-gray-500 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none shadow-lg z-30">
        <div className="w-0.5 h-4 bg-gray-400 mx-0.5"></div>
        <div className="w-0.5 h-4 bg-gray-400 mx-0.5"></div>
      </div>
    </div>
  );
};
```

Uso:

```tsx
{!isMobile && sidebarWidth > 0 && (
  <Resizer onMouseDown={startResizingSidebar} isVisible={true} />
)}
```

Puoi riciclare questo componente in qualunque layout a colonne che richieda resize orizzontale.

---

## 5. Pattern Complessivo (Pseudo‑Template)

Di seguito un **template sintetico** che combina tutti i pattern visti, pronto da copiare in una nuova app:

```tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Resizer } from './Resizer';

const SNAP_THRESHOLD = 250;

export const AdvancedModal: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [modalSize, setModalSize] = useState({ width: 1100, height: 750 });
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [isResizing, setIsResizing] = useState(false);

  const isResizingSidebarRef = useRef(false);
  const isResizingModalRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, w: 0, h: 0 });
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const startResizingSidebar = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isResizingSidebarRef.current = true;
    dragStartRef.current = { x: e.clientX, y: e.clientY, w: sidebarWidth || 300, h: 0 };
    setIsResizing(true);
  }, [sidebarWidth]);

  const startResizingModal = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isResizingModalRef.current = true;
    dragStartRef.current = { x: e.clientX, y: e.clientY, w: modalSize.width, h: modalSize.height };
    setIsResizing(true);
  }, [modalSize]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizingSidebarRef.current && !isResizingModalRef.current) return;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);

      rafRef.current = requestAnimationFrame(() => {
        const deltaX = e.clientX - dragStartRef.current.x;
        const deltaY = e.clientY - dragStartRef.current.y;

        if (isResizingSidebarRef.current) {
          const rawWidth = dragStartRef.current.w + deltaX;
          if (rawWidth < SNAP_THRESHOLD) {
            setSidebarWidth(0);
          } else {
            const newWidth = Math.max(SNAP_THRESHOLD, Math.min(500, rawWidth));
            setSidebarWidth(newWidth);
          }
        }

        if (isResizingModalRef.current) {
          const newWidth = Math.max(700, Math.min(window.innerWidth - 20, dragStartRef.current.w + deltaX));
          const newHeight = Math.max(500, Math.min(window.innerHeight - 20, dragStartRef.current.h + deltaY));
          setModalSize({ width: newWidth, height: newHeight });
        }
      });
    };

    const handleMouseUp = () => {
      if (isResizingSidebarRef.current || isResizingModalRef.current) {
        isResizingSidebarRef.current = false;
        isResizingModalRef.current = false;
        setIsResizing(false);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-50 flex items-center justify-center p-0 md:p-6">
      <div
        className={`bg-gray-800 shadow-2xl border-gray-700 flex flex-col overflow-hidden relative
                    ${isMobile ? 'w-full h-full rounded-none' : 'md:rounded-2xl'}
                    ${isResizing ? 'transition-none' : 'transition-all duration-200 ease-out'}`}
        style={{
          width: isMobile ? '100%' : `${modalSize.width}px`,
          height: isMobile ? '100%' : `${modalSize.height}px`,
        }}
      >
        {/* Header */}
        <div className="h-10 bg-gray-950 border-b border-gray-700 flex items-center justify-between px-3">
          <span className="text-xs font-bold text-gray-300">Advanced Modal</span>
          <button onClick={onClose} className="text-gray-500 hover:text-white p-1">
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex flex-1 min-h-0">
          {/* Sidebar */}
          <div
            className="flex-shrink-0 h-full transition-[width] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] relative z-10"
            style={{ width: isMobile ? '100%' : (sidebarWidth === 0 ? '48px' : `${sidebarWidth}px`) }}
          >
            {sidebarWidth === 0 && !isMobile ? (
              <div className="w-full h-full">
                <button
                  onClick={() => setSidebarWidth(300)}
                  className="h-full w-full bg-gray-900/95 border-r border-gray-800 rounded-r-2xl flex flex-col items-center justify-between py-8 hover:bg-gray-800 hover:border-cyan-500/60 transition-all duration-300 group"
                >
                  <div className="p-2 rounded-lg bg-gray-800/50 text-cyan-500 group-hover:bg-cyan-500 group-hover:text-white transition-all mb-4">
                    {/* Icona */}
                    <span className="block w-5 h-5 rotate-[-90deg]">›</span>
                  </div>
                  <div className="flex-1 flex items-center justify-center w-full overflow-hidden py-4">
                    <div className="rotate-180 [writing-mode:vertical-rl] text-xs font-bold tracking-[0.3em] text-gray-500 group-hover:text-cyan-200 transition-colors whitespace-nowrap uppercase flex items-center gap-4">
                      <span>Sidebar</span>
                      <span className="w-px h-8 bg-gray-700 group-hover:bg-cyan-500/50 transition-colors"></span>
                    </div>
                  </div>
                  <div className="w-1.5 h-1.5 rounded-full bg-gray-700 group-hover:bg-cyan-400 transition-colors mt-4"></div>
                </button>
              </div>
            ) : (
              <div className="bg-gray-900/60 h-full border-r border-gray-700 flex flex-col w-full overflow-hidden">
                {/* Contenuto reale della sidebar qui */}
                <div className="p-3 border-b border-gray-700 text-xs font-bold text-gray-400 uppercase">Sidebar</div>
                <div className="flex-1 overflow-auto p-3 text-gray-300 text-xs">...</div>
              </div>
            )}
          </div>

          {/* Gutter Resizer */}
          {!isMobile && sidebarWidth > 0 && (
            <Resizer onMouseDown={startResizingSidebar} isVisible={true} />
          )}

          {/* Area principale */}
          <div className="flex-1 flex flex-col bg-gray-900">
            {/* Contenuto principale qui */}
          </div>
        </div>

        {/* Corner resizer per il modale */}
        {!isMobile && (
          <div
            className="absolute bottom-0 right-0 w-5 h-5 cursor-nwse-resize z-50 flex items-end justify-end p-1 group"
            onMouseDown={startResizingModal}
          >
            <div className="w-2 h-2 border-r-2 border-b-2 border-gray-600 group-hover:border-cyan-400 transition-colors" />
          </div>
        )}
      </div>
    </div>
  );
};
```

Con questi frammenti hai tutto il necessario per **ricreare il modale e le sue funzionalità** (resize, sidebar collassabile, layout responsive) in altre app.</content>,
