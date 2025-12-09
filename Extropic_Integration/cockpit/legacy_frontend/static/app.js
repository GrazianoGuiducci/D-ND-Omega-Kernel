const API_BASE = "/api";

// DOM Elements
const intentInput = document.getElementById('intentInput');
const processBtn = document.getElementById('processBtn');
const dipoleContainer = document.getElementById('dipoleContainer');
const manifestoOutput = document.getElementById('manifestoOutput');
const taxonomyList = document.getElementById('taxonomyList');

// Metrics
const valCoherence = document.getElementById('valCoherence');
const barCoherence = document.getElementById('barCoherence');
const valTension = document.getElementById('valTension');
const barTension = document.getElementById('barTension');
const valDensity = document.getElementById('valDensity');
const valEnergy = document.getElementById('valEnergy');

// Docs Elements
const docsModal = document.getElementById('docsModal');
const docsList = document.getElementById('docsList');
const docsViewer = document.getElementById('docsViewer');
const docsResizer = document.getElementById('docsResizer');
const docsSidebarGutter = document.getElementById('docsSidebarGutter');

// Visualization Elements
const btnViewCortex = document.getElementById('btnViewCortex');
const btnViewTensor = document.getElementById('btnViewTensor');
const btnViewPipeline = document.getElementById('btnViewPipeline');
const cortexCanvas = document.getElementById('cortexCanvas');
const tensorCanvas = document.getElementById('tensorCanvas');
const hudEntropy = document.getElementById('hudEntropy');
const hudGravity = document.getElementById('hudGravity');
const didacticPanel = document.getElementById('didacticPanel');
// Updated selector for new layout
const visualContent = document.querySelector('.visualizer-area') || document.querySelector('.visual-content');

// Global State for Redraw
let currentState = {
    lattice: [],
    tensor_field: [],
    metrics: { temperature: 0.5, gravity: 1.0 }
};
let currentView = 'cortex'; // 'cortex' or 'tensor'

function initViewSwitcher() {
    if (!btnViewCortex || !btnViewTensor || !btnViewPipeline || !cortexCanvas || !tensorCanvas || !didacticPanel || !visualContent) {
        console.error("View Switcher Elements not found!", {
            btnViewCortex, btnViewTensor, btnViewPipeline, cortexCanvas, tensorCanvas, didacticPanel, visualContent
        });
        return;
    }

    btnViewCortex.addEventListener('click', () => setActiveView('cortex'));
    btnViewTensor.addEventListener('click', () => setActiveView('tensor'));
    btnViewPipeline.addEventListener('click', () => setActiveView('pipeline'));

    setActiveView('cortex');
    console.log("View Switcher Initialized");
}

function setActiveView(mode) {
    btnViewCortex.classList.remove('active');
    btnViewTensor.classList.remove('active');
    btnViewPipeline.classList.remove('active');

    visualContent.classList.remove('hidden');
    didacticPanel.style.display = 'none';
    cortexCanvas.classList.add('hidden');
    tensorCanvas.classList.add('hidden');

    if (mode === 'cortex') {
        btnViewCortex.classList.add('active');
        cortexCanvas.classList.remove('hidden');
        currentView = 'cortex';
        // Use requestAnimationFrame to ensure DOM is updated (class removed) before resize
        requestAnimationFrame(() => redrawCanvas());
    } else if (mode === 'tensor') {
        btnViewTensor.classList.add('active');
        tensorCanvas.classList.remove('hidden');
        currentView = 'tensor';
        requestAnimationFrame(() => redrawCanvas());
    } else if (mode === 'pipeline') {
        btnViewPipeline.classList.add('active');
        visualContent.classList.add('hidden');
        didacticPanel.style.display = 'block';
    }
}

function redrawCanvas() {
    if (currentView === 'cortex') {
        resizeCanvas(cortexCanvas);
        if (currentState.lattice && currentState.lattice.length > 0) {
            renderVisualCortex(currentState.lattice, currentState.metrics.temperature);
        }
    } else if (currentView === 'tensor') {
        resizeCanvas(tensorCanvas);
        if (currentState.tensor_field && currentState.tensor_field.length > 0) {
            renderMetricTensor(currentState.tensor_field, currentState.metrics.gravity);
        }
    }
}

// --- Core Logic ---

async function processIntent() {
    const intent = intentInput.value.trim();
    if (!intent) return;

    // UI Feedback
    processBtn.disabled = true;
    processBtn.innerText = "PROCESSING...";
    manifestoOutput.innerText = "> Injecting Intent into Omega Kernel...\n> Warping Metric...\n> Collapsing Field...";

    try {
        const response = await fetch(`${API_BASE}/intent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ intent: intent, steps: 300 })
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        updateUI(data);

    } catch (error) {
        manifestoOutput.innerText = `ERROR: ${error.message}`;
    } finally {
        processBtn.disabled = false;
        processBtn.innerText = "INJECT INTENT";
    }
}

function updateUI(data) {
    // Update Global State
    if (data.didactic) {
        currentState.lattice = data.didactic.lattice || [];
        currentState.tensor_field = data.didactic.tensor_field || [];
        currentState.metrics.temperature = data.metrics.temperature || 0.5;
        currentState.metrics.gravity = data.didactic.gravity || 1.0;
    }

    // 1. Update Manifesto
    manifestoOutput.innerText = data.manifesto;

    // 2. Update Metrics
    updateMetric(valCoherence, barCoherence, data.metrics.coherence);
    updateMetric(valTension, barTension, data.metrics.tension);
    valDensity.innerText = data.metrics.logic_density.toFixed(4);
    valEnergy.innerText = data.metrics.energy.toFixed(4);

    // Update Energy Graph
    renderEnergyGraph(data.metrics.energy);

    // 3. Update Dipoles
    renderDipoles(data.dipoles);

    // 4. Refresh Taxonomy (Archivist)
    fetchTaxonomy();

    // 5. Update Didactic Layer (Expert Mode)
    if (data.didactic) {
        renderDidactic(data.didactic, data.metrics);

        // 6. Update Visual Cortex (Entropy Visualization)
        redrawCanvas();

        // Update HUD
        if (hudEntropy) hudEntropy.innerText = data.didactic.entropy.toFixed(3);
        if (hudGravity) hudGravity.innerText = data.didactic.gravity.toFixed(3);
    }
}

function updateMetric(valEl, barEl, value) {
    valEl.innerText = value.toFixed(4);
    barEl.style.width = `${Math.min(value * 100, 100)}%`;
}

function renderDipoles(dipoles) {
    dipoles = dipoles || [];
    dipoleContainer.innerHTML = '';

    if (dipoles.length === 0) {
        dipoleContainer.innerHTML = '<div class="empty-state">No strong dipoles detected.</div>';
        return;
    }

    dipoles.forEach(([concept, charge]) => {
        const tag = document.createElement('div');
        tag.className = `dipole-tag ${charge > 0 ? 'positive' : 'negative'}`;
        tag.innerText = `${concept} (${charge > 0 ? '+' : ''}${charge})`;
        dipoleContainer.appendChild(tag);
    });
}


function renderDidactic(info, metrics) {
    const panel = document.getElementById('didacticPanel');
    const content = document.getElementById('didacticContent');
    // panel.style.display = 'block'; // REMOVED: Do not force open. Let user toggle.
    content.innerHTML = '';

    // Create Terminal Feed Container
    const terminalFeed = document.createElement('div');
    terminalFeed.className = 'terminal-feed';
    content.appendChild(terminalFeed);

    // Render Timeline if available
    if (info.timeline && info.timeline.length > 0) {
        const header = document.createElement('div');
        header.className = 'terminal-header';
        header.innerHTML = '<strong>🔄 D-ND → Physics Translation Pipeline</strong>';
        terminalFeed.appendChild(header);

        info.timeline.forEach((event, idx) => {
            const line = document.createElement('div');
            line.className = 'terminal-line';

            let lineHTML = `
                <span class="step-icon">${event.icon || '•'}</span>
                <span class="step-label">[${event.step}]</span>
                <div class="step-translation">
                    <span class="d-nd-concept">${event.d_nd}</span>
                    <span class="arrow">→</span>
                    <span class="physics-concept">${event.physics}</span>
                </div>
                <div class="step-detail">${event.detail}</div>
            `;

            // Add data if available (e.g., dipole list)
            if (event.data && event.data.length > 0) {
                lineHTML += `<div class="step-data">${event.data.join(', ')}</div>`;
            }

            line.innerHTML = lineHTML;
            terminalFeed.appendChild(line);

            // Animate appearance
            setTimeout(() => line.classList.add('visible'), idx * 80);
        });
    }

    // Create 2-column layout for Rosetta Stone + Metrics
    const bottomGrid = document.createElement('div');
    bottomGrid.className = 'didactic-bottom-grid';
    content.appendChild(bottomGrid);

    // Add Rosetta Stone section if available
    if (info.rosetta_stone) {
        const rosetta = document.createElement('div');
        rosetta.className = 'rosetta-stone';
        rosetta.innerHTML = '<div class="rosetta-header">📖 Rosetta Stone (D-ND ↔ Physics)</div>';

        const table = document.createElement('div');
        table.className = 'rosetta-table';
        Object.entries(info.rosetta_stone).forEach(([dnd, physics]) => {
            const row = document.createElement('div');
            row.className = 'rosetta-row';
            row.innerHTML = `
                <span class="rosetta-dnd">${dnd}</span>
                <span class="rosetta-arrow">↔</span>
                <span class="rosetta-physics">${physics}</span>
            `;
            table.appendChild(row);
        });
        rosetta.appendChild(table);
        bottomGrid.appendChild(rosetta);
    }

    // Add Metrics Card (Compact View)
    if (metrics) {
        const metricsCard = document.createElement('div');
        metricsCard.className = 'didactic-metrics-card';
        metricsCard.innerHTML = `
            <div class="rosetta-header">📊 Snapshot Metrics</div>
            <div class="didactic-metrics-grid">
                <div class="didactic-metric">
                    <label>Coherence</label>
                    <div class="value">${metrics.coherence.toFixed(4)}</div>
                </div>
                <div class="didactic-metric">
                    <label>Tension</label>
                    <div class="value">${metrics.tension.toFixed(4)}</div>
                </div>
                <div class="didactic-metric">
                    <label>Logic Density</label>
                    <div class="value">${metrics.logic_density.toFixed(4)}</div>
                </div>
                <div class="didactic-metric">
                    <label>Energy</label>
                    <div class="value">${metrics.energy.toFixed(4)}</div>
                </div>
            </div>
        `;
        bottomGrid.appendChild(metricsCard);
    }

    // Legacy info (backward compatibility)
    if (info.gravity_info && !info.timeline) {
        const div = document.createElement('div');
        div.className = 'didactic-item';
        div.innerHTML = `
            <span class="didactic-label">vE_Scultore [Dynamic Gravity]</span>
            Detected semantic mass <strong>"${info.gravity_info.source}"</strong>. 
            Warped spacetime metric to density <strong>${info.gravity_info.gravity}</strong>.
        `;
        content.appendChild(div);
    }
}


async function fetchTaxonomy() {
    try {
        const response = await fetch(`${API_BASE}/state`);
        const data = await response.json();

        taxonomyList.innerHTML = '';
        const taxonomy = data.taxonomy || {};

        Object.entries(taxonomy).forEach(([concept, stats]) => {
            const item = document.createElement('div');
            item.className = 'taxonomy-item';
            item.innerHTML = `
                <div class="concept">${concept.toUpperCase()}</div>
                <div class="count">Count: ${stats.count} | Q: ${stats.avg_charge.toFixed(2)}</div>
            `;
            taxonomyList.appendChild(item);
        });

    } catch (e) {
        console.error("Failed to fetch taxonomy", e);
    }
}

async function systemReset() {
    if (!confirm("WARNING: This will wipe all System Memory and Taxonomy. Continue?")) return;

    const manifestoPre = document.getElementById("manifestoOutput");
    manifestoPre.textContent += "\n>> INITIATING SYSTEM RESET... ";

    try {
        await fetch('/api/reset', { method: 'POST' });
        manifestoPre.textContent += "SUCCESS.\n";

        // Clear UI
        document.getElementById('dipoleContainer').innerHTML = '<div class="empty-state">System Memory Cleared.</div>';
        document.getElementById('taxonomyList').innerHTML = '';
        energyHistory = []; // Reset graph history
        renderEnergyGraph(0); // Clear graph

        // Reset metrics
        updateMetric(valCoherence, barCoherence, 0);
        updateMetric(valTension, barTension, 0);
        valDensity.innerText = "0.2000"; // Default
        valEnergy.innerText = "0.0000";

    } catch (e) {
        manifestoPre.textContent += `FAILED: ${e}\n`;
    }
}

// --- Energy Graph Logic ---
let energyHistory = [];

function initEnergyGraph() {
    const energyCard = document.getElementById('valEnergy').parentElement;
    if (!energyCard.querySelector('canvas')) {
        const canvas = document.createElement('canvas');
        canvas.id = 'energyGraph';

        // High DPI Scaling
        const dpr = window.devicePixelRatio || 1;
        const rect = energyCard.getBoundingClientRect();
        // We want it to fill the width of the card roughly, minus padding
        const width = rect.width - 32; // approx padding
        const height = 60;

        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        canvas.width = width * dpr;
        canvas.height = height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        canvas.style.marginTop = '0.5rem';
        energyCard.appendChild(canvas);
    }
}

function renderEnergyGraph(currentEnergy) {
    const canvas = document.getElementById('energyGraph');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    // Use CSS dimensions for drawing logic since we scaled the context
    const width = parseFloat(canvas.style.width);
    const height = parseFloat(canvas.style.height);

    energyHistory.push(currentEnergy);
    if (energyHistory.length > 50) energyHistory.shift();

    ctx.clearRect(0, 0, width, height);

    // Dynamic Scaling
    let min = Math.min(...energyHistory);
    let max = Math.max(...energyHistory);

    // Add some padding to the range so it's not flat if constant
    if (max === min) {
        max += 0.1;
        min -= 0.1;
    }

    const range = max - min;

    ctx.beginPath();
    ctx.strokeStyle = '#06b6d4'; // cyan
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';

    // Draw Gradient Area
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

    const step = width / (50 - 1); // 50 points

    // Build path
    energyHistory.forEach((val, index) => {
        const x = index * step;
        const normalized = (val - min) / range;
        // Invert Y because canvas 0 is top
        // We want higher energy (bad) to be higher? Or lower energy (ground state) to be lower?
        // Usually graphs show value going up.
        const y = height - (normalized * height);

        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();

    // Fill area
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
}

// --- Docs Logic ---
function toggleDocs() {
    if (!docsModal) return;
    if (docsModal.style.display === 'block') {
        docsModal.style.display = 'none';
    } else {
        docsModal.style.display = 'block';
        fetchDocsList();
        initDocsTabs();
    }
}

// Close modal when clicking outside
window.onclick = function (event) {
    if (event.target == docsModal) {
        docsModal.style.display = "none";
    }
}

async function fetchDocsList() {
    try {
        const response = await fetch('/api/docs');
        const data = await response.json();
        docsList.innerHTML = '';

        data.files.forEach(file => {
            const div = document.createElement('div');
            div.className = 'docs-file';
            div.textContent = file;
            div.onclick = () => loadDoc(file, div);
            docsList.appendChild(div);
        });
    } catch (error) {
        console.error('Error fetching docs list:', error);
        docsList.innerHTML = '<div style="color: var(--danger)">Error loading docs.</div>';
    }
}

async function loadDoc(filename, element) {
    // Highlight active
    document.querySelectorAll('.docs-file').forEach(el => el.classList.remove('active'));
    if (element) element.classList.add('active');

    try {
        const response = await fetch(`/api/docs/${filename}`);
        const data = await response.json();
        // Simple markdown rendering (headers and code blocks)
        let html = data.content
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/\*\*(.*)\*\*/gim, '<b>$1</b>')
            .replace(/\*(.*)\*/gim, '<i>$1</i>')
            .replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>')
            .replace(/\n/gim, '<br>');

        docsViewer.innerHTML = html;
    } catch (error) {
        console.error('Error loading doc:', error);
        docsViewer.innerHTML = '<div style="color: var(--danger)">Error loading document.</div>';
    }
}

async function loadDiagnostics() {
    const outEl = document.getElementById('docsDiagnosticsContent');
    if (!outEl) return;
    outEl.textContent = '// Loading diagnostics...';
    try {
        const response = await fetch('/api/state');
        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }
        const data = await response.json();
        outEl.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        console.error('Error loading diagnostics:', error);
        outEl.textContent = '// Error loading diagnostics: ' + (error.message || error);
    }
}

function initDocsTabs() {
    const tabs = document.querySelectorAll('.docs-tab');
    const panels = document.querySelectorAll('[data-docspanel]');
    if (!tabs.length || !panels.length) return;

    tabs.forEach((tab) => {
        tab.addEventListener('click', () => {
            const target = tab.getAttribute('data-docstab');
            if (!target) return;

            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            panels.forEach(panel => {
                if (panel.getAttribute('data-docspanel') === target) {
                    panel.style.display = 'block';
                } else {
                    panel.style.display = 'none';
                }
            });

            if (target === 'diagnostics') {
                loadDiagnostics();
            }
        });
    });
}

/**
 * Universal Resizable Panel Logic
 * Handles drag-to-resize, ghost zone collapse, and gutter drag-to-open.
 */
function initResizablePanel({ shell, resizer, gutter, side = 'left', minWidth = 200, maxWidth = 500, ghostThreshold = 220, onResize }) {
    if (!shell || !resizer) return;

    let startX = 0;
    let startWidth = 0;
    let isGutterDragging = false;

    // Helper to calculate width based on side
    const calculateWidth = (startW, dx) => {
        return side === 'left' ? startW + dx : startW - dx;
    };

    // --- RESIZER DRAG (Main Handle) ---
    function onMouseMoveResize(e) {
        const dx = e.clientX - startX;
        let newWidth = calculateWidth(startWidth, dx);

        // Ghost Zone Check -> Collapse
        if (newWidth < ghostThreshold) {
            shell.classList.add('collapsed');
            shell.style.width = '32px'; // Gutter width
            if (onResize) onResize();
            window.removeEventListener('mousemove', onMouseMoveResize);
            window.removeEventListener('mouseup', onMouseUpResize);
            return;
        }

        if (shell.classList.contains('collapsed')) {
            shell.classList.remove('collapsed');
        }

        // Clamp
        if (newWidth < minWidth) newWidth = minWidth;
        if (newWidth > maxWidth) newWidth = maxWidth;
        
        shell.style.width = `${newWidth}px`;
        if (onResize) onResize();
    }

    function onMouseUpResize() {
        window.removeEventListener('mousemove', onMouseMoveResize);
        window.removeEventListener('mouseup', onMouseUpResize);
    }

    resizer.addEventListener('mousedown', (e) => {
        e.preventDefault();
        // If collapsed, open to default before drag starts
        if (shell.classList.contains('collapsed')) {
            shell.classList.remove('collapsed');
            shell.style.width = `${minWidth + 50}px`;
        }

        startX = e.clientX;
        startWidth = shell.getBoundingClientRect().width;
        window.addEventListener('mousemove', onMouseMoveResize);
        window.addEventListener('mouseup', onMouseUpResize);
    });

    // --- GUTTER DRAG (Afferra e Apri) ---
    if (gutter) {
        gutter.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const gutterStartX = e.clientX;
            let gutterStartWidth = shell.getBoundingClientRect().width;
            isGutterDragging = false;

            function onMouseMoveGutter(ev) {
                const dx = ev.clientX - gutterStartX;

                // Threshold to detect drag vs click
                if (!isGutterDragging && Math.abs(dx) > 3) {
                    isGutterDragging = true;
                }
                if (!isGutterDragging) return;

                // If opening from collapsed state
                if (shell.classList.contains('collapsed')) {
                    shell.classList.remove('collapsed');
                    // Don't jump to minWidth immediately, allow smooth drag from 32px
                }

                let newWidth = calculateWidth(gutterStartWidth, dx);

                // Check if dragging back into Ghost Zone -> Re-collapse
                if (newWidth < ghostThreshold) {
                    shell.classList.add('collapsed');
                    shell.style.width = '32px';
                    if (onResize) onResize();
                    // Don't stop listening, user might drag back out
                    return; 
                } else {
                    // If we are out of ghost zone, ensure collapsed class is removed
                    shell.classList.remove('collapsed');
                }

                // Clamp max only during drag (allow < minWidth for smooth opening)
                if (newWidth > maxWidth) newWidth = maxWidth;
                
                // Visual feedback: ensure we don't go below gutter width visually if not collapsed
                if (newWidth < 32) newWidth = 32;

                shell.style.width = `${newWidth}px`;
                if (onResize) onResize();
            }

            function onMouseUpGutter() {
                window.removeEventListener('mousemove', onMouseMoveGutter);
                window.removeEventListener('mouseup', onMouseUpGutter);

                // If it was just a click (no drag), toggle open to default
                if (!isGutterDragging) {
                    shell.classList.remove('collapsed');
                    shell.style.width = `${minWidth + 60}px`; // Default open width
                } else {
                    // If drag ended, check if we are below minWidth but above ghost
                    // Snap to minWidth to ensure content visibility
                    const currentW = shell.getBoundingClientRect().width;
                    if (!shell.classList.contains('collapsed') && currentW < minWidth) {
                        shell.style.width = `${minWidth}px`;
                    }
                }
                if (onResize) onResize();
            }

            window.addEventListener('mousemove', onMouseMoveGutter);
            window.addEventListener('mouseup', onMouseUpGutter);
        });
    }
}

function initDocsResizer() {
    // Wrapper for the Docs Modal using the universal function
    if (!docsResizer || !docsList) return;
    const shell = docsList.parentElement; // docsShell
    
    initResizablePanel({
        shell: shell,
        resizer: docsResizer,
        gutter: docsSidebarGutter,
        side: 'left',
        minWidth: 160,
        maxWidth: 480,
        ghostThreshold: 140 // Slightly lower for docs
    });
}

function initHomePanels() {
    // Left Panel (Control Matrix)
    const leftShell = document.getElementById('leftShell');
    const leftResizer = document.getElementById('leftResizer');
    const leftGutter = document.getElementById('leftGutter');

    initResizablePanel({
        shell: leftShell,
        resizer: leftResizer,
        gutter: leftGutter,
        side: 'left',
        minWidth: 250,
        maxWidth: 500,
        ghostThreshold: 200,
        onResize: () => {
            requestAnimationFrame(redrawCanvas);
        }
    });

    // Right Panel (Output Manifestation)
    const rightShell = document.getElementById('rightShell');
    const rightResizer = document.getElementById('rightResizer');
    const rightGutter = document.getElementById('rightGutter');

    initResizablePanel({
        shell: rightShell,
        resizer: rightResizer,
        gutter: rightGutter,
        side: 'right',
        minWidth: 300,
        maxWidth: 600,
        ghostThreshold: 250,
        onResize: () => {
            requestAnimationFrame(redrawCanvas);
        }
    });
    
    // Window resize handler
    window.addEventListener('resize', () => {
        requestAnimationFrame(redrawCanvas);
    });
}

function initDocsDrag() {
    if (!docsModal) return;
    const header = docsModal.querySelector('.modal-header');
    const content = docsModal.querySelector('.modal-content');
    if (!header || !content) return;

    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startTop = 0;
    let startLeft = 0;

    header.addEventListener('mousedown', (e) => {
        // solo tasto sinistro
        if (e.button !== 0) return;
        e.preventDefault();

        const rect = content.getBoundingClientRect();
        // passa a posizionamento esplicito alla prima drag
        content.style.position = 'fixed';
        content.style.margin = '0';
        content.style.top = `${rect.top}px`;
        content.style.left = `${rect.left}px`;

        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        startTop = rect.top;
        startLeft = rect.left;

        function onMouseMove(ev) {
            if (!isDragging) return;
            const dx = ev.clientX - startX;
            const dy = ev.clientY - startY;
            content.style.top = `${startTop + dy}px`;
            content.style.left = `${startLeft + dx}px`;
        }

        function onMouseUp() {
            isDragging = false;
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        }

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    });
}

// --- Protocols ---

async function runProtocol(protocolName) {
    intentInput.value = "";
    let intent = "";

    switch (protocolName) {
        case 'CHAOS':
            intent = "Generate absolute chaos and entropy";
            break;
        case 'ORDER':
            intent = "Restore perfect order and symmetry";
            break;
        case 'GENESIS':
            intent = "Synthesize a complex structure from balanced forces";
            break;
    }

    intentInput.value = intent;
    processIntent();
}

 // Initialize
window.onload = function () {
    initViewSwitcher();
    initEnergyGraph();
    bindTooltips();
    initDocsResizer();
    initDocsDrag();
    initHomePanels();

    // Init canvases resolution
    resizeCanvas(cortexCanvas);
    resizeCanvas(tensorCanvas);
    window.addEventListener('resize', () => {
        resizeCanvas(cortexCanvas);
        resizeCanvas(tensorCanvas);
    });

    // Fetch initial state to render immediately
    fetch(API_BASE + '/state')
        .then(response => response.json())
        .then(data => {
            console.log("Initial State Loaded:", data);
            
            // Update Global State
            if (data.lattice) currentState.lattice = data.lattice;
            else {
                // Dummy data
                currentState.lattice = Array.from({ length: 64 }, (_, i) => ({
                    id: i,
                    x: i % 8,
                    y: Math.floor(i / 8),
                    spin: Math.random() > 0.5 ? 1 : -1,
                    stability: 0.1
                }));
            }

            if (data.tensor_field) currentState.tensor_field = data.tensor_field;
            else {
                // Dummy tensor
                const size = 8;
                currentState.tensor_field = Array(size).fill(0).map(() => Array(size).fill(0.1));
            }
            
            // Initial Draw
            redrawCanvas();
        })
        .catch(err => console.error("Failed to load initial state:", err));
};

function resizeCanvas(canvas) {
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
    }
}

function renderVisualCortex(nodes, temp) {
    const canvas = document.getElementById('cortexCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Determine Phase based on temp (Logic from React App)
    let phase = 'IDLE';
    if (temp > 1.5) phase = 'PERTURBATION';
    else if (temp > 0.5) phase = 'ANNEALING';
    else if (temp > 0) phase = 'CRYSTALLIZED';

    // Clear with Deep Void
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);

    // Layout Logic (Match React: cellSize = width / gridSize)
    // Use ceil to ensure grid covers everything if not perfect square
    const gridSize = Math.ceil(Math.sqrt(nodes.length)); 
    const cellSize = width / gridSize;
    const radius = cellSize * 0.35;

    // Draw Grid Lines (Subtle)
    ctx.strokeStyle = 'rgba(0, 243, 255, 0.03)';
    ctx.lineWidth = 0.5;
    // Draw enough lines to cover height even if cells are large
    const rows = Math.ceil(height / cellSize);
    
    for (let i = 0; i <= gridSize; i++) {
        ctx.beginPath();
        ctx.moveTo(i * cellSize, 0);
        ctx.lineTo(i * cellSize, height);
        ctx.stroke();
    }
    for (let i = 0; i <= rows; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * cellSize);
        ctx.lineTo(width, i * cellSize);
        ctx.stroke();
    }

    // Draw Nodes
    nodes.forEach(node => {
        const px = node.x * cellSize + cellSize / 2;
        const py = node.y * cellSize + cellSize / 2;
        const alpha = 0.2 + (node.stability * 0.8);

        if (node.spin === 1) {
            // Magenta (Chaos)
            ctx.fillStyle = `rgba(255, 0, 255, ${alpha})`;
            ctx.shadowColor = 'rgba(255, 0, 255, 0.5)';
        } else {
            // Cyan (Order)
            ctx.fillStyle = `rgba(0, 243, 255, ${alpha})`;
            ctx.shadowColor = 'rgba(0, 243, 255, 0.5)';
        }

        ctx.shadowBlur = node.stability * 10;
        ctx.beginPath();
        ctx.arc(px, py, radius * (0.8 + node.stability * 0.4), 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Connections (Coupling)
        if (node.stability > 0.4) {
            const rightIdx = node.id + 1;
            const downIdx = node.id + gridSize;

            // Helper for drawing links
            const drawLink = (tx, ty) => {
                ctx.strokeStyle = node.spin === 1 ? 'rgba(255, 0, 255, 0.2)' : 'rgba(0, 243, 255, 0.2)';
                ctx.lineWidth = node.stability;
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(tx, ty);
                ctx.stroke();
            };

            if (node.x < gridSize - 1 && nodes[rightIdx] && nodes[rightIdx].spin === node.spin) {
                drawLink(nodes[rightIdx].x * cellSize + cellSize / 2, nodes[rightIdx].y * cellSize + cellSize / 2);
            }
            if (node.y < gridSize - 1 && nodes[downIdx] && nodes[downIdx].spin === node.spin) {
                drawLink(nodes[downIdx].x * cellSize + cellSize / 2, nodes[downIdx].y * cellSize + cellSize / 2);
            }
        }
    });

    // Horizon Line
    const horizonY = height * 0.5;
    const gradient = ctx.createLinearGradient(0, horizonY, width, horizonY);
    gradient.addColorStop(0, 'rgba(0,0,0,0)');
    gradient.addColorStop(0.5, 'rgba(0, 243, 255, 0.5)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 10]);
    ctx.beginPath();
    ctx.moveTo(0, horizonY);
    ctx.lineTo(width, horizonY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Overlay Text HUD (Match React)
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#1f2937';
    ctx.fillText(`MATRIX_DIM: ${nodes.length}`, 10, 20);
    ctx.fillStyle = phase === 'ANNEALING' ? '#ff00ff' : '#00f3ff';
    ctx.fillText(`STATUS: ${phase}`, 10, 35);
}

function drawLink(ctx, x1, y1, x2, y2, spin, stability) {
    ctx.strokeStyle = spin === 1 ? 'rgba(255, 0, 255, 0.2)' : 'rgba(0, 243, 255, 0.2)';
    ctx.lineWidth = stability;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function renderMetricTensor(tensorMap, gravity) {
    const canvas = document.getElementById('tensorCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const gridSize = tensorMap.length;
    const cellSize = width / gridSize; // Match React: width-based

    ctx.clearRect(0, 0, width, height);

    // Draw Matrix Heatmap
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const value = tensorMap[y][x] || 0;
            const normalized = Math.min(1, value / (1.5 + gravity));

            let r, g, b;
            if (normalized < 0.5) {
                // Blue to Purple
                const t = normalized * 2;
                r = Math.floor(t * 100);
                g = 0;
                b = Math.floor(50 + t * 150);
            } else {
                // Purple to White/Cyan
                const t = (normalized - 0.5) * 2;
                r = Math.floor(100 + t * 155);
                g = Math.floor(t * 255);
                b = 200;
            }

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            const gap = 2;
            ctx.fillRect(x * cellSize + gap, y * cellSize + gap, cellSize - gap * 2, cellSize - gap * 2);

            if (cellSize > 30) {
                ctx.fillStyle = 'rgba(255,255,255,0.4)';
                ctx.font = '10px JetBrains Mono';
                ctx.fillText(value.toFixed(2), x * cellSize + 8, y * cellSize + 16);
            }
        }
    }

    // Warp Grid Overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let y = 0; y <= gridSize; y++) {
        ctx.moveTo(0, y * cellSize);
        ctx.lineTo(width, y * cellSize);
    }
    for (let x = 0; x <= gridSize; x++) {
        ctx.moveTo(x * cellSize, 0);
        ctx.lineTo(x * cellSize, height);
    }
    ctx.stroke();
}

// --- Tooltip Engine (Semantic Hints) ---
let tooltipLayer = null;

function ensureTooltipLayer() {
    if (!tooltipLayer) {
        const el = document.createElement('div');
        el.className = 'tooltip-layer';
        el.setAttribute('data-visible', 'false');
        document.body.appendChild(el);
        tooltipLayer = el;
    }
    return tooltipLayer;
}

function hideTooltip() {
    if (!tooltipLayer) return;
    tooltipLayer.setAttribute('data-visible', 'false');
}

function showTooltip(target, text) {
    const layer = ensureTooltipLayer();
    if (!target || !text) {
        hideTooltip();
        return;
    }

    // Debug: log which element is showing a tooltip
    console.log('[Tooltip] show for', target.id || target.className || target.tagName, 'text:', text);

    // Escape basic HTML to avoid breaking layout
    const safe = String(text)
        .replace(/&/g, '&')
        .replace(/</g, '<')
        .replace(/>/g, '>');

    layer.innerHTML = safe;

    const rect = target.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const GAP = 12;
    const ESTIMATED_WIDTH = 280; // ~18rem
    const halfWidth = ESTIMATED_WIDTH / 2;

    // Position tooltip BELOW the element, then clamp horizontally towards center if near edges
    let top = rect.bottom + window.scrollY + 8;
    let centerX = rect.left + rect.width / 2 + window.scrollX;

    let left = centerX;
    const minLeft = GAP + halfWidth;
    const maxLeft = viewportWidth - GAP - halfWidth;
    if (left < minLeft) left = minLeft;
    if (left > maxLeft) left = maxLeft;

    layer.style.top = `${top}px`;
    layer.style.left = `${left}px`;
    layer.style.transform = 'translateX(-50%)';
    layer.setAttribute('data-visible', 'true');
}

function bindTooltips() {
    ensureTooltipLayer();
    const tooltipTargets = document.querySelectorAll('[data-tooltip]');
    console.log('[Tooltip] binding', tooltipTargets.length, 'elements');

    tooltipTargets.forEach((el) => {
        const t = el.getAttribute('data-tooltip');
        if (!t) return;

        el.addEventListener('mouseenter', () => showTooltip(el, t));
        el.addEventListener('mouseleave', hideTooltip);
        el.addEventListener('focus', () => showTooltip(el, t));
        el.addEventListener('blur', hideTooltip);
    });

    window.addEventListener('scroll', hideTooltip, true);
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') hideTooltip();
    });

    document.addEventListener('click', hideTooltip);
}
