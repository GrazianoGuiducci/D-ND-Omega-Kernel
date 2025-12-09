
import React, { useRef, useEffect } from 'react';
import { LatticeNode } from '../types';

interface VisualCortexProps {
    nodes: LatticeNode[];
    phase: string;
}

const VisualCortex: React.FC<VisualCortexProps> = ({ nodes, phase }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Resize handling
        const parent = canvas.parentElement;
        if (parent) {
            canvas.width = parent.clientWidth;
            canvas.height = parent.clientHeight;
        }

        const width = canvas.width;
        const height = canvas.height;
        const gridSize = Math.sqrt(nodes.length);
        const cellSize = width / gridSize;
        const radius = cellSize * 0.35; // Slightly smaller for "Void" feel

        // Deep Void Background
        ctx.fillStyle = '#050505';
        ctx.fillRect(0, 0, width, height);
        
        // Draw Grid Lines (Subtle)
        ctx.strokeStyle = 'rgba(0, 243, 255, 0.03)';
        ctx.lineWidth = 0.5;
        for(let i=0; i<=gridSize; i++) {
            ctx.beginPath();
            ctx.moveTo(i*cellSize, 0);
            ctx.lineTo(i*cellSize, height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i*cellSize);
            ctx.lineTo(width, i*cellSize);
            ctx.stroke();
        }

        // Draw Nodes
        nodes.forEach(node => {
            const px = node.x * cellSize + cellSize / 2;
            const py = node.y * cellSize + cellSize / 2;

            const alpha = 0.2 + (node.stability * 0.8);

            // D-ND Colors: Cyan (Order/Logic) vs Magenta (Chaos/Entropy)
            if (node.spin === 1) {
                // Magenta/Chaos (Spin +1)
                ctx.fillStyle = `rgba(255, 0, 255, ${alpha})`; 
                ctx.shadowColor = 'rgba(255, 0, 255, 0.5)';
            } else {
                // Cyan/Order (Spin -1)
                ctx.fillStyle = `rgba(0, 243, 255, ${alpha})`;
                ctx.shadowColor = 'rgba(0, 243, 255, 0.5)';
            }
            
            ctx.shadowBlur = node.stability * 10;
            
            ctx.beginPath();
            ctx.arc(px, py, radius * (0.8 + node.stability * 0.4), 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0; // Reset

            // Draw connections (Coupling J) - Only draw if stable to save perf
            if (node.stability > 0.4) {
                const drawLink = (target: LatticeNode) => {
                     if (target && target.spin === node.spin) {
                        ctx.strokeStyle = node.spin === 1 ? 'rgba(255, 0, 255, 0.2)' : 'rgba(0, 243, 255, 0.2)';
                        ctx.lineWidth = node.stability;
                        ctx.beginPath();
                        ctx.moveTo(px, py);
                        ctx.lineTo(target.x * cellSize + cellSize / 2, target.y * cellSize + cellSize / 2);
                        ctx.stroke();
                     }
                };

                if (node.x < gridSize - 1) drawLink(nodes[node.id + 1]);
                if (node.y < gridSize - 1) drawLink(nodes[node.id + gridSize]);
            }
        });

        // Overlay Text HUD for Asset Tickers (Econophysics)
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '9px JetBrains Mono';
        
        nodes.forEach(node => {
            if (node.assetTicker) {
                const px = node.x * cellSize + cellSize / 2;
                const py = node.y * cellSize + cellSize / 2;
                
                // Draw pill background
                const textWidth = ctx.measureText(node.assetTicker).width + 6;
                ctx.fillStyle = 'rgba(0,0,0,0.8)';
                ctx.fillRect(px - textWidth/2, py - 6, textWidth, 12);
                
                // Draw Text
                ctx.fillStyle = '#ffffff';
                ctx.fillText(node.assetTicker, px, py);
            }
        });

        // Event Horizon Line (t=0)
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

        // Overlay Text HUD
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#1f2937';
        ctx.fillText(`MATRIX_DIM: ${nodes.length}`, 10, 20);
        ctx.fillStyle = phase === 'ANNEALING' ? '#ff00ff' : '#00f3ff';
        ctx.fillText(`STATUS: ${phase}`, 10, 35);

    }, [nodes, phase]);

    return (
        <div className="w-full h-full relative bg-[#050505] overflow-hidden rounded-lg border border-[#1f1f1f]">
            <canvas ref={canvasRef} className="block" />
            
            {/* Scanline Effect */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 bg-[length:100%_4px,3px_100%] opacity-20"></div>
        </div>
    );
};

export default VisualCortex;
