
import React, { useEffect, useRef } from 'react';

interface MetricTensorVisProps {
    tensorMap: number[][]; // 8x8 Matrix
    gravity: number;
}

const MetricTensorVis: React.FC<MetricTensorVisProps> = ({ tensorMap, gravity }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !tensorMap || tensorMap.length === 0) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Resize
        const parent = canvas.parentElement;
        if (parent) {
            canvas.width = parent.clientWidth;
            canvas.height = parent.clientHeight;
        }

        const width = canvas.width;
        const height = canvas.height;
        const gridSize = tensorMap.length;
        const cellSize = width / gridSize;

        ctx.clearRect(0, 0, width, height);

        // Draw Matrix Heatmap
        for (let y = 0; y < gridSize; y++) {
            for (let x = 0; x < gridSize; x++) {
                const value = tensorMap[y][x] || 0;
                
                // Normalize value roughly for visualization
                // High gravity means we expect higher field values, so we scale visualization accordingly
                const normalized = Math.min(1, value / (1.5 + gravity));
                
                // Color mapping: 
                // Low = Deep Void (Dark Blue/Black)
                // Mid = Tensor Stress (Purple)
                // High = Singularity (White/Cyan)
                
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
                
                // Draw Cell with gap
                const gap = 2;
                ctx.fillRect(x * cellSize + gap, y * cellSize + gap, cellSize - (gap*2), cellSize - (gap*2));
                
                // Draw numeric value
                if (cellSize > 30) {
                    ctx.fillStyle = 'rgba(255,255,255,0.4)';
                    ctx.font = '10px JetBrains Mono';
                    ctx.fillText(value.toFixed(2), x * cellSize + 8, y * cellSize + 16);
                }
            }
        }
        
        // Overlay: "Warp" Grid lines to simulate gravity
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let y = 0; y <= gridSize; y++) {
            // Horizontal lines warp based on gravity center (simulated)
            ctx.moveTo(0, y * cellSize);
            ctx.lineTo(width, y * cellSize);
        }
        for (let x = 0; x <= gridSize; x++) {
            ctx.moveTo(x * cellSize, 0);
            ctx.lineTo(x * cellSize, height);
        }
        ctx.stroke();

    }, [tensorMap, gravity]);

    return (
        <div className="w-full h-full relative bg-[#050505] rounded-lg border border-[#1f1f1f] overflow-hidden">
            <canvas ref={canvasRef} className="block" />
            
            <div className="absolute top-3 left-3 flex gap-2">
                <div className="text-[9px] font-mono text-white bg-black/60 px-2 py-1 rounded border border-white/10 backdrop-blur">
                    TENSOR FIELD (g_uv)
                </div>
                {gravity > 2.0 && (
                    <div className="text-[9px] font-mono text-red-400 bg-red-900/20 px-2 py-1 rounded border border-red-500/30 animate-pulse">
                        HIGH CURVATURE
                    </div>
                )}
            </div>
            
            {/* Holographic Scanline */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0)_2px,rgba(0,0,0,0.4)_2px)] bg-[length:100%_4px] pointer-events-none opacity-30"></div>
        </div>
    );
};

export default MetricTensorVis;
