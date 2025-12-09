import React, { useRef, useEffect } from 'react';

interface EnergyGraphProps {
    data: number[];
    width?: number;
    height?: number;
}

const EnergyGraph: React.FC<EnergyGraphProps> = ({ data, width = 300, height = 60 }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // High DPI Scaling
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        // Use prop width/height if not constrained by parent, but usually we want 100%
        // We'll rely on CSS for display size and dpr for internal resolution
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, rect.height);

        if (data.length < 2) return;

        // Dynamic Scaling
        let min = Math.min(...data);
        let max = Math.max(...data);

        if (max === min) {
            max += 0.1;
            min -= 0.1;
        }

        const range = max - min;
        const step = rect.width / (data.length - 1);

        ctx.beginPath();
        ctx.strokeStyle = '#06b6d4'; // cyan-500
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';

        // Draw Gradient Area
        const gradient = ctx.createLinearGradient(0, 0, 0, rect.height);
        gradient.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
        gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

        data.forEach((val, index) => {
            const x = index * step;
            const normalized = (val - min) / range;
            // Invert Y because canvas 0 is top
            const y = rect.height - (normalized * rect.height);

            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();

        // Fill area
        ctx.lineTo(rect.width, rect.height);
        ctx.lineTo(0, rect.height);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

    }, [data, width, height]);

    return (
        <canvas
            ref={canvasRef}
            className="w-full h-full block"
            style={{ width: '100%', height: '100%' }}
        />
    );
};

export default EnergyGraph;
