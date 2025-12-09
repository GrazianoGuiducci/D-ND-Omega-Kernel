
import React from 'react';
import { SystemLog } from '../types';

interface SystemStatusFooterProps {
    logs: SystemLog[];
    kernelStatus: string;
}

const SystemStatusFooter: React.FC<SystemStatusFooterProps> = ({ logs, kernelStatus }) => {
    const latestLog = logs[0];

    return (
        <footer className="h-8 bg-black border-t border-white/10 flex items-center justify-between px-4 shrink-0 font-mono text-[10px] select-none z-50 relative">
            {/* LEFT: System Stream */}
            <div className="flex items-center gap-4 overflow-hidden flex-1">
                <div className="flex items-center gap-2 text-slate-500 shrink-0">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
                    <span className="uppercase tracking-widest font-bold">SYSTEM_STREAM</span>
                </div>
                
                <div className="h-4 w-px bg-white/10"></div>
                
                {latestLog && (
                    <div className="flex items-center gap-2 animate-slide-up-fade truncate">
                        <span className="text-slate-600">[{latestLog.timestamp}]</span>
                        <span className={`font-bold uppercase ${
                            latestLog.source === 'KERNEL' ? 'text-purple-400' :
                            latestLog.source === 'AI' ? 'text-neon-cyan' :
                            latestLog.source === 'USER' ? 'text-white' : 'text-slate-400'
                        }`}>
                            {latestLog.source}::
                        </span>
                        <span className={`${
                            latestLog.type === 'error' ? 'text-red-400' :
                            latestLog.type === 'success' ? 'text-green-400' :
                            'text-slate-300'
                        }`}>
                            {latestLog.message}
                        </span>
                    </div>
                )}
            </div>

            {/* RIGHT: Kernel Stats */}
            <div className="flex items-center gap-4 text-slate-600 hidden sm:flex">
                <div className="flex items-center gap-1">
                    <span>KERNEL_LATENCY:</span>
                    <span className="text-slate-300">14ms</span>
                </div>
                <div className="flex items-center gap-1">
                    <span>MEMORY_HEAP:</span>
                    <span className="text-slate-300">42%</span>
                </div>
                <div className="px-2 py-0.5 bg-white/5 rounded border border-white/5 text-slate-300 uppercase font-bold">
                    {kernelStatus}
                </div>
            </div>
        </footer>
    );
};

export default SystemStatusFooter;
