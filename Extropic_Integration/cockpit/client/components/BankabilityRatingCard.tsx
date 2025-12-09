
import React from 'react';
import { ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts';
import Card from './Card';
import { ScaleIcon } from './icons/ScaleIcon';
import { RatingData } from '../types';
import { Theme, THEMES, getSemanticColor } from '../themes';

interface BankabilityRatingCardProps {
    data: RatingData;
    onAction?: () => void;
    currentTheme: Theme;
    onToggleSize?: () => void;
    onDelete?: () => void;
    isFullWidth?: boolean;
}

const BankabilityRatingCard: React.FC<BankabilityRatingCardProps> = ({ data, onAction, currentTheme, onToggleSize, onDelete, isFullWidth }) => {
    const primaryRgb = THEMES[currentTheme].vars['--col-primary-rgb'];

    // Use Semantic Color for Score (Theme Aware)
    const scoreColor = getSemanticColor('score', currentTheme);

    const chartData = [{ name: 'Score', value: data.score, fill: scoreColor }];

    // Color logic for rating text - using theme color as base glow
    const getRatingColor = (r: string) => {
        if (r.startsWith('A')) return 'text-green-400 drop-shadow-[0_0_15px_rgba(74,222,128,0.6)]';
        if (r.startsWith('B')) return 'text-yellow-400 drop-shadow-[0_0_15px_rgba(250,204,21,0.6)]';
        return 'text-red-400 drop-shadow-[0_0_15px_rgba(248,113,113,0.6)]';
    }

    return (
        <Card
            title="Bankability Radar"
            icon={<ScaleIcon className="h-6 w-6" />}
            onAnalysisClick={onAction}
            onToggleSize={onToggleSize}
            onDelete={onDelete}
            isFullWidth={isFullWidth}
        >
            <div className="flex flex-col sm:flex-row items-center justify-center h-full gap-6 relative overflow-hidden">

                {/* LEFT: The Sonar Visualization - Rigorous Layout Fix */}
                <div className="w-full sm:w-1/2 h-60 sm:h-full relative flex-shrink-0 flex items-center justify-center min-h-[200px] min-w-0">

                    {/* 1. Static Grid Circles (Sonar Rings) */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="w-48 h-48 rounded-full border border-slate-800/30"></div>
                        <div className="absolute w-32 h-32 rounded-full border border-slate-800/30"></div>
                        <div className="absolute w-16 h-16 rounded-full border border-slate-800/30"></div>
                    </div>

                    {/* 2. Active Radar Sweep Animation */}
                    <div className="absolute w-48 h-48 rounded-full overflow-hidden opacity-20 pointer-events-none animate-radar-spin">
                        <div className="w-full h-full radar-sweep"></div>
                    </div>

                    {/* 3. The Main Chart - Secure Container */}
                    <div className="absolute inset-0 z-10">
                        <ResponsiveContainer width="100%" height="100%" debounce={50}>
                            <RadialBarChart
                                innerRadius="70%"
                                outerRadius="100%"
                                data={chartData}
                                startAngle={180}
                                endAngle={0}
                                barSize={20}
                            >
                                <PolarAngleAxis
                                    type="number"
                                    domain={[0, 100]}
                                    angleAxisId={0}
                                    tick={false}
                                />
                                <RadialBar
                                    background={{ fill: `rgba(${primaryRgb}, 0.05)` }}
                                    dataKey="value"
                                    cornerRadius={30}
                                />
                            </RadialBarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* 4. Center Score (Pulsating) */}
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/3 text-center z-20 pointer-events-none">
                        <div className="relative">
                            <div className={`text-5xl font-bold font-sans ${getRatingColor(data.rating)} block animate-breathing-glow`}>
                                {data.rating}
                            </div>
                        </div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-[0.2em] font-mono mt-2 block bg-black/40 px-2 rounded">
                            Score: {data.score}/100
                        </span>
                    </div>
                </div>

                {/* RIGHT: Metrics Breakdown */}
                <div className="w-full sm:w-1/2 flex flex-col justify-center gap-3 relative z-10 p-2 min-w-0">
                    {data.metrics.map((metric, idx) => (
                        <div
                            key={metric.label}
                            className="border rounded-sm p-3 relative overflow-hidden group transition-colors"
                            style={{
                                backgroundColor: 'rgba(var(--bg-surface-rgb), 0.3)',
                                borderColor: 'var(--col-muted)',
                                animationDelay: `${idx * 100}ms`
                            }}
                        >
                            <div
                                className="absolute bottom-0 left-0 h-[2px] transition-all duration-1000"
                                style={{ width: `${metric.score}%`, backgroundColor: 'var(--col-primary)' }}
                            ></div>

                            <div className="flex justify-between items-end pl-2">
                                <div>
                                    <p className="text-[9px] font-mono uppercase tracking-wide mb-0.5" style={{ color: 'var(--text-sub)' }}>Metric Analysis</p>
                                    <p className="text-xs font-bold font-sans uppercase tracking-wider group-hover:text-white" style={{ color: 'var(--text-sub)' }}>{metric.label}</p>
                                </div>
                                <p className="text-lg font-bold font-mono transition-colors drop-shadow-md" style={{ color: 'var(--text-main)' }}>{metric.value}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </Card>
    );
};

export default BankabilityRatingCard;
