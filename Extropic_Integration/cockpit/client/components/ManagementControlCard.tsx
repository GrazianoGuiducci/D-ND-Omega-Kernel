
import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import Card from './Card';
import { ChartBarIcon } from './icons/ChartBarIcon';
import { MonthlyData } from '../types';
import { Theme, THEMES, getSemanticColor } from '../themes';

interface ManagementControlCardProps {
    data: MonthlyData[];
    onAction?: () => void;
    currentTheme: Theme;
    onToggleSize?: () => void;
    onDelete?: () => void;
    isFullWidth?: boolean;
}

const formatCurrency = (value: number) => `€${(value / 1000).toFixed(0)}k`;

const CustomTooltip = ({ active, payload, label, currentTheme }: any) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        const revenueColor = getSemanticColor('revenue', currentTheme);
        const costsColor = getSemanticColor('costs', currentTheme);
        const profitColor = getSemanticColor('profit', currentTheme);

        return (
            <div className="backdrop-blur-xl border p-0 rounded-sm shadow-[0_0_30px_rgba(var(--col-primary-rgb),0.2)] text-xs font-mono min-w-[220px] overflow-hidden relative"
                style={{ backgroundColor: 'var(--bg-base)', borderColor: 'rgba(var(--col-primary-rgb), 0.5)' }}>
                {/* Decorative header line */}
                <div className="h-1 w-full bg-gradient-to-r from-neon-cyan via-neon-purple to-transparent"></div>

                <div className="p-3 relative z-10">
                    <p className="font-bold text-sm mb-3 border-b pb-2 tracking-wider uppercase flex items-center justify-between font-sans"
                        style={{ color: 'var(--col-primary)', borderColor: 'var(--col-muted)' }}>
                        <span className="px-2 py-0.5 rounded" style={{ backgroundColor: 'rgba(var(--col-primary-rgb), 0.1)', color: 'var(--col-primary)' }}>{label}</span>
                        <span className="text-[10px] font-mono" style={{ color: 'var(--text-sub)' }}>P&L DATA</span>
                    </p>

                    <div className="space-y-3">
                        {/* Revenue */}
                        <div className="flex justify-between items-center group">
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-sm shadow-[0_0_5px]" style={{ backgroundColor: revenueColor }}></div>
                                <span className="uppercase tracking-wide text-[10px] transition-colors group-hover:text-white" style={{ color: 'var(--text-sub)' }}>Revenue</span>
                            </div>
                            <span className="font-bold font-mono text-sm" style={{ color: 'var(--text-main)' }}>€{data.revenue.toLocaleString()}</span>
                        </div>

                        {/* Costs */}
                        <div className="flex justify-between items-center group">
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-sm shadow-[0_0_5px]" style={{ backgroundColor: costsColor }}></div>
                                <span className="uppercase tracking-wide text-[10px] transition-colors group-hover:text-white" style={{ color: 'var(--text-sub)' }}>Costs</span>
                            </div>
                            <span className="font-mono text-sm" style={{ color: 'var(--text-sub)' }}>€{data.costs.toLocaleString()}</span>
                        </div>

                        {/* Profit Badge */}
                        <div className="mt-2 pt-2 border-t border-dashed flex justify-between items-center p-2 rounded border"
                            style={{ borderColor: 'var(--col-muted)', backgroundColor: 'rgba(255,255,255,0.05)' }}>
                            <span className="font-bold tracking-widest uppercase text-[10px]" style={{ color: profitColor }}>Net Profit</span>
                            <span className="font-bold text-base font-mono drop-shadow-md" style={{ color: 'var(--text-main)' }}>€{data.profit.toLocaleString()}</span>
                        </div>
                    </div>
                </div>

                {/* Background Grid Pattern */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[length:10px_10px] pointer-events-none z-0"></div>
            </div>
        );
    }
    return null;
};


const ManagementControlCard: React.FC<ManagementControlCardProps> = ({ data, onAction, currentTheme, onToggleSize, onDelete, isFullWidth }) => {
    const primaryRgb = THEMES[currentTheme].vars['--col-primary-rgb'];

    const revenueColor = getSemanticColor('revenue', currentTheme);
    const costsColor = getSemanticColor('costs', currentTheme);
    const profitColor = getSemanticColor('profit', currentTheme);

    // --- INTERACTIVE LEGEND STATE ---
    const [hiddenKeys, setHiddenKeys] = useState<string[]>([]);

    const toggleKeyVisibility = (key: string) => {
        setHiddenKeys(prev =>
            prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
        );
    };

    // Custom Interactive Legend Renderer
    const renderLegend = (props: any) => {
        const { payload } = props;

        return (
            <div className="flex justify-center gap-4 mt-2 pt-2 border-t border-dashed select-none"
                style={{ borderColor: 'rgba(var(--col-muted-rgb), 0.5)' }}>
                {payload.map((entry: any, index: number) => {
                    const isHidden = hiddenKeys.includes(entry.value);
                    let color = entry.color;
                    if (entry.value === 'Revenue') color = revenueColor;
                    if (entry.value === 'Costs') color = costsColor;
                    if (entry.value === 'Profit') color = profitColor;

                    return (
                        <button
                            key={`legend-${index}`}
                            onClick={() => toggleKeyVisibility(entry.value)}
                            className={`
                                flex items-center gap-2 px-3 py-1.5 rounded-md border transition-all duration-300
                                ${isHidden
                                    ? 'border-transparent opacity-50'
                                    : 'shadow-[0_2px_10px_rgba(0,0,0,0.2)] hover:shadow-[0_2px_15px_rgba(0,0,0,0.3)]'
                                }
                            `}
                            style={{
                                backgroundColor: isHidden ? 'rgba(var(--col-muted-rgb), 0.5)' : 'rgba(255,255,255,0.05)',
                                borderColor: isHidden ? 'transparent' : 'rgba(255,255,255,0.1)',
                                color: isHidden ? 'var(--text-sub)' : 'var(--text-main)'
                            }}
                        >
                            <div
                                className={`w-2 h-2 rounded-sm transition-all duration-300 ${isHidden ? 'scale-75 grayscale' : 'scale-100 rotate-45'}`}
                                style={{ backgroundColor: color, boxShadow: isHidden ? 'none' : `0 0 8px ${color}` }}
                            />
                            <span className={`text-[10px] font-bold font-sans uppercase tracking-wider ${isHidden ? 'line-through decoration-slate-600' : ''}`}>
                                {entry.value}
                            </span>
                        </button>
                    );
                })}
            </div>
        );
    };

    return (
        <Card
            title="Management Control"
            icon={<ChartBarIcon className="h-6 w-6" />}
            onAnalysisClick={onAction}
            onToggleSize={onToggleSize}
            onDelete={onDelete}
            isFullWidth={isFullWidth}
        >
            {/* STRICT LAYOUT CONTAINER FOR RECHARTS */}
            <div className="h-full w-full min-h-[200px] flex flex-col">
                <div className="flex-grow w-full h-0 relative">
                    <div className="absolute inset-0">
                        <ResponsiveContainer width="99%" height="100%" debounce={50}>
                            <BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }} barGap={2}>
                                <defs>
                                    <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={revenueColor} stopOpacity={1} />
                                        <stop offset="100%" stopColor={revenueColor} stopOpacity={0.4} />
                                    </linearGradient>
                                    <linearGradient id="costsGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={costsColor} stopOpacity={1} />
                                        <stop offset="100%" stopColor={costsColor} stopOpacity={0.4} />
                                    </linearGradient>
                                    <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={profitColor} stopOpacity={1} />
                                        <stop offset="100%" stopColor={profitColor} stopOpacity={0.4} />
                                    </linearGradient>
                                </defs>

                                <CartesianGrid strokeDasharray="3 3" stroke={`rgba(${primaryRgb}, 0.05)`} vertical={false} />
                                <XAxis
                                    dataKey="month"
                                    stroke={`rgba(${primaryRgb}, 0.5)`}
                                    fontSize={10}
                                    tickLine={false}
                                    axisLine={false}
                                    fontFamily="JetBrains Mono"
                                />
                                <YAxis
                                    stroke={`rgba(${primaryRgb}, 0.5)`}
                                    fontSize={10}
                                    tickFormatter={formatCurrency}
                                    tickLine={false}
                                    axisLine={false}
                                    fontFamily="JetBrains Mono"
                                />
                                <Tooltip
                                    content={<CustomTooltip currentTheme={currentTheme} />}
                                    cursor={{ fill: `rgba(${primaryRgb}, 0.03)` }}
                                    wrapperStyle={{ outline: 'none', zIndex: 1000 }}
                                    allowEscapeViewBox={{ x: true, y: true }}
                                />
                                <Legend content={renderLegend} />

                                <Bar
                                    dataKey="revenue"
                                    fill="url(#revenueGradient)"
                                    name="Revenue"
                                    radius={[3, 3, 0, 0]}
                                    maxBarSize={isFullWidth ? 30 : 20}
                                    animationDuration={1500}
                                    hide={hiddenKeys.includes('Revenue')}
                                />
                                <Bar
                                    dataKey="costs"
                                    fill="url(#costsGradient)"
                                    name="Costs"
                                    radius={[3, 3, 0, 0]}
                                    maxBarSize={isFullWidth ? 30 : 20}
                                    animationDuration={1500}
                                    animationBegin={200}
                                    hide={hiddenKeys.includes('Costs')}
                                />
                                <Bar
                                    dataKey="profit"
                                    fill="url(#profitGradient)"
                                    name="Profit"
                                    radius={[3, 3, 0, 0]}
                                    maxBarSize={isFullWidth ? 30 : 20}
                                    animationDuration={1500}
                                    animationBegin={400}
                                    hide={hiddenKeys.includes('Profit')}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </Card>
    );
};

export default ManagementControlCard;
