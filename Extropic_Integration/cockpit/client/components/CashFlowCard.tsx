
import React, { useMemo } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, Area } from 'recharts';
import Card from './Card';
import { CurrencyDollarIcon } from './icons/CurrencyDollarIcon';
import { CashFlowDataPoint } from '../types';
import { Theme, THEMES, getSemanticColor } from '../themes';

interface CashFlowCardProps {
    data: CashFlowDataPoint[];
    onAction?: () => void;
    currentTheme: Theme;
    onToggleSize?: () => void;
    onDelete?: () => void;
    isFullWidth?: boolean;
}

const formatCurrency = (value: number) => `€${(value / 1000).toFixed(0)}k`;

// Custom tooltip to handle range explanation
const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        const dataPoint = payload[0].payload;
        const color = payload[0].stroke || payload[0].fill;
        const isForecast = dataPoint.type === 'Forecast';

        return (
            <div className="backdrop-blur-xl border p-0 rounded-sm shadow-[0_0_30px_rgba(var(--col-primary-rgb),0.2)] text-xs font-mono min-w-[200px] overflow-hidden"
                style={{ backgroundColor: 'var(--bg-base)', borderColor: 'rgba(var(--col-primary-rgb), 0.5)' }}>
                <div className="h-1 w-full" style={{ backgroundColor: color }}></div>
                <div className="p-4 relative">
                    <div className="flex justify-between items-center mb-3 pb-2 border-b" style={{ borderColor: 'var(--col-muted)' }}>
                        <span className="font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>{label}</span>
                        <span className={`px-1.5 py-0.5 rounded-[2px] text-[9px] font-bold uppercase tracking-widest border`}
                            style={{
                                backgroundColor: `${color}15`,
                                color: color,
                                borderColor: `${color}50`
                            }}
                        >
                            {dataPoint.type}
                        </span>
                    </div>

                    <div className="flex flex-col gap-1">
                        <span className="uppercase text-[10px] tracking-widest" style={{ color: 'var(--text-sub)' }}>Liquidity Balance</span>
                        <div className="flex items-baseline gap-1">
                            <span className="font-bold text-xl font-mono drop-shadow-md" style={{ color: color }}>
                                €{dataPoint.cash.toLocaleString()}
                            </span>
                        </div>
                    </div>

                    {/* Uncertainty Range Display for Forecast */}
                    {isForecast && dataPoint.upperBound && (
                        <div className="mt-3 pt-2 border-t text-[10px]" style={{ borderColor: 'var(--col-muted)' }}>
                            <div className="flex justify-between" style={{ color: 'var(--text-sub)' }}>
                                <span>Optimistic:</span>
                                <span className="text-green-400/80">€{Math.round(dataPoint.upperBound).toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between" style={{ color: 'var(--text-sub)' }}>
                                <span>Pessimistic:</span>
                                <span className="text-red-400/80">€{Math.round(dataPoint.lowerBound).toLocaleString()}</span>
                            </div>
                        </div>
                    )}

                    {/* Decorative Scanlines */}
                    <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(0,0,0,0)_2px,rgba(0,0,0,0.2)_2px)] bg-[length:100%_4px]"></div>
                </div>
            </div>
        );
    }
    return null;
};

const CashFlowCard: React.FC<CashFlowCardProps> = ({ data, onAction, currentTheme, onToggleSize, onDelete, isFullWidth }) => {
    const primaryRgb = THEMES[currentTheme].vars['--col-primary-rgb'];
    const cashColor = getSemanticColor('cash', currentTheme);

    // Transform data to add "Uncertainty Cone" logic
    const processedData = useMemo(() => {
        let forecastIndex = 0;
        return data.map((point) => {
            if (point.type === 'Past') return point;

            // Create artificial uncertainty that grows with time
            forecastIndex++;
            const uncertaintyFactor = forecastIndex * 0.05; // 5% variance growth per month
            const variance = point.cash * uncertaintyFactor;

            return {
                ...point,
                upperBound: point.cash + variance,
                lowerBound: point.cash - variance,
                range: [point.cash - variance, point.cash + variance]
            };
        });
    }, [data]);

    const forecastStartIndex = processedData.findIndex(p => p.type === 'Forecast');

    return (
        <Card
            title="Cash Flow Horizon"
            icon={<CurrencyDollarIcon className="h-6 w-6" />}
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
                            <LineChart data={processedData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={`rgba(${primaryRgb}, 0.1)`} vertical={false} />
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
                                    content={<CustomTooltip />}
                                    wrapperStyle={{ outline: 'none', zIndex: 1000 }}
                                    allowEscapeViewBox={{ x: true, y: true }}
                                />
                                <Legend wrapperStyle={{ fontSize: "12px", fontFamily: "JetBrains Mono", paddingTop: "10px" }} />

                                {/* Define Gradients */}
                                <defs>
                                    <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="1" y2="0">
                                        <stop offset="0%" stopColor={cashColor} stopOpacity={0} />
                                        <stop offset="100%" stopColor={cashColor} stopOpacity={0.2} />
                                    </linearGradient>
                                </defs>

                                {/* 1. The Cone of Uncertainty (Range Area) */}
                                <Area
                                    type="monotone"
                                    dataKey="range"
                                    name="Forecast Variance"
                                    stroke="none"
                                    fill="url(#uncertaintyGradient)"
                                    connectNulls
                                />

                                {/* 2. The Main Line */}
                                <Line
                                    type="monotone"
                                    dataKey="cash"
                                    name="Cash Position"
                                    stroke={cashColor}
                                    strokeWidth={2}
                                    dot={(props) => {
                                        const { cx, cy, payload } = props;
                                        const isForecast = payload.type === 'Forecast';
                                        return (
                                            <circle
                                                cx={cx}
                                                cy={cy}
                                                r={isForecast ? 3 : 4}
                                                fill={isForecast ? 'var(--bg-base)' : cashColor}
                                                stroke={cashColor}
                                                strokeWidth={isForecast ? 1 : 0}
                                            />
                                        );
                                    }}
                                    activeDot={{ r: 6, fill: cashColor, stroke: 'white', strokeWidth: 2 }}
                                />

                                {forecastStartIndex > 0 && (
                                    <ReferenceLine
                                        x={processedData[forecastStartIndex].month}
                                        stroke={`rgba(${primaryRgb}, 0.4)`}
                                        strokeDasharray="3 3"
                                    >
                                        <text
                                            y={10}
                                            fill={`rgba(${primaryRgb}, 0.7)`}
                                            fontSize={9}
                                            textAnchor="middle"
                                            className="uppercase font-mono tracking-widest"
                                        >
                                            Now
                                        </text>
                                    </ReferenceLine>
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </Card>
    );
};

export default CashFlowCard;
