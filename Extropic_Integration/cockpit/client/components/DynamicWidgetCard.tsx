
import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ComposedChart, PieChart, Pie, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, RadialBarChart, RadialBar } from 'recharts';
import Card from './Card';
import { SparklesIcon } from './icons/SparklesIcon'; 
import { WidgetConfig } from '../types';
import { Theme, THEMES, getSemanticColor, getHologramColor, isSemanticKey, HOLOGRAM_CONFIG } from '../themes';

interface DynamicWidgetCardProps {
    config: WidgetConfig;
    data: any[]; // Generic data input
    onDelete: (id: string) => void;
    currentTheme: Theme;
    onToggleSize?: () => void;
    isFullWidth?: boolean;
    onAction?: () => void; 
}

const formatCurrency = (value: number) => `€${(value / 1000).toFixed(0)}k`;

const CustomTooltip = ({ active, payload, label, colorTheme }: any) => {
    if (active && payload && payload.length) {
        // Dynamic border color based on Hologram Theme
        const borderColor = HOLOGRAM_CONFIG[colorTheme]?.hex || '#00f3ff';
        
        return (
            <div className="backdrop-blur-xl border p-0 rounded-sm shadow-neon text-xs font-mono min-w-[200px] overflow-hidden" style={{ backgroundColor: 'var(--bg-surface)', borderColor: borderColor }}>
                <div className="p-3 border-b border-white/10 flex justify-between items-center" style={{ backgroundColor: 'rgba(var(--bg-base-rgb), 0.5)' }}>
                    <span className="font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>{label || payload[0].name}</span>
                    <span className="text-[10px]" style={{ color: 'var(--text-sub)' }}>METRICS</span>
                </div>
                <div className="p-4 space-y-3 relative">
                    {payload.map((entry: any, index: number) => (
                        <div key={index} className="flex justify-between items-center gap-4 relative z-10">
                            <div className="flex items-center gap-2">
                                {/* Dynamic color dot from Recharts entry color */}
                                <div 
                                    className="w-2 h-2 rounded-sm shadow-[0_0_5px]" 
                                    style={{ backgroundColor: entry.color || entry.fill, boxShadow: `0 0 5px ${entry.color || entry.fill}` }}
                                ></div>
                                <span className="capitalize font-sans text-[11px] tracking-wide" style={{ color: 'var(--text-sub)' }}>{entry.name}</span> 
                            </div>
                            <span className="font-bold font-mono text-sm drop-shadow-md" style={{ color: 'var(--text-main)' }}>
                                {typeof entry.value === 'number' 
                                    ? (entry.name === 'score' ? entry.value : `€${entry.value.toLocaleString()}`) 
                                    : entry.value}
                            </span>
                        </div>
                    ))}
                     {/* Subtle Grid Background */}
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(128,128,128,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(128,128,128,0.05)_1px,transparent_1px)] bg-[length:12px_12px] pointer-events-none z-0"></div>
                </div>
            </div>
        );
    }
    return null;
};

const DynamicWidgetCard: React.FC<DynamicWidgetCardProps> = ({ config, data, onDelete, currentTheme, onToggleSize, isFullWidth, onAction }) => {
    const themeVars = THEMES[currentTheme].vars;
    const localPrimary = config.colorTheme ? HOLOGRAM_CONFIG[config.colorTheme].rgb : themeVars['--col-primary-rgb'];
    const gridColor = `rgba(${localPrimary}, 0.1)`;
    const textColor = themeVars['--text-sub'];

    const [hiddenKeys, setHiddenKeys] = useState<string[]>([]);
    const toggleKeyVisibility = (key: string) => {
        setHiddenKeys(prev => 
            prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
        );
    };

    const resolveColor = (key: string) => {
        if (config.dataKeys.length === 1 && config.colorTheme) return getHologramColor(config.colorTheme);
        if (isSemanticKey(key)) return getSemanticColor(key, currentTheme);
        return getHologramColor(config.colorTheme);
    };

    const resolveOpacity = (key: string, index: number): number => {
        if (config.dataKeys.length === 1) return 1;
        if (isSemanticKey(key)) return 0.9; 
        return Math.max(0.3, 0.8 - (index * 0.2)); 
    };

    const renderInteractiveLegend = (props: any) => {
        const { payload } = props;
        return (
            <div className="flex flex-wrap justify-center gap-2 mt-2 select-none">
                {payload.map((entry: any, index: number) => {
                    const isHidden = hiddenKeys.includes(entry.value);
                    const color = entry.color;
                    
                    return (
                        <button
                            key={`legend-${index}`}
                            onClick={() => toggleKeyVisibility(entry.value)}
                            className={`
                                flex items-center gap-2 px-2 py-1 rounded-sm border transition-all duration-300
                                ${isHidden 
                                    ? 'border-transparent bg-slate-800/50 text-slate-600 opacity-50' 
                                    : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/30 shadow-[0_0_10px_rgba(0,0,0,0.2)]'
                                }
                            `}
                        >
                            <div 
                                className={`w-2 h-2 rounded-full transition-all duration-300 ${isHidden ? 'scale-50 grayscale' : 'scale-100 shadow-[0_0_5px]'}`}
                                style={{ backgroundColor: color, boxShadow: isHidden ? 'none' : `0 0 5px ${color}` }}
                            />
                            <span className={`text-[10px] font-mono uppercase tracking-wider ${isHidden ? 'line-through' : ''}`} style={{ color: isHidden ? 'var(--text-sub)' : 'var(--text-main)' }}>
                                {entry.value}
                            </span>
                        </button>
                    );
                })}
            </div>
        );
    };

    const renderChart = () => {
        if (!data || data.length === 0) {
            return <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono uppercase tracking-widest">No Signal Data</div>;
        }

        const commonProps = {
            data: data,
            margin: { top: 5, right: 20, left: 10, bottom: 5 }
        };

        const AxisGroup = (
            <>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis dataKey="month" stroke={textColor} fontSize={10} tickLine={false} axisLine={false} fontFamily="JetBrains Mono" />
                <YAxis stroke={textColor} fontSize={10} tickFormatter={formatCurrency} tickLine={false} axisLine={false} fontFamily="JetBrains Mono" />
                <Tooltip 
                    content={<CustomTooltip colorTheme={config.colorTheme} />} 
                    cursor={{ fill: `rgba(${localPrimary}, 0.05)` }}
                    wrapperStyle={{ outline: 'none', zIndex: 1000 }}
                    allowEscapeViewBox={{ x: true, y: true }}
                />
                <Legend content={renderInteractiveLegend} />
            </>
        );

        const ChartContent = (() => {
            switch (config.type) {
                case 'pie': {
                    let pieData = [];
                    if (config.dataKeys.length > 1) {
                        pieData = config.dataKeys
                            .filter(key => !hiddenKeys.includes(key))
                            .map(key => ({
                                name: key,
                                value: data.reduce((acc, curr) => acc + (curr[key] || 0), 0)
                            }));
                    } else {
                        const key = config.dataKeys[0] || 'value';
                        pieData = data.slice(-12).map(d => ({
                            name: d.month,
                            value: d[key]
                        }));
                    }
                    return (
                        <PieChart>
                             <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                                stroke="none"
                            >
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={resolveColor(entry.name)} strokeWidth={1} fillOpacity={resolveOpacity(entry.name, index)} />
                                ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip colorTheme={config.colorTheme} />} wrapperStyle={{ outline: 'none', zIndex: 1000 }}/>
                            {config.dataKeys.length > 1 && <Legend content={renderInteractiveLegend} />}
                        </PieChart>
                    );
                }
                case 'radar':
                    return (
                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data.slice(-6)}>
                            <PolarGrid stroke={gridColor} />
                            <PolarAngleAxis dataKey="month" tick={{ fill: textColor, fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                            <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={false} axisLine={false} />
                            {config.dataKeys.map((key, i) => (
                                <Radar key={key} name={key} dataKey={key} stroke={resolveColor(key)} strokeWidth={2} fill={resolveColor(key)} fillOpacity={resolveOpacity(key, i) * 0.5} hide={hiddenKeys.includes(key)} />
                            ))}
                            <Tooltip content={<CustomTooltip colorTheme={config.colorTheme} />} wrapperStyle={{ outline: 'none', zIndex: 1000 }}/>
                            <Legend content={renderInteractiveLegend} />
                        </RadarChart>
                    );
                case 'radial': {
                    const latest = data[data.length - 1] || {};
                    const radialData = config.dataKeys
                        .filter(k => !hiddenKeys.includes(k))
                        .map((key) => ({
                            name: key, value: latest[key] || latest['score'] || 0, fill: resolveColor(key)
                        }));
                    return (
                        <RadialBarChart cx="50%" cy="50%" innerRadius="30%" outerRadius="100%" barSize={20} data={radialData} startAngle={180} endAngle={0}>
                            <RadialBar background={{ fill: 'rgba(128,128,128,0.1)' }} label={{ position: 'insideStart', fill: '#fff', fontSize: 10 }} dataKey="value" cornerRadius={10} />
                            <Legend content={renderInteractiveLegend} layout="vertical" verticalAlign="middle" wrapperStyle={{fontSize: "10px", fontFamily: "JetBrains Mono", right: 0, color: textColor}}/>
                            <Tooltip content={<CustomTooltip colorTheme={config.colorTheme} />} wrapperStyle={{ outline: 'none', zIndex: 1000 }}/>
                        </RadialBarChart>
                    );
                }
                case 'area':
                    return (
                        <AreaChart {...commonProps}>
                            {AxisGroup}
                            {config.dataKeys.map((key, i) => <Area key={key} type="monotone" dataKey={key} stackId="1" stroke={resolveColor(key)} fill={resolveColor(key)} fillOpacity={resolveOpacity(key, i) * 0.6} hide={hiddenKeys.includes(key)} />)}
                        </AreaChart>
                    );
                case 'line':
                    return (
                        <LineChart {...commonProps}>
                            {AxisGroup}
                            {config.dataKeys.map((key, i) => <Line key={key} type="monotone" dataKey={key} stroke={resolveColor(key)} strokeWidth={3} dot={{r: 3, stroke: resolveColor(key), fill: '#000'}} strokeOpacity={resolveOpacity(key, i)} hide={hiddenKeys.includes(key)} />)}
                        </LineChart>
                    );
                case 'composed':
                     return (
                        <ComposedChart {...commonProps}>
                            {AxisGroup}
                            {config.dataKeys.map((key, i) => i === 0 ? <Bar key={key} dataKey={key} fill={resolveColor(key)} barSize={20} radius={[4,4,0,0]} fillOpacity={resolveOpacity(key, i)} hide={hiddenKeys.includes(key)} /> : <Line key={key} type="monotone" dataKey={key} stroke={resolveColor(key)} strokeWidth={3} dot={{r:3}} strokeOpacity={resolveOpacity(key, i)} hide={hiddenKeys.includes(key)} />)}
                        </ComposedChart>
                    );
                case 'bar':
                default:
                    return (
                        <BarChart {...commonProps}>
                            {AxisGroup}
                            {config.dataKeys.map((key, i) => <Bar key={key} dataKey={key} fill={resolveColor(key)} radius={[4, 4, 0, 0]} fillOpacity={resolveOpacity(key, i)} hide={hiddenKeys.includes(key)} />)}
                        </BarChart>
                    );
            }
        })();

        return (
            <ResponsiveContainer width="99%" height="100%" debounce={50}>
                {ChartContent}
            </ResponsiveContainer>
        );
    };

    return (
        <Card 
            title={config.title} 
            icon={<SparklesIcon className="h-6 w-6" />}
            className="relative"
            onToggleSize={onToggleSize}
            isFullWidth={isFullWidth}
            onAnalysisClick={onAction}
            onDelete={() => onDelete(config.id)}
            colorTheme={config.colorTheme} 
        >
            {/* STRICT LAYOUT CONTAINER FOR RECHARTS */}
            <div className="h-full w-full min-h-[200px] flex flex-col">
                <div className="flex-grow w-full h-0 relative">
                    <div className="absolute inset-0">
                        {renderChart()}
                    </div>
                </div>
            </div>
        </Card>
    );
};

export default DynamicWidgetCard;
