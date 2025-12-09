
import React from 'react';
import Card from './Card';
import { BanknotesIcon } from './icons/BanknotesIcon';
import { FinanceOpportunity } from '../types';

interface SubsidizedFinanceCardProps {
    data: FinanceOpportunity[];
    onAction?: () => void;
    onToggleSize?: () => void;
    onDelete?: () => void;
    isFullWidth?: boolean;
}

const SubsidizedFinanceCard: React.FC<SubsidizedFinanceCardProps> = ({ data, onAction, onToggleSize, onDelete, isFullWidth }) => {
    return (
        <Card
            title="Subsidized Finance Ops"
            icon={<BanknotesIcon className="h-6 w-6" />}
            onAnalysisClick={onAction}
            onToggleSize={onToggleSize}
            onDelete={onDelete}
            isFullWidth={isFullWidth}
        >
            <div className="overflow-x-auto h-full">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="border-b" style={{ borderColor: 'var(--col-muted)' }}>
                            <th className="p-3 text-[10px] uppercase tracking-widest font-normal font-mono" style={{ color: 'var(--text-sub)' }}>Opportunity</th>
                            <th className="p-3 text-[10px] uppercase tracking-widest font-normal font-mono" style={{ color: 'var(--text-sub)' }}>Type</th>
                            <th className="p-3 text-[10px] uppercase tracking-widest font-normal font-mono hidden sm:table-cell" style={{ color: 'var(--text-sub)' }}>Amount</th>
                            <th className="p-3 text-[10px] uppercase tracking-widest font-normal font-mono" style={{ color: 'var(--text-sub)' }}>Deadline</th>
                        </tr>
                    </thead>
                    <tbody className="font-mono text-sm">
                        {data.map((item, index) => (
                            <tr key={item.id} className="border-b transition-colors group" style={{ borderColor: 'rgba(var(--col-muted-rgb), 0.5)' }}>
                                <td className="p-3 font-medium group-hover:text-white" style={{ color: 'var(--text-main)' }}>{item.name}</td>
                                <td className="p-3">
                                    <span className="px-2 py-1 text-[10px] font-bold rounded-sm border uppercase tracking-wide"
                                        style={{
                                            backgroundColor: 'rgba(var(--bg-surface-rgb), 0.5)',
                                            color: 'var(--col-primary)',
                                            borderColor: 'var(--col-muted)'
                                        }}>
                                        {item.type}
                                    </span>
                                </td>
                                <td className="p-3 hidden sm:table-cell" style={{ color: 'var(--text-sub)' }}>{item.amount}</td>
                                <td className="p-3">
                                    <div className="flex items-center gap-2">
                                        <div className="h-1.5 w-1.5 rounded-full shadow-[0_0_5px]" style={{ backgroundColor: 'var(--col-secondary)' }}></div>
                                        <span style={{ color: 'var(--text-sub)' }}>{item.deadline}</span>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </Card>
    );
};

export default SubsidizedFinanceCard;