
import { MonthlyData, CashFlowDataPoint, RatingData, FinanceOpportunity } from '../types';

export const generateManagementData = (): MonthlyData[] => {
    const data: MonthlyData[] = [];
    
    // Stabilize base payroll to simulate realistic HR costs
    let basePayroll = 35000; 

    for (let i = 11; i >= 0; i--) {
        const date = new Date();
        date.setMonth(date.getMonth() - i);
        
        // Revenue has some seasonality and volatility
        const revenue = Math.floor(Math.random() * 50000) + 120000;
        
        // Costs are correlated but have their own variance
        const costs = Math.floor(revenue * (Math.random() * 0.15 + 0.6));
        
        // Payroll drifts slightly but is mostly fixed cost
        basePayroll = basePayroll * (1 + (Math.random() - 0.4) * 0.05);
        
        data.push({
            month: date.toLocaleString('default', { month: 'short' }),
            revenue,
            costs,
            profit: revenue - costs,
            payroll: Math.floor(basePayroll) 
        });
    }
    return data;
};

export const generateCashFlowData = (): CashFlowDataPoint[] => {
    const data: CashFlowDataPoint[] = [];
    let lastCash = 50000;
    // Past 6 months
    for (let i = 6; i > 0; i--) {
        const date = new Date();
        date.setMonth(date.getMonth() - i);
        lastCash += (Math.random() - 0.45) * 20000;
        data.push({
            month: date.toLocaleString('default', { month: 'short' }) + ' ' + date.getFullYear().toString().slice(-2),
            cash: Math.max(0, lastCash),
            type: 'Past',
        });
    }

    // Current and next 11 months
    for (let i = 0; i < 12; i++) {
        const date = new Date();
        date.setMonth(date.getMonth() + i);
        lastCash += (Math.random() - 0.4) * 25000;
        data.push({
            month: date.toLocaleString('default', { month: 'short' }) + ' ' + date.getFullYear().toString().slice(-2),
            cash: Math.max(0, lastCash),
            type: 'Forecast',
        });
    }
    return data;
};

export const generateRatingData = (): RatingData => {
    const score = Math.floor(Math.random() * 40) + 60;
    let rating = 'C';
    if (score > 95) rating = 'AAA';
    else if (score > 90) rating = 'AA';
    else if (score > 85) rating = 'A';
    else if (score > 75) rating = 'BBB';
    else if (score > 65) rating = 'BB';
    else if (score > 55) rating = 'B';
    
    return {
        score,
        rating,
        metrics: [
            { 
                label: 'Debt-to-Equity', 
                value: (Math.random() * 1.5 + 0.2).toFixed(2),
                score: Math.floor(Math.random() * 40) + 60 
            },
            { 
                label: 'Current Ratio', 
                value: (Math.random() * 1.8 + 0.8).toFixed(2),
                score: Math.floor(Math.random() * 40) + 60 
            },
            { 
                label: 'Profit Margin', 
                value: `${(Math.random() * 15 + 5).toFixed(1)}%`,
                score: Math.floor(Math.random() * 40) + 60 
            },
        ]
    };
};

export const generateFinanceData = (): FinanceOpportunity[] => {
    return [
        { id: '1', name: 'Fondo Innovazione PMI', type: 'Grant', amount: '€250,000', deadline: '2024-12-31' },
        { id: '2', name: 'Credito d\'Imposta R&S', type: 'Tax Credit', amount: '40% of expenses', deadline: '2025-06-30' },
        { id: '3', name: 'Garanzia SACE Export', type: 'Guaranteed Loan', amount: 'Up to €5M', deadline: 'Rolling' },
        { id: '4', name: 'Bando Transizione Digitale', type: 'Voucher', amount: '€10,000', deadline: '2024-11-15' },
    ];
};
