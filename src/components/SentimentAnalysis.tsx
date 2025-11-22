import { Trade } from '@/lib/supabase';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Gauge } from 'lucide-react';

interface SentimentAnalysisProps {
    trades: Trade[];
}

export function SentimentAnalysis({ trades }: SentimentAnalysisProps) {
    // Use open trades for current sentiment, or last 10 closed if no open trades
    const activeTrades = trades.filter(t => t.status === 'open');
    const sentimentSource = activeTrades.length > 0 ? activeTrades : trades.slice(0, 10);
    
    const longs = sentimentSource.filter(t => t.side === 'buy').length;
    const shorts = sentimentSource.filter(t => t.side === 'sell').length;
    const total = longs + shorts;
    
    // Calculate bullish percentage (0 to 100)
    const bullishPct = total > 0 ? (longs / total) * 100 : 50;
    
    // Gauge Data: [Bearish, Bullish]
    const data = [
        { name: '看空', value: shorts, color: '#ef4444' },
        { name: '看多', value: longs, color: '#06b6d4' },
    ];
    
    // If no data, show neutral
    if (total === 0) {
        data[0].value = 1;
        data[1].value = 1;
        data[0].color = '#334155';
        data[1].color = '#334155';
    }

    const sentimentLabel = bullishPct > 60 ? '看多' : bullishPct < 40 ? '看空' : '中性';
    const sentimentColor = bullishPct > 60 ? 'text-accent-cyan' : bullishPct < 40 ? 'text-accent-danger' : 'text-slate-400';

    return (
        <div className="glass-panel rounded-xl p-6 h-full flex flex-col">
             <div className="flex items-center gap-2 mb-4">
                <Gauge className="text-slate-400" size={20} />
                <h3 className="text-lg font-semibold text-white">市场情绪</h3>
            </div>

            <div className="flex-1 relative min-h-[150px]">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="100%"
                            startAngle={180}
                            endAngle={0}
                            innerRadius={60}
                            outerRadius={80}
                            paddingAngle={0}
                            dataKey="value"
                            stroke="none"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Pie>
                    </PieChart>
                </ResponsiveContainer>
                
                <div className="absolute bottom-0 left-0 right-0 flex flex-col items-center justify-end pb-2">
                    <span className={`text-2xl font-bold ${sentimentColor} tracking-tight`}>
                        {sentimentLabel}
                    </span>
                    <span className="text-xs text-slate-500 mt-1">
                        {longs} 多头 / {shorts} 空头
                    </span>
                </div>
            </div>
            
            <div className="mt-4 flex justify-between text-xs text-slate-500 font-medium px-2">
                <span>看空</span>
                <span>看多</span>
            </div>
        </div>
    );
}
