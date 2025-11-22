import { Trade } from '@/lib/supabase';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Layers } from 'lucide-react';

interface SymbolPerformanceProps {
    trades: Trade[];
}

export function SymbolPerformance({ trades }: SymbolPerformanceProps) {
    // Aggregate data by symbol
    const symbolStats = trades.reduce((acc, trade) => {
        if (!acc[trade.symbol]) {
            acc[trade.symbol] = {
                symbol: trade.symbol,
                pnl: 0,
                trades: 0,
                wins: 0,
                volume: 0
            };
        }
        acc[trade.symbol].pnl += trade.pnl_net;
        acc[trade.symbol].trades += 1;
        acc[trade.symbol].volume += trade.quantity;
        if (trade.pnl_net > 0) acc[trade.symbol].wins += 1;
        return acc;
    }, {} as Record<string, { symbol: string; pnl: number; trades: number; wins: number; volume: number }>);

    const data = Object.values(symbolStats)
        .sort((a, b) => b.pnl - a.pnl) // Sort by PnL descending
        .slice(0, 5); // Top 5

    const COLORS = ['#6366f1', '#a855f7', '#06b6d4', '#10b981', '#f43f5e'];

    return (
        <div className="glass-panel p-6 rounded-xl h-full flex flex-col">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">交易表现</h3>
                <div className="p-2 rounded-lg bg-white/5 text-slate-400">
                    <Layers size={16} />
                </div>
            </div>

            <div className="flex flex-col gap-6 flex-1 min-h-0">
                {/* Chart Section */}
                <div className="relative h-[220px] flex-shrink-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="trades"
                                stroke="none"
                            >
                                {data.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={COLORS[index % COLORS.length]}
                                    />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(3, 7, 18, 0.9)',
                                    backdropFilter: 'blur(8px)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    borderRadius: '8px',
                                    color: '#f8fafc',
                                    padding: '8px 12px'
                                }}
                                itemStyle={{ color: '#fff', fontSize: '13px' }}
                                formatter={(value: number) => [`${value} 笔交易`, '交易量']}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                    {/* Center Text Overlay */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-3xl font-bold text-white tracking-tight">{trades.length}</span>
                        <span className="text-[10px] text-slate-500 uppercase tracking-widest font-medium mt-1">交易</span>
                    </div>
                </div>

                {/* List Section */}
                <div className="overflow-y-auto custom-scrollbar pr-2 flex-1">
                    <div className="space-y-2">
                        {data.map((item, index) => (
                            <div key={item.symbol} className="flex items-center justify-between p-2.5 rounded-lg hover:bg-white/5 transition-colors">
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                                    <div>
                                        <p className="text-sm font-medium text-slate-200">{item.symbol}</p>
                                        <p className="text-xs text-slate-500">{item.trades} 笔</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`text-sm font-mono font-medium ${item.pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                        {item.pnl >= 0 ? '+' : ''}{item.pnl.toFixed(2)}
                                    </p>
                                    <div className="flex items-center justify-end gap-1 text-[10px] text-slate-500">
                                        <span>{((item.wins / item.trades) * 100).toFixed(0)}% 胜率</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
