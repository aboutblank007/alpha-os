"use client";

import { Trade } from '@/lib/supabase';
import { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface StrategyBreakdownProps {
    trades: Trade[];
}

export function StrategyBreakdown({ trades }: StrategyBreakdownProps) {
    const data = useMemo(() => {
        const map = new Map<string, { count: number; pnl: number; wins: number }>();
        
        trades.forEach(t => {
            // Mock strategies if missing
            const strategies = t.strategies && t.strategies.length > 0 
                ? t.strategies 
                : [t.pnl_net > 50 ? '趋势跟踪' : t.pnl_net < -50 ? '反转交易' : '超短线']; // Dummy categorization

            strategies.forEach(strat => {
                if (!map.has(strat)) {
                    map.set(strat, { count: 0, pnl: 0, wins: 0 });
                }
                const entry = map.get(strat)!;
                entry.count += 1;
                entry.pnl += t.pnl_net;
                if (t.pnl_net > 0) entry.wins += 1;
            });
        });

        return Array.from(map.entries()).map(([name, stats]) => ({
            name,
            value: stats.count,
            pnl: stats.pnl,
            winRate: (stats.wins / stats.count) * 100
        }));
    }, [trades]);

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

    return (
        <div className="glass-panel p-6 rounded-xl h-full flex flex-col">
            <h3 className="text-lg font-semibold text-white mb-6">策略表现分析</h3>
            
            <div className="flex-1 min-h-[200px] flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            fill="#8884d8"
                            paddingAngle={5}
                            dataKey="value"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip 
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const d = payload[0].payload;
                                    return (
                                        <div className="bg-black/90 border border-white/10 p-3 rounded text-xs">
                                            <div className="font-bold text-white mb-1">{d.name}</div>
                                            <div className="text-slate-400">交易次数: {d.value}</div>
                                            <div className="text-slate-400">胜率: <span className="text-white">{d.winRate.toFixed(1)}%</span></div>
                                            <div className="text-slate-400">净盈亏: <span className={d.pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}>${d.pnl.toFixed(0)}</span></div>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Legend verticalAlign="bottom" height={36} />
                    </PieChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-6 space-y-3">
                {data.sort((a, b) => b.pnl - a.pnl).map((item, idx) => (
                    <div key={item.name} className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></div>
                            <span className="text-slate-300">{item.name}</span>
                        </div>
                        <div className="flex items-center gap-4">
                            <span className="text-slate-500 text-xs">{item.winRate.toFixed(0)}% 胜率</span>
                            <span className={`font-mono font-medium ${item.pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                ${item.pnl.toFixed(0)}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

