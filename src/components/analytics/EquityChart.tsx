"use client";

import { useMemo } from 'react';
import { Trade } from '@/lib/supabase';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface EquityChartProps {
    trades: Trade[];
}

export function EquityChart({ trades }: EquityChartProps) {
    const data = useMemo(() => {
        if (!trades.length) return [];

        // Sort by date ascending
        const sorted = [...trades].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

        let cumulative = 0;
        return sorted.map(t => {
            cumulative += t.pnl_net;
            return {
                date: new Date(t.created_at),
                value: cumulative,
                symbol: t.symbol,
                pnl: t.pnl_net
            };
        });
    }, [trades]);

    if (data.length === 0) {
        return <div className="h-full flex items-center justify-center text-slate-500 text-xs">暂无数据</div>;
    }

    // Determine colors based on overall result
    const isPositive = data.length > 0 && data[data.length - 1].value >= 0;
    const color = isPositive ? "#10b981" : "#ef4444"; // accent-success or accent-danger

    return (
        <div className="w-full h-full min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="colorPnL" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={color} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                    <XAxis
                        dataKey="date"
                        tickFormatter={(d) => d.toLocaleDateString([], { month: '2-digit', day: '2-digit' })}
                        stroke="#64748b"
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        minTickGap={30}
                    />
                    <YAxis
                        stroke="#64748b"
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        tickFormatter={(v) => `$${v}`}
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        itemStyle={{ color: '#fff' }}
                        labelFormatter={(d) => d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        formatter={(value: number) => [`$${value.toFixed(2)}`, '累计盈亏']}
                    />
                    <Area
                        type="monotone"
                        dataKey="value"
                        stroke={color}
                        fillOpacity={1}
                        fill="url(#colorPnL)"
                        strokeWidth={2}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
