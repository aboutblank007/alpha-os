"use client";

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface DrawdownData {
    date: string;
    drawdown: number; // e.g. -5.2 for -5.2%
}

interface DrawdownChartProps {
    data: DrawdownData[];
    height?: number;
}

const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-900 border border-white/10 p-3 rounded-lg shadow-xl">
                <p className="text-slate-400 text-xs mb-1">{label}</p>
                <p className="text-accent-danger text-lg font-bold font-mono">
                    {payload[0].value.toFixed(2)}%
                </p>
            </div>
        );
    }
    return null;
};

export function DrawdownChart({ data, height = 300 }: DrawdownChartProps) {

    return (
        <div style={{ width: '100%', height }}>
            <ResponsiveContainer>
                <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis
                        dataKey="date"
                        stroke="#94a3b8"
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        minTickGap={30}
                    />
                    <YAxis
                        stroke="#94a3b8"
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        tickFormatter={(val) => `${val}%`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                        type="monotone"
                        dataKey="drawdown"
                        stroke="#ef4444"
                        fillOpacity={1}
                        fill="url(#colorDrawdown)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}

