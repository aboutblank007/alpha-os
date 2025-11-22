"use client";

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ReferenceDot } from 'recharts';
import React from 'react';

type Period = '1W' | '1M' | '3M' | 'YTD' | 'ALL';

interface TagItem { date: string; label: string }

interface EquityCurveProps {
    data: Array<{ date: string; equity: number }>;
    period?: Period;
    overlays?: { ema?: boolean; bb?: boolean };
    tags?: TagItem[];
    onPeriodChange?: (p: Period) => void;
    onToggleOverlay?: (k: 'ema' | 'bb') => void;
}

function ema(values: number[], k = 10) {
    const alpha = 2 / (k + 1);
    let prev = values[0] ?? 0;
    return values.map(v => {
        const e = alpha * v + (1 - alpha) * prev;
        prev = e;
        return e;
    });
}

function sma(values: number[], k = 20) {
    const res: number[] = [];
    for (let i = 0; i < values.length; i++) {
        const start = Math.max(0, i - k + 1);
        const slice = values.slice(start, i + 1);
        res.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    return res;
}

function std(values: number[], mean: number[], k = 20) {
    const res: number[] = [];
    for (let i = 0; i < values.length; i++) {
        const m = mean[i];
        const start = Math.max(0, i - k + 1);
        const slice = values.slice(start, i + 1);
        const s = Math.sqrt(slice.reduce((acc, v) => acc + Math.pow(v - m, 2), 0) / slice.length);
        res.push(s);
    }
    return res;
}

export function EquityCurve({ data, period = '1M', overlays = { ema: true, bb: false }, tags = [], onPeriodChange, onToggleOverlay }: EquityCurveProps) {
    const periods: Period[] = ['1W', '1M', '3M', 'YTD', 'ALL'];
    const windows = React.useMemo<Record<Period, number>>(() => ({ '1W': 7, '1M': 30, '3M': 90, 'YTD': 365, 'ALL': data.length }), [data.length]);
    const filtered = React.useMemo(() => data.slice(-windows[period]), [data, period, windows]);
    const values = filtered.map(d => d.equity);
    const emaSeries = overlays.ema ? ema(values, 10) : [];
    const ma20 = overlays.bb ? sma(values, 20) : [];
    const sd20 = overlays.bb ? std(values, ma20, 20) : [];
    const bbUpper = overlays.bb ? ma20.map((m, i) => m + 2 * (sd20[i] ?? 0)) : [];
    const bbLower = overlays.bb ? ma20.map((m, i) => m - 2 * (sd20[i] ?? 0)) : [];
    const chartData = filtered.map((d, i) => ({ ...d, ema: emaSeries[i], bbU: bbUpper[i], bbL: bbLower[i] }));
    const minEquity = Math.min(...values) * 0.995;
    const maxEquity = Math.max(...values) * 1.005;

    return (
        <div className="glass-panel p-6 rounded-xl h-[500px] flex flex-col">
            <div className="flex justify-between items-start mb-6">
                <div>
                    <h3 className="text-lg font-semibold text-white">权益曲线</h3>
                    <p className="text-sm text-slate-500 mt-1">净资产价值 (NAV)</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex gap-1 bg-white/5 p-1 rounded-lg">
                        {periods.map((p) => (
                            <button
                                key={p}
                                onClick={() => onPeriodChange?.(p)}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${p === period ? 'bg-white/10 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}
                            >
                                {p}
                            </button>
                        ))}
                    </div>
                    <div className="flex items-center gap-2">
                        <button className={`px-2 py-1 rounded text-xs ${overlays.ema ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}`} onClick={() => onToggleOverlay?.('ema')}>EMA</button>
                        <button className={`px-2 py-1 rounded text-xs ${overlays.bb ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}`} onClick={() => onToggleOverlay?.('bb')}>BB</button>
                    </div>
                </div>
            </div>

            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis dataKey="date" stroke="#475569" fontSize={11} tickLine={false} axisLine={false} dy={10} />
                        <YAxis stroke="#475569" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v.toLocaleString()}`} domain={[minEquity, maxEquity]} />
                        <Tooltip contentStyle={{ backgroundColor: 'rgba(3, 7, 18, 0.9)', backdropFilter: 'blur(8px)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)', padding: '8px 12px' }} itemStyle={{ color: '#fff', fontWeight: 500, fontSize: '13px' }} labelStyle={{ color: '#94a3b8', marginBottom: '4px', fontSize: '11px' }} formatter={(value: number) => [`$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, 'Equity']} cursor={{ stroke: '#6366f1', strokeWidth: 1, strokeDasharray: '4 4' }} />
                        <Area type="monotone" dataKey="equity" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorEquity)" />
                        {overlays.ema && <Line type="monotone" dataKey="ema" stroke="#06b6d4" strokeWidth={1.5} dot={false} />}
                        {overlays.bb && <Line type="monotone" dataKey="bbU" stroke="#a855f7" strokeWidth={1} dot={false} />}
                        {overlays.bb && <Line type="monotone" dataKey="bbL" stroke="#a855f7" strokeWidth={1} dot={false} />}
                        {tags.map(tag => {
                            const y = chartData.find(d => d.date === tag.date)?.equity;
                            return y ? <ReferenceDot key={`${tag.date}-${tag.label}`} x={tag.date} y={y} r={4} fill="#ef4444" stroke="#fff" /> : null;
                        })}
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
