"use client";

import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, Cell, ReferenceLine, Label } from 'recharts';
import { Card } from '@/components/Card';

interface MaeMfeData {
    ticket: string;
    mae: number;
    mfe: number;
    pnl: number;
}

interface MaeMfeScatterChartProps {
    data: MaeMfeData[];
    height?: number;
}

export function MaeMfeScatterChart({ data, height = 400 }: MaeMfeScatterChartProps) {
    // 处理数据，确保均为正数以便于展示，或者保持原样
    // MAE通常是负数（不利），MFE是正数（有利）。
    // 为了图表直观，我们可以取绝对值展示幅度，或者保留符号。
    // 这里我们假设传入的 MAE 为负值（如 -50.5），MFE 为正值（如 100.2）。
    // 散点图 X 轴为 MAE (0 to -Infinity), Y 轴为 MFE (0 to +Infinity)
    
    // 格式化 tooltip
    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-slate-900 border border-white/10 p-3 rounded-lg shadow-xl">
                    <p className="text-slate-300 text-sm mb-1">Ticket: <span className="text-white font-mono">{d.ticket}</span></p>
                    <p className="text-accent-danger text-sm">MAE: {d.mae.toFixed(2)}</p>
                    <p className="text-accent-success text-sm">MFE: {d.mfe.toFixed(2)}</p>
                    <p className={`text-sm font-bold ${d.pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                        PnL: ${d.pnl.toFixed(2)}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div style={{ width: '100%', height }}>
            <ResponsiveContainer>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <XAxis 
                        type="number" 
                        dataKey="mae" 
                        name="MAE" 
                        unit="" 
                        stroke="#94a3b8" 
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        label={{ value: '最大不利偏移 (MAE)', position: 'bottom', offset: 0, fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis 
                        type="number" 
                        dataKey="mfe" 
                        name="MFE" 
                        unit="" 
                        stroke="#94a3b8" 
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        label={{ value: '最大有利偏移 (MFE)', angle: -90, position: 'left', fill: '#94a3b8', fontSize: 12 }}
                    />
                    <ZAxis type="number" range={[50, 50]} /> {/* 固定点的大小 */}
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                    <ReferenceLine x={0} stroke="#475569" />
                    <ReferenceLine y={0} stroke="#475569" />
                    <Scatter name="Trades" data={data} fill="#8884d8">
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? '#10b981' : '#ef4444'} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
}

