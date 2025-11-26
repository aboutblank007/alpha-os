"use client";

import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, Cell, ReferenceLine, TooltipProps } from 'recharts';
import { Trade } from '@/lib/supabase';
import { useMemo } from 'react';

interface MaeMfeScatterChartProps {
    trades: Trade[];
    height?: number;
}

export function MaeMfeScatterChart({ trades, height = 400 }: MaeMfeScatterChartProps) {
    const data = useMemo(() => {
        return trades
            .filter(t => t.status === 'closed')
            .map(t => {
                // Mock logic for Preview if real data is missing
                // In production, this should come from DB
                const simulatedMae = t.mae ?? (t.pnl_net < 0 ? t.pnl_net * 1.2 : -Math.abs(t.pnl_net * 0.5));
                const simulatedMfe = t.mfe ?? (t.pnl_net > 0 ? t.pnl_net * 1.5 : Math.abs(t.pnl_net * 0.2));

                return {
                    ticket: t.id.slice(0, 8),
                    mae: Math.abs(simulatedMae), // Show as positive magnitude usually, or keep negative? Standard is negative x-axis or absolute.
                    // Let's use absolute for X-axis but invert logic: X is MAE (Adverse), Y is MFE (Favorable)
                    // Usually MAE is plotted on X (0 to large), MFE on Y (0 to large)
                    mfe: Math.abs(simulatedMfe),
                    pnl: t.pnl_net,
                    win: t.pnl_net >= 0
                };
            });
    }, [trades]);

    // Custom Tooltip
    const CustomTooltip = ({ active, payload }: TooltipProps<number, string>) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-slate-900 border border-white/10 p-3 rounded-lg shadow-xl backdrop-blur-md">
                    <p className="text-slate-400 text-xs mb-1">ID: <span className="text-white font-mono">{d.ticket}</span></p>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                        <span className="text-slate-400">MAE:</span>
                        <span className="text-accent-danger font-mono">{d.mae.toFixed(2)}</span>
                        
                        <span className="text-slate-400">MFE:</span>
                        <span className="text-accent-success font-mono">{d.mfe.toFixed(2)}</span>
                        
                        <span className="text-slate-400">PnL:</span>
                        <span className={`font-mono font-bold ${d.win ? 'text-accent-success' : 'text-accent-danger'}`}>
                            {d.pnl >= 0 ? '+' : ''}{d.pnl.toFixed(2)}
                        </span>
                    </div>
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
                        stroke="#64748b"
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        axisLine={{ stroke: '#334155' }}
                        label={{ value: '最大不利偏移 (MAE) - 潜在止损优化', position: 'bottom', offset: 0, fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis
                        type="number"
                        dataKey="mfe"
                        name="MFE"
                        stroke="#64748b"
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        axisLine={{ stroke: '#334155' }}
                        label={{ value: '最大有利偏移 (MFE) - 潜在止盈优化', angle: -90, position: 'left', fill: '#94a3b8', fontSize: 12 }}
                    />
                    <ZAxis type="number" range={[60, 60]} />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#ffffff30' }} />
                    
                    {/* Diagonal reference line (1:1 Risk/Reward proxy roughly) */}
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1000, y: 1000 }]} stroke="#ffffff10" strokeDasharray="5 5" />
                    
                    <Scatter name="Trades" data={data}>
                        {data.map((entry, index) => (
                            <Cell 
                                key={`cell-${index}`} 
                                fill={entry.win ? '#10b981' : '#ef4444'} 
                                fillOpacity={0.6}
                                stroke={entry.win ? '#10b981' : '#ef4444'}
                            />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
}
