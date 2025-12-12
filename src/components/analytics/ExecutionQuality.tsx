"use client";

import { useMemo } from 'react';
import { Trade } from '@/lib/supabase';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Target } from 'lucide-react';

interface ExecutionQualityProps {
    trades: Trade[];
}

export function ExecutionQuality({ trades }: ExecutionQualityProps) {
    const maeData = useMemo(() => {
        return trades
            .filter(t => t.mae !== undefined && t.mae !== null)
            .map(t => ({
                x: t.mae, // MAE is usually negative or 0 distance from entry? Or price? 
                // Usually MAE is "Maximum Adverse Excursion" in pips or currency. 
                // Let's assume it's positive number representing distance against trade.
                y: t.pnl_net,
                symbol: t.symbol
            }));
    }, [trades]);

    const mfeData = useMemo(() => {
        return trades
            .filter(t => t.mfe !== undefined && t.mfe !== null)
            .map(t => ({
                x: t.mfe, // Positive excursion
                y: t.pnl_net,
                symbol: t.symbol
            }));
    }, [trades]);

    if (maeData.length === 0 && mfeData.length === 0) {
        return <div className="h-full flex items-center justify-center text-slate-500 text-xs">暂无 MAE/MFE 数据</div>;
    }

    return (
        <div className="h-full flex flex-col">
            <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                <Target size={16} /> 执行质量 (Execution Quality)
            </h3>

            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 min-h-[250px]">
                {/* MAE Chart */}
                <div className="glass-panel bg-black/20 rounded-lg p-2 border border-white/5 flex flex-col">
                    <h4 className="text-[10px] text-slate-400 uppercase tracking-wider mb-2 text-center">MAE vs PnL (Risk Taken)</h4>
                    <div className="flex-1 min-h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                                <XAxis type="number" dataKey="x" name="MAE" unit="$" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'MAE ($)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                                <YAxis type="number" dataKey="y" name="PnL" unit="$" stroke="#64748b" tick={{ fontSize: 10 }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} itemStyle={{ color: '#fff' }} />
                                <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
                                <Scatter name="Trades" data={maeData} fill="#f43f5e" fillOpacity={0.6} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* MFE Chart */}
                <div className="glass-panel bg-black/20 rounded-lg p-2 border border-white/5 flex flex-col">
                    <h4 className="text-[10px] text-slate-400 uppercase tracking-wider mb-2 text-center">MFE vs PnL (Potential Left)</h4>
                    <div className="flex-1 min-h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                                <XAxis type="number" dataKey="x" name="MFE" unit="$" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'MFE ($)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                                <YAxis type="number" dataKey="y" name="PnL" unit="$" stroke="#64748b" tick={{ fontSize: 10 }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} itemStyle={{ color: '#fff' }} />
                                <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
                                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1000, y: 1000 }]} stroke="#10b981" strokeDasharray="3 3" opacity={0.3} />
                                <Scatter name="Trades" data={mfeData} fill="#10b981" fillOpacity={0.6} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
}
