"use client";

import { useMemo } from 'react';
import { Trade } from '@/lib/supabase';
import { Trophy, Ban } from 'lucide-react';

interface SymbolBreakdownProps {
    trades: Trade[];
}

export function SymbolBreakdown({ trades }: SymbolBreakdownProps) {
    const data = useMemo(() => {
        if (!trades.length) return [];

        const map = new Map<string, {
            symbol: string;
            pnl: number;
            wins: number;
            total: number;
            grossProfit: number;
            grossLoss: number;
        }>();

        trades.forEach(t => {
            const sym = t.symbol;
            if (!map.has(sym)) {
                map.set(sym, { symbol: sym, pnl: 0, wins: 0, total: 0, grossProfit: 0, grossLoss: 0 });
            }
            const entry = map.get(sym)!;
            entry.pnl += t.pnl_net;
            entry.total++;
            if (t.pnl_net > 0) {
                entry.wins++;
                entry.grossProfit += t.pnl_net;
            } else {
                entry.grossLoss += Math.abs(t.pnl_net);
            }
        });

        // Convert to array and sort by PnL Descending
        return Array.from(map.values())
            .map(item => ({
                ...item,
                winRate: (item.wins / item.total) * 100,
                pf: item.grossLoss === 0 ? item.grossProfit : item.grossProfit / item.grossLoss
            }))
            .sort((a, b) => b.pnl - a.pnl);

    }, [trades]);

    if (data.length === 0) {
        return <div className="h-full flex items-center justify-center text-slate-500 text-xs text-center"><p>无品种数据</p></div>;
    }

    return (
        <div className="h-full flex flex-col">
            <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <Trophy size={16} /> 品种表现 (By Symbol)
            </h3>
            <div className="flex-1 overflow-auto custom-scrollbar">
                <table className="w-full text-left border-collapse">
                    <thead className="sticky top-0 bg-[#0f172a] z-10">
                        <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-white/5">
                            <th className="py-2 pl-2">Symbol</th>
                            <th className="py-2 text-right">PnL</th>
                            <th className="py-2 text-right">Win%</th>
                            <th className="py-2 text-right pr-2">PF</th>
                        </tr>
                    </thead>
                    <tbody className="text-xs">
                        {data.map((row) => (
                            <tr key={row.symbol} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                <td className="py-2.5 pl-2 font-medium text-slate-300">
                                    {row.symbol}
                                    <div className="text-[10px] text-slate-500 font-normal">{row.total} trades</div>
                                </td>
                                <td className={`py-2.5 text-right font-bold ${row.pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                    ${row.pnl.toFixed(2)}
                                </td>
                                <td className="py-2.5 text-right text-slate-300">
                                    {row.winRate.toFixed(0)}%
                                </td>
                                <td className="py-2.5 text-right pr-2 text-slate-400 font-mono">
                                    {row.pf.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
