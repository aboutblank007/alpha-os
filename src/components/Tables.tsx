"use client";

import { Trade } from "@/lib/supabase";
import { ArrowUpRight, ArrowDownRight, MoreHorizontal } from 'lucide-react';

interface PositionsTableProps {
    trades: Trade[];
}

export function PositionsTable({ trades }: PositionsTableProps) {
    const openTrades = trades.filter(t => t.status === 'open');

    if (openTrades.length === 0) {
        return (
            <div className="glass-panel p-8 rounded-2xl text-center">
                <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">💤</span>
                </div>
                <h3 className="text-lg font-medium text-white">No Open Positions</h3>
                <p className="text-slate-400 text-sm mt-1">Waiting for market opportunities...</p>
            </div>
        );
    }

    return (
        <div className="glass-panel rounded-2xl overflow-hidden">
            <div className="p-6 border-b border-white/5 flex justify-between items-center">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                    Active Positions
                </h3>
                <span className="text-xs font-mono text-slate-400 bg-slate-800/50 px-2 py-1 rounded">
                    LIVE
                </span>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left">
                    <thead>
                        <tr className="text-xs text-slate-400 uppercase tracking-wider border-b border-white/5 bg-slate-900/30">
                            <th className="px-6 py-4 font-medium">Symbol</th>
                            <th className="px-6 py-4 font-medium">Side</th>
                            <th className="px-6 py-4 font-medium text-right">Size</th>
                            <th className="px-6 py-4 font-medium text-right">Entry</th>
                            <th className="px-6 py-4 font-medium text-right">Market</th>
                            <th className="px-6 py-4 font-medium text-right">PnL</th>
                            <th className="px-6 py-4 font-medium"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {openTrades.map((trade) => {
                            const isProfit = trade.pnl_net >= 0;
                            return (
                                <tr key={trade.id} className="group hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4">
                                        <span className="font-bold text-white">{trade.symbol}</span>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-medium border ${
                                            trade.side === 'buy' 
                                                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                                                : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                                        }`}>
                                            {trade.side === 'buy' ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                                            {trade.side.toUpperCase()}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-slate-300">
                                        {trade.quantity}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-slate-300">
                                        {trade.entry_price.toFixed(2)}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-white">
                                        {/* Mock current price logic or real logic if available */}
                                        {(trade.entry_price + (trade.pnl_net / trade.quantity / (trade.side === 'buy' ? 1 : -1))).toFixed(2)}
                                    </td>
                                    <td className={`px-6 py-4 text-right font-mono font-bold ${isProfit ? 'text-emerald-400 text-glow-success' : 'text-rose-400 text-glow-danger'}`}>
                                        {isProfit ? '+' : ''}{trade.pnl_net.toFixed(2)}
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <button className="text-slate-500 hover:text-white transition-colors">
                                            <MoreHorizontal size={16} />
                                        </button>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

interface HistoryTableProps {
    trades: Trade[];
}

export function HistoryTable({ trades }: HistoryTableProps) {
    // Only show closed trades, sorted by date desc
    const closedTrades = trades
        .filter(t => t.status === 'closed')
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        .slice(0, 10); // Recent 10

    return (
        <div className="glass-panel rounded-2xl overflow-hidden mt-8">
            <div className="p-6 border-b border-white/5 flex justify-between items-center">
                <h3 className="text-lg font-semibold text-white">Recent History</h3>
                <button className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors">View All</button>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left">
                    <thead>
                        <tr className="text-xs text-slate-400 uppercase tracking-wider border-b border-white/5 bg-slate-900/30">
                            <th className="px-6 py-4 font-medium">Date</th>
                            <th className="px-6 py-4 font-medium">Symbol</th>
                            <th className="px-6 py-4 font-medium">Side</th>
                            <th className="px-6 py-4 font-medium text-right">PnL</th>
                            <th className="px-6 py-4 font-medium text-center">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {closedTrades.map((trade) => {
                            const isProfit = trade.pnl_net > 0;
                            const date = new Date(trade.created_at).toLocaleDateString('en-US', {
                                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                            });
                            
                            return (
                                <tr key={trade.id} className="group hover:bg-white/5 transition-colors text-sm">
                                    <td className="px-6 py-4 text-slate-400">
                                        {date}
                                    </td>
                                    <td className="px-6 py-4 font-medium text-white">
                                        {trade.symbol}
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={trade.side === 'buy' ? 'text-emerald-400' : 'text-rose-400'}>
                                            {trade.side.toUpperCase()}
                                        </span>
                                    </td>
                                    <td className={`px-6 py-4 text-right font-mono font-medium ${isProfit ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        {isProfit ? '+' : ''}{trade.pnl_net.toFixed(2)}
                                    </td>
                                    <td className="px-6 py-4 text-center">
                                        <span className={`inline-block w-2 h-2 rounded-full ${isProfit ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

