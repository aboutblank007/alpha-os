import { Trade } from '@/lib/supabase';
import { ArrowUpRight, ArrowDownRight, Filter } from 'lucide-react';

interface RecentTradesProps {
    trades: Trade[];
}

export function RecentTrades({ trades }: RecentTradesProps) {
    return (
        <div className="glass-panel rounded-xl p-6 overflow-hidden flex flex-col h-full">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">近期交易</h3>
                <div className="flex gap-2">
                    <button className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
                        <Filter size={16} />
                    </button>
                    <button className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-all font-medium border border-white/5">
                        查看全部
                    </button>
                </div>
            </div>

            <div className="overflow-x-auto flex-1 custom-scrollbar">
                <table className="w-full text-left text-sm">
                    <thead>
                        <tr className="text-xs uppercase text-slate-500 font-medium border-b border-white/5">
                            <th className="px-4 py-3 font-medium">时间</th>
                            <th className="px-4 py-3 font-medium">品种</th>
                            <th className="px-4 py-3 font-medium">方向</th>
                            <th className="px-4 py-3 font-medium">价格</th>
                            <th className="px-4 py-3 font-medium">数量</th>
                            <th className="px-4 py-3 font-medium text-right">盈亏</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {trades.slice(0, 8).map((trade) => (
                            <tr
                                key={trade.id}
                                className="group hover:bg-white/[0.02] transition-colors"
                            >
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs">
                                    {new Date(trade.created_at).toLocaleTimeString('zh-CN', {
                                        hour: '2-digit',
                                        minute: '2-digit',
                                        hour12: false
                                    })}
                                </td>
                                <td className="px-4 py-3 font-medium text-slate-200 group-hover:text-white transition-colors">
                                    {trade.symbol}
                                </td>
                                <td className="px-4 py-3">
                                    <span
                                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${trade.side === 'buy'
                                            ? 'bg-accent-success/10 text-accent-success'
                                            : 'bg-accent-danger/10 text-accent-danger'
                                            }`}
                                    >
                                        {trade.side === 'buy' ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                                        {trade.side === 'buy' ? '买入' : '卖出'}
                                    </span>
                                </td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs">{trade.entry_price}</td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs">{trade.quantity}</td>
                                <td
                                    className={`px-4 py-3 text-right font-medium font-mono text-xs ${trade.pnl_net >= 0
                                        ? 'text-accent-success'
                                        : 'text-accent-danger'
                                        }`}
                                >
                                    {trade.pnl_net >= 0 ? '+' : ''}
                                    {trade.pnl_net.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {trades.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-64 text-slate-500">
                        <p className="text-sm font-medium">暂无交易记录</p>
                    </div>
                )}
            </div>
        </div>
    );
}
