import React from 'react';
import { Trade } from '@/lib/supabase';
import { ArrowUpRight, ArrowDownRight, Filter } from 'lucide-react';
import { TableVirtuoso } from 'react-virtuoso';
import { StatusBadge } from './ui/StatusBadge';
import { cn } from '@/lib/utils';

interface RecentTradesProps {
    trades: Trade[];
}

export function RecentTrades({ trades }: RecentTradesProps) {
    return (
        <div className="glass-panel rounded-xl p-6 overflow-hidden flex flex-col h-full">
            <div className="flex items-center justify-between mb-6 shrink-0">
                <h3 className="text-lg font-semibold text-white">近期交易</h3>
                <div className="flex gap-2">
                    <button className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
                        <Filter size={16} />
                    </button>
                    <button className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-all font-medium border border-surface-border">
                        {trades.length} 笔记录
                    </button>
                </div>
            </div>

            <div className="flex-1 min-h-0">
                {trades.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500">
                        <p className="text-sm font-medium">暂无交易记录</p>
                    </div>
                ) : (
                    <TableVirtuoso
                        data={trades}
                        className="custom-scrollbar"
                        components={{
                            Table: (props) => <table {...props} className="w-full text-left text-sm border-collapse" />,
                            TableRow: (props) => <tr {...props} className="group hover:bg-white/[0.02] transition-colors border-b border-surface-border/50 last:border-0" />
                        }}
                        fixedHeaderContent={() => (
                            <tr className="text-xs uppercase text-slate-500 font-medium border-b border-surface-border bg-[#030712] z-10">
                                <th className="px-4 py-3 font-medium bg-[#030712]">时间</th>
                                <th className="px-4 py-3 font-medium bg-[#030712]">品种</th>
                                <th className="px-4 py-3 font-medium bg-[#030712]">方向</th>
                                <th className="px-4 py-3 font-medium bg-[#030712]">价格</th>
                                <th className="px-4 py-3 font-medium bg-[#030712]">数量</th>
                                <th className="px-4 py-3 font-medium text-right bg-[#030712]">盈亏</th>
                            </tr>
                        )}
                        itemContent={(index, trade) => (
                            <>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs whitespace-nowrap">
                                    {new Date(trade.created_at).toLocaleTimeString('zh-CN', {
                                        hour: '2-digit',
                                        minute: '2-digit',
                                        second: '2-digit',
                                        hour12: false
                                    })}
                                </td>
                                <td className="px-4 py-3 font-medium text-slate-200 group-hover:text-white transition-colors whitespace-nowrap">
                                    {trade.symbol}
                                </td>
                                <td className="px-4 py-3 whitespace-nowrap">
                                    <StatusBadge
                                        variant={trade.side === 'buy' ? 'success' : 'danger'}
                                        className="text-[10px] uppercase tracking-wide px-2 py-0.5"
                                    >
                                        {trade.side === 'buy' ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                                        {trade.side === 'buy' ? '买入' : '卖出'}
                                    </StatusBadge>
                                </td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs whitespace-nowrap">{trade.entry_price}</td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs whitespace-nowrap">{trade.quantity}</td>
                                <td
                                    className={cn(
                                        "px-4 py-3 text-right font-medium font-mono text-xs whitespace-nowrap",
                                        trade.pnl_net >= 0 ? 'text-accent-success' : 'text-accent-danger'
                                    )}
                                >
                                    {trade.pnl_net >= 0 ? '+' : ''}
                                    {trade.pnl_net.toFixed(2)}
                                </td>
                            </>
                        )}
                    />
                )}
            </div>
        </div>
    );
}
