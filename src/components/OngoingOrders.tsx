import { ArrowUpRight, ArrowDownRight, Clock, WifiOff, Zap, XCircle } from 'lucide-react';
import { useState } from 'react';
import { useMarketStore } from '@/store/useMarketStore';
import { useTradeStore, MT5Position } from '@/store/useTradeStore';

export function OngoingOrders() {
    // Use Stores directly
    const isBridgeConnected = useMarketStore(state => state.isConnected);
    const positions = useTradeStore(state => state.positions);

    const [executingTicket, setExecutingTicket] = useState<number | null>(null);

    // 平仓函数
    const handleClose = async (pos: MT5Position) => {
        if (executingTicket) return;
        if (!confirm(`确认平仓 #${pos.ticket} (${pos.symbol})?`)) return;

        setExecutingTicket(pos.ticket);
        try {
            const res = await fetch('/api/bridge/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'CLOSE',
                    ticket: pos.ticket,
                    symbol: pos.symbol,
                    volume: pos.volume
                })
            });

            // We should assume success if we get a response, even if not 200, because EA might process it.
            // But ideally we check response.
            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || 'Failed to close position');
            }

            // 乐观更新：虽然下次轮询会更新，但我们可以先从列表移除
            // setPositions(prev => prev.filter(p => p.ticket !== ticket));
            // Note: With Zustand, we rely on the next poll update or we can invoke an action to optimistically remove.
            // For now, we wait for the poll to update the store.

        } catch (e) {
            alert('平仓失败: ' + (e instanceof Error ? e.message : '未知错误'));
        } finally {
            setExecutingTicket(null);
        }
    };

    return (
        <div className="glass-panel rounded-xl p-6 overflow-hidden flex flex-col h-full">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Clock size={18} className="text-accent-primary" />
                    MT5 持仓
                </h3>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5" title="MT5 Bridge Status">
                        {isBridgeConnected ? (
                            <Zap size={14} className="text-accent-warning fill-accent-warning" />
                        ) : (
                            <WifiOff size={14} className="text-slate-600" />
                        )}
                        <span className="text-xs text-slate-400">
                            {isBridgeConnected ? '实时同步' : '连接断开'}
                        </span>
                    </div>
                    <span className="text-xs font-medium px-2 py-1 rounded bg-white/5 text-slate-400">
                        {positions.length} 活跃
                    </span>
                </div>
            </div>

            <div className="overflow-x-auto flex-1 custom-scrollbar">
                <table className="w-full text-left text-sm">
                    <thead>
                        <tr className="text-xs uppercase text-slate-500 font-medium border-b border-white/5">
                            <th className="px-4 py-3 font-medium">Ticket</th>
                            <th className="px-4 py-3 font-medium">品种</th>
                            <th className="px-4 py-3 font-medium">方向</th>
                            <th className="px-4 py-3 font-medium">手数</th>
                            <th className="px-4 py-3 font-medium">开仓价</th>
                            <th className="px-4 py-3 font-medium">现价</th>
                            <th className="px-4 py-3 font-medium">SL / TP</th>
                            <th className="px-4 py-3 font-medium text-right">盈亏</th>
                            <th className="px-4 py-3 font-medium text-right">操作</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {positions.map((pos) => {
                            const pnl = pos.pnl + pos.swap; // 总盈亏含隔夜利息

                            return (
                                <tr
                                    key={pos.ticket}
                                    className="group hover:bg-white/[0.02] transition-colors"
                                >
                                    <td className="px-4 py-3 text-slate-500 font-mono text-xs">
                                        #{pos.ticket}
                                    </td>
                                    <td className="px-4 py-3 font-medium text-slate-200 group-hover:text-white transition-colors">
                                        {pos.symbol}
                                    </td>
                                    <td className="px-4 py-3">
                                        <span
                                            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${pos.type === 'BUY'
                                                ? 'bg-accent-success/10 text-accent-success'
                                                : 'bg-accent-danger/10 text-accent-danger'
                                                }`}
                                        >
                                            {pos.type === 'BUY' ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                                            {pos.type === 'BUY' ? '买入' : '卖出'}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-slate-300 font-mono text-xs">{pos.volume}</td>
                                    <td className="px-4 py-3 text-slate-400 font-mono text-xs">{pos.open_price}</td>
                                    <td className="px-4 py-3 text-slate-400 font-mono text-xs">{pos.current_price}</td>
                                    <td className="px-4 py-3 text-slate-500 font-mono text-xs">
                                        <div>SL: <span className={pos.sl > 0 ? 'text-accent-danger' : ''}>{pos.sl > 0 ? pos.sl : '-'}</span></div>
                                        <div>TP: <span className={pos.tp > 0 ? 'text-accent-success' : ''}>{pos.tp > 0 ? pos.tp : '-'}</span></div>
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                        <div className="flex flex-col items-end gap-0.5">
                                            <span className={`text-xs font-mono font-bold ${pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                                            </span>
                                            <span className="text-[10px] text-slate-600">
                                                Swap: {pos.swap}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                        <button
                                            onClick={() => handleClose(pos)}
                                            disabled={executingTicket === pos.ticket}
                                            className="p-1 hover:bg-white/10 rounded text-slate-400 hover:text-accent-danger transition-colors disabled:opacity-50"
                                            title="平仓"
                                        >
                                            <XCircle size={16} />
                                        </button>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                {positions.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-40 text-slate-500">
                        <p className="text-sm font-medium">
                            {isBridgeConnected ? '暂无持仓' : '等待 MT5 连接...'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
