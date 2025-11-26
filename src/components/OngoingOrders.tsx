import { Clock, WifiOff, Zap, XCircle } from 'lucide-react';
import { useState } from 'react';
import { useMarketStore } from '@/store/useMarketStore';
import { useTradeStore, MT5Position } from '@/store/useTradeStore';

interface OngoingOrdersProps {
    compact?: boolean;
}

export function OngoingOrders({ compact = false }: OngoingOrdersProps) {
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

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || 'Failed to close position');
            }
        } catch (e) {
            alert('平仓失败: ' + (e instanceof Error ? e.message : '未知错误'));
        } finally {
            setExecutingTicket(null);
        }
    };

    return (
        <div className={`overflow-hidden flex flex-col h-full ${compact ? 'bg-transparent' : 'glass-panel rounded-xl p-6'}`}>
            {!compact && (
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
            )}

            {/* Compact Header */}
            {compact && (
                <div className="px-4 py-2 border-b border-white/5 flex justify-between items-center bg-white/5 backdrop-blur-sm">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">当前持仓 ({positions.length})</span>
                </div>
            )}

            <div className={`overflow-x-auto flex-1 custom-scrollbar ${compact ? 'p-0' : ''}`}>
                <table className="w-full text-left text-sm">
                    <thead>
                        <tr className="text-xs uppercase text-slate-500 font-medium border-b border-white/5">
                            <th className="px-4 py-2 font-medium">品种</th>
                            <th className="px-4 py-2 font-medium text-right">盈亏</th>
                            <th className="px-4 py-2 font-medium text-right">操作</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {positions.map((pos) => {
                            const pnl = pos.pnl + pos.swap; 
                            return (
                                <tr key={pos.ticket} className="group hover:bg-white/[0.02]">
                                    <td className="px-4 py-2.5">
                                        <div className="flex flex-col">
                                            <span className="font-medium text-slate-200">{pos.symbol}</span>
                                            <div className="flex items-center gap-1 mt-0.5">
                                                <span className={`text-[10px] font-bold px-1 rounded ${pos.type === 'BUY' ? 'bg-blue-500/20 text-blue-400' : 'bg-red-500/20 text-red-400'}`}>
                                                    {pos.type}
                                        </span>
                                                <span className="text-[10px] text-slate-500">{pos.volume}</span>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-2.5 text-right">
                                        <div className={`font-mono font-bold ${pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                                        </div>
                                    </td>
                                    <td className="px-4 py-2.5 text-right">
                                        <button
                                            onClick={() => handleClose(pos)}
                                            disabled={executingTicket === pos.ticket}
                                            className="p-1.5 bg-white/5 hover:bg-red-500/20 rounded text-slate-400 hover:text-red-400 transition-colors"
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
                    <div className="flex flex-col items-center justify-center h-24 text-slate-500 text-xs">
                        暂无持仓
                    </div>
                )}
            </div>
        </div>
    );
}
