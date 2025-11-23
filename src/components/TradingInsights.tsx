import { Trade } from '@/lib/supabase';
import { Lightbulb, TrendingUp, TrendingDown, Award, AlertCircle } from 'lucide-react';

interface TradingInsightsProps {
    trades: Trade[];
}

export function TradingInsights({ trades }: TradingInsightsProps) {
    const closedTrades = trades.filter(t => t.status === 'closed');

    const bestTrade = closedTrades.length > 0 
        ? closedTrades.reduce((prev, current) => (prev.pnl_net > current.pnl_net) ? prev : current)
        : null;

    const worstTrade = closedTrades.length > 0 
        ? closedTrades.reduce((prev, current) => (prev.pnl_net < current.pnl_net) ? prev : current)
        : null;

    const winningTrades = closedTrades.filter(t => t.pnl_net > 0);
    const losingTrades = closedTrades.filter(t => t.pnl_net < 0);

    const avgWin = winningTrades.length > 0
        ? winningTrades.reduce((sum, t) => sum + t.pnl_net, 0) / winningTrades.length
        : 0;

    const avgLoss = losingTrades.length > 0
        ? losingTrades.reduce((sum, t) => sum + t.pnl_net, 0) / losingTrades.length
        : 0;

    // Calculate Win Streak (most recent consecutive wins)
    let currentWinStreak = 0;
    // Sort by date descending for streak calculation
    const sortedByDateDesc = [...closedTrades].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    for (const trade of sortedByDateDesc) {
        if (trade.pnl_net > 0) {
            currentWinStreak++;
        } else {
            break;
        }
    }

    return (
        <div className="glass-panel rounded-xl p-4 h-full flex flex-col overflow-hidden">
            <div className="flex items-center gap-2 mb-3 flex-shrink-0">
                <Lightbulb className="text-accent-warning" size={18} />
                <h3 className="text-base font-semibold text-white">交易洞察</h3>
            </div>

            <div className="space-y-2 flex-1 flex flex-col justify-between min-h-0">
                {closedTrades.length === 0 ? (
                    <p className="text-slate-500 text-sm">数据不足，无法生成洞察。</p>
                ) : (
                    <>
                        <div className="p-2.5 rounded-lg bg-white/5 border border-white/5">
                            <div className="flex items-start gap-2.5">
                                <div className="p-1.5 rounded-md bg-accent-success/10 text-accent-success flex-shrink-0">
                                    <Award size={16} />
                                </div>
                                <div className="min-w-0 flex-1">
                                    <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wider mb-0.5">最佳交易</p>
                                    <div className="flex items-baseline gap-1.5">
                                        <span className="text-base font-bold text-white whitespace-nowrap">
                                            +${bestTrade?.pnl_net.toFixed(2)}
                                        </span>
                                        <span className="text-[10px] text-slate-500 truncate">{bestTrade?.symbol}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="p-2.5 rounded-lg bg-white/5 border border-white/5">
                            <div className="flex items-start gap-2.5">
                                <div className="p-1.5 rounded-md bg-accent-danger/10 text-accent-danger flex-shrink-0">
                                    <AlertCircle size={16} />
                                </div>
                                <div className="min-w-0 flex-1">
                                    <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wider mb-0.5">最差交易</p>
                                    <div className="flex items-baseline gap-1.5">
                                        <span className="text-base font-bold text-white whitespace-nowrap">
                                            ${worstTrade?.pnl_net.toFixed(2)}
                                        </span>
                                        <span className="text-[10px] text-slate-500 truncate">{worstTrade?.symbol}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <div className="p-2 rounded-lg bg-white/5 border border-white/5 min-w-0">
                                <p className="text-[10px] text-slate-400 mb-0.5 truncate">平均盈利</p>
                                <div className="flex items-center gap-1 text-accent-success font-medium text-xs">
                                    <TrendingUp size={11} className="flex-shrink-0" />
                                    <span className="truncate">${avgWin.toFixed(2)}</span>
                                </div>
                            </div>
                            <div className="p-2 rounded-lg bg-white/5 border border-white/5 min-w-0">
                                <p className="text-[10px] text-slate-400 mb-0.5 truncate">平均亏损</p>
                                <div className="flex items-center gap-1 text-accent-danger font-medium text-xs">
                                    <TrendingDown size={11} className="flex-shrink-0" />
                                    <span className="truncate">${avgLoss.toFixed(2)}</span>
                                </div>
                            </div>
                        </div>
                        
                        {currentWinStreak > 1 && (
                             <div className="p-2 rounded-lg bg-gradient-to-r from-accent-primary/20 to-transparent border border-accent-primary/20">
                                <p className="text-[10px] text-accent-primary font-bold uppercase tracking-wider mb-0.5">🔥 连胜中</p>
                                <p className="text-xs text-white leading-tight">
                                    您正处于 <span className="font-bold">{currentWinStreak}</span> 笔连胜！
                                </p>
                             </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
