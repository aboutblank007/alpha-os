"use client";

import { useMemo } from 'react';
import { Trade } from '@/lib/supabase';
import { DollarSign, Percent, TrendingUp, TrendingDown, Activity, BarChart2 } from 'lucide-react';

interface PerformanceMetricsProps {
    trades: Trade[];
}

export function PerformanceMetrics({ trades }: PerformanceMetricsProps) {
    const stats = useMemo(() => {
        if (!trades.length) return null;

        let totalPnL = 0;
        let grossProfit = 0;
        let grossLoss = 0;
        let winningTrades = 0;
        let maxDrawdown = 0;
        let peakEquity = 0;
        let currentEquity = 0;

        // Calculate basic stats
        trades.forEach(t => {
            const pnl = t.pnl_net;
            totalPnL += pnl;

            if (pnl > 0) {
                grossProfit += pnl;
                winningTrades++;
            } else {
                grossLoss += Math.abs(pnl);
            }

            // Drawdown calculation (simplified based on closed trade stream)
            currentEquity += pnl;
            if (currentEquity > peakEquity) {
                peakEquity = currentEquity;
            }
            const dd = peakEquity - currentEquity;
            if (dd > maxDrawdown) {
                maxDrawdown = dd;
            }
        });

        const totalTrades = trades.length;
        const winRate = (winningTrades / totalTrades) * 100;
        const profitFactor = grossLoss === 0 ? grossProfit : grossProfit / grossLoss;
        const avgTrade = totalPnL / totalTrades;

        // Sharpe Ratio (Simplified Annualized)
        // Avg PnL / StdDev PnL * sqrt(252 * trades_per_day)
        // Here we just use per-trade Sharpe: Avg / StdDev
        const meanPnL = totalPnL / totalTrades;
        const variance = trades.reduce((sum, t) => sum + Math.pow(t.pnl_net - meanPnL, 2), 0) / totalTrades;
        const stdDev = Math.sqrt(variance);
        const sharpe = stdDev === 0 ? 0 : meanPnL / stdDev;

        return {
            totalPnL,
            winRate,
            profitFactor,
            sharpe,
            avgTrade,
            maxDrawdown,
            trades: totalTrades
        };
    }, [trades]);

    if (!stats) return null;

    const metrics = [
        {
            label: "净利润 (Net Profit)",
            value: `$${stats.totalPnL.toFixed(2)}`,
            subValue: `${stats.trades} 笔交易`,
            icon: DollarSign,
            color: stats.totalPnL >= 0 ? "text-accent-success" : "text-accent-danger",
            bg: stats.totalPnL >= 0 ? "bg-accent-success/10" : "bg-accent-danger/10"
        },
        {
            label: "胜率 (Win Rate)",
            value: `${stats.winRate.toFixed(1)}%`,
            subValue: "盈利交易占比",
            icon: Percent,
            color: "text-accent-primary",
            bg: "bg-accent-primary/10"
        },
        {
            label: "盈亏因子 (Profit Factor)",
            value: stats.profitFactor.toFixed(2),
            subValue: "> 1.5 为优秀",
            icon: BarChart2,
            color: stats.profitFactor > 1.5 ? "text-purple-400" : "text-slate-400",
            bg: "bg-purple-500/10"
        },
        {
            label: "夏普比率 (Sharpe)",
            value: stats.sharpe.toFixed(2),
            subValue: "风险调整后收益",
            icon: Activity,
            color: stats.sharpe > 1 ? "text-orange-400" : "text-slate-400",
            bg: "bg-orange-500/10"
        },
        {
            label: "数学期望 (Expectancy)",
            value: `$${stats.avgTrade.toFixed(2)}`,
            subValue: "每笔平均收益",
            icon: TrendingUp,
            color: stats.avgTrade > 0 ? "text-emerald-400" : "text-rose-400",
            bg: "bg-slate-700/50"
        },
        {
            label: "最大回撤 (Max DD)",
            value: `$${stats.maxDrawdown.toFixed(2)}`,
            subValue: "基于平仓盈亏",
            icon: TrendingDown,
            color: "text-rose-500",
            bg: "bg-rose-500/10"
        }
    ];

    return (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {metrics.map((m, idx) => (
                <div key={idx} className="glass-panel p-3 rounded-xl flex flex-col justify-between min-h-[100px] border border-white/5 hover:border-white/10 transition-colors">
                    <div className="flex items-start justify-between mb-2">
                        <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">{m.label}</span>
                        <div className={`p-1.5 rounded-md ${m.bg} ${m.color}`}>
                            <m.icon size={14} />
                        </div>
                    </div>
                    <div>
                        <div className={`text-lg font-bold ${m.color} truncate`}>
                            {m.value}
                        </div>
                        <div className="text-[10px] text-slate-500 truncate mt-0.5">
                            {m.subValue}
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}
