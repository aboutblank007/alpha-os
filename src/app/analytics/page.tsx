"use client";

import { useEffect, useState } from 'react';
import { Card } from "@/components/Card";
import { BarChart2, PieChart, TrendingUp, Activity } from "lucide-react";
import { MaeMfeScatterChart } from "@/components/charts/MaeMfeScatterChart";
import { DrawdownChart } from "@/components/charts/DrawdownChart";

interface Trade {
    id: string;
    entry_time: string;
    pnl_net: number;
    mae: number;
    mfe: number;
    external_ticket?: string;
}

interface Stats {
    profitFactor: number;
    sharpeRatio: number;
    avgWin: number;
    winRate: number;
}

export default function AnalyticsPage() {
    // const [trades, setTrades] = useState<Trade[]>([]); // Removed unused state
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState<Stats>({
        profitFactor: 0,
        sharpeRatio: 0,
        avgWin: 0,
        winRate: 0
    });
    const [drawdownData, setDrawdownData] = useState<{ date: string; drawdown: number }[]>([]);
    const [maeMfeData, setMaeMfeData] = useState<{ ticket: string; mae: number; mfe: number; pnl: number }[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch('/api/trades?status=closed');
                const { data } = await res.json();
                if (data) {
                    processData(data);
                }
            } catch (error) {
                console.error('Error fetching trades:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const processData = (rawTrades: Trade[]) => {
        // setTrades(rawTrades); // Removed unused state update

        // 1. Calculate Basic Stats
        const wins = rawTrades.filter(t => t.pnl_net > 0);
        const losses = rawTrades.filter(t => t.pnl_net < 0);
        const totalWinPnl = wins.reduce((sum, t) => sum + t.pnl_net, 0);
        const totalLossPnl = Math.abs(losses.reduce((sum, t) => sum + t.pnl_net, 0));

        const winRate = rawTrades.length > 0 ? (wins.length / rawTrades.length) * 100 : 0;
        const profitFactor = totalLossPnl > 0 ? totalWinPnl / totalLossPnl : totalWinPnl > 0 ? 999 : 0;
        const avgWin = wins.length > 0 ? totalWinPnl / wins.length : 0;

        // Simple Sharpe Ratio (using 0 as risk free rate)
        const returns = rawTrades.map(t => t.pnl_net);
        const avgReturn = returns.reduce((a, b) => a + b, 0) / (returns.length || 1);
        const stdDev = Math.sqrt(returns.map(x => Math.pow(x - avgReturn, 2)).reduce((a, b) => a + b, 0) / (returns.length || 1));
        const sharpeRatio = stdDev > 0 ? avgReturn / stdDev : 0;

        setStats({
            profitFactor,
            sharpeRatio,
            avgWin,
            winRate
        });

        // 2. Process MAE/MFE Data
        // Filter out trades with no MAE/MFE data (or 0/0 which might be default)
        const mmData = rawTrades
            .filter(t => t.mae !== 0 || t.mfe !== 0)
            .map(t => ({
                ticket: t.external_ticket || t.id.substring(0, 8),
                mae: t.mae,  // Assuming DB stores negative value for MAE, or we invert if positive
                mfe: t.mfe,
                pnl: t.pnl_net
            }));
        setMaeMfeData(mmData);

        // 3. Process Drawdown Data
        let peak = 0;
        // const balance = 10000; // unused
        // If we want % drawdown, we usually need a base balance. 
        // Let's assume a starting balance of 10000 for calculation if not available, 
        // OR just calculate absolute drawdown from peak PnL accumulation.
        // Standard definition: Drawdown % = (Peak Equity - Current Equity) / Peak Equity

        let currentEquity = 10000; // Base balance assumption for % calc
        peak = currentEquity;

        const ddData = rawTrades.map(t => {
            currentEquity += t.pnl_net;
            if (currentEquity > peak) peak = currentEquity;

            const dd = peak > 0 ? ((currentEquity - peak) / peak) * 100 : 0;

            return {
                date: new Date(t.entry_time).toLocaleDateString(),
                drawdown: dd
            };
        });

        // Add initial point
        if (ddData.length > 0) {
            ddData.unshift({ date: 'Start', drawdown: 0 });
        }

        setDrawdownData(ddData);
    };

    if (loading) {
        return <div className="text-white text-center py-20">加载数据中...</div>;
    }

    return (
        <div className="space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">数据分析</h1>
                    <p className="text-slate-400 mt-2">深入分析您的交易指标</p>
                </div>
                <div className="flex gap-2 bg-surface-glass p-1 rounded-xl border border-surface-border">
                    <button className="px-4 py-2 rounded-lg bg-accent-primary text-white shadow-lg shadow-accent-primary/20">概览</button>
                    <button className="px-4 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all">表现</button>
                    <button className="px-4 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all">风险</button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                    { label: '盈亏比', value: stats.profitFactor.toFixed(2), icon: TrendingUp, color: 'text-accent-success' },
                    { label: '夏普比率', value: stats.sharpeRatio.toFixed(2), icon: Activity, color: 'text-accent-primary' },
                    { label: '平均盈利', value: `$${stats.avgWin.toFixed(2)}`, icon: BarChart2, color: 'text-accent-cyan' },
                    { label: '胜率', value: `${stats.winRate.toFixed(1)}%`, icon: PieChart, color: 'text-accent-secondary' },
                ].map((stat, i) => (
                    <Card key={i} className="hover:scale-105 transition-transform duration-300">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-sm font-medium text-slate-400">{stat.label}</p>
                                <h3 className={`text-2xl font-bold mt-2 ${stat.color}`}>{stat.value}</h3>
                            </div>
                            <div className={`p-3 rounded-xl bg-white/5 ${stat.color}`}>
                                <stat.icon size={20} />
                            </div>
                        </div>
                    </Card>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <Card>
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-semibold text-white">MAE / MFE 分析</h3>
                        <div className="text-xs text-slate-400 bg-white/5 px-2 py-1 rounded">
                            基于 {maeMfeData.length} 笔交易
                        </div>
                    </div>
                    <div className="h-80 w-full">
                        {maeMfeData.length > 0 ? (
                            <MaeMfeScatterChart data={maeMfeData} height={320} />
                        ) : (
                            <div className="h-full flex items-center justify-center text-slate-500">
                                暂无 MAE/MFE 数据
                            </div>
                        )}
                    </div>
                </Card>

                <Card>
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-semibold text-white">动态回撤分析</h3>
                        <div className="text-xs text-slate-400 bg-white/5 px-2 py-1 rounded">
                            最大回撤: {Math.min(...drawdownData.map(d => d.drawdown)).toFixed(2)}%
                        </div>
                    </div>
                    <div className="h-80 w-full">
                        {drawdownData.length > 0 ? (
                            <DrawdownChart data={drawdownData} height={320} />
                        ) : (
                            <div className="h-full flex items-center justify-center text-slate-500">
                                暂无回撤数据
                            </div>
                        )}
                    </div>
                </Card>
            </div>
        </div>
    );
}
