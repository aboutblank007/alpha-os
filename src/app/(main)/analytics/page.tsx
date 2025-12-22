"use client";

import React, { useState, useEffect, useMemo } from "react";
import { GlassCard, CardHeader, CardTitle, CardContent } from "@/components/ui/GlassCard";
import { Button } from "@/components/ui/Button";
import { 
    TrendingUp, TrendingDown, BarChart3, PieChart, Activity, 
    Calendar, Target, Zap, Award, AlertTriangle, Clock, 
    DollarSign, ArrowUpRight, ArrowDownRight, RefreshCw
} from "lucide-react";
import { cn } from "@/lib/utils";

// 交易数据类型
interface Trade {
    id: string;
    symbol: string;
    type: 'BUY' | 'SELL';
    profit: number;
    volume: number;
    entryPrice: number;
    exitPrice: number;
    openTime: string;
    closeTime: string;
    swap: number;
    commission: number;
}

// 统计指标卡片
function StatCard({ 
    icon: Icon, 
    label, 
    value, 
    subValue, 
    trend, 
    color = "primary" 
}: { 
    icon: React.ElementType;
    label: string;
    value: string;
    subValue?: string;
    trend?: 'up' | 'down' | 'neutral';
    color?: 'primary' | 'success' | 'danger' | 'warning';
}) {
    const colorMap = {
        primary: 'text-primary',
        success: 'text-success',
        danger: 'text-danger',
        warning: 'text-warning',
    };
    
    return (
        <GlassCard className="p-4 relative overflow-hidden group hover:border-primary/30 transition-all">
            <div className="absolute top-0 right-0 p-3 opacity-5 group-hover:opacity-10 transition-opacity">
                <Icon size={50} />
            </div>
            <div className="flex items-center gap-2 mb-1">
                <Icon size={14} className={colorMap[color]} />
                <span className="text-[10px] font-medium text-text-muted uppercase tracking-wider">{label}</span>
            </div>
            <div className="flex items-end gap-2">
                <span className={cn("text-xl font-bold", colorMap[color])}>{value}</span>
                {trend && (
                    <span className={cn("flex items-center text-[10px] mb-0.5", 
                        trend === 'up' ? 'text-success' : trend === 'down' ? 'text-danger' : 'text-text-muted'
                    )}>
                        {trend === 'up' ? <ArrowUpRight size={12} /> : trend === 'down' ? <ArrowDownRight size={12} /> : null}
                        {subValue}
                    </span>
                )}
                {!trend && subValue && (
                    <span className="text-[10px] text-text-muted mb-0.5">{subValue}</span>
                )}
            </div>
        </GlassCard>
    );
}

// 简易柱状图组件
function MiniBarChart({ data, height = 100 }: { data: number[]; height?: number }) {
    const max = Math.max(...data.map(Math.abs), 0.01);
    
    return (
        <div className="flex items-end justify-between gap-0.5 h-full" style={{ height }}>
            {data.map((value, i) => {
                const h = max > 0 ? (Math.abs(value) / max) * 100 : 0;
                const isPositive = value >= 0;
                return (
                    <div key={i} className="flex-1 flex flex-col justify-end items-center group cursor-pointer">
                        <div 
                            className={cn(
                                "w-full rounded-t transition-all group-hover:opacity-80",
                                isPositive ? "bg-success" : "bg-danger"
                            )}
                            style={{ height: `${h}%`, minHeight: value !== 0 ? 2 : 0 }}
                            title={`$${value.toFixed(2)}`}
                        />
                    </div>
                );
            })}
        </div>
    );
}

// 资金曲线图组件
function EquityCurve({ trades, initialEquity = 10000 }: { trades: Trade[]; initialEquity?: number }) {
    const equityData = useMemo(() => {
        // 按时间排序（从早到晚）
        const sorted = [...trades].sort((a, b) => 
            new Date(a.closeTime || a.openTime).getTime() - new Date(b.closeTime || b.openTime).getTime()
        );
        
        let equity = initialEquity;
        return sorted.map(t => {
            equity += t.profit;
            return equity;
        });
    }, [trades, initialEquity]);
    
    if (equityData.length === 0) {
        return <div className="h-full flex items-center justify-center text-text-muted">暂无数据</div>;
    }
    
    const min = Math.min(...equityData) * 0.98;
    const max = Math.max(...equityData) * 1.02;
    const range = max - min || 1;
    
    const points = equityData.map((v, i) => {
        const x = (i / Math.max(equityData.length - 1, 1)) * 100;
        const y = 100 - ((v - min) / range) * 100;
        return `${x},${y}`;
    }).join(' ');
    
    const currentEquity = equityData[equityData.length - 1];
    const totalReturn = ((currentEquity - initialEquity) / initialEquity) * 100;
    
    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-3">
                <div>
                    <div className="text-xl font-bold text-text-primary">${currentEquity.toFixed(2)}</div>
                    <div className={cn("text-xs font-medium", totalReturn >= 0 ? "text-success" : "text-danger")}>
                        {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}% 总收益 ({trades.length} 笔)
                    </div>
                </div>
            </div>
            
            <div className="flex-1 relative min-h-[150px]">
                <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <defs>
                        <linearGradient id="equityGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="var(--color-primary)" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="var(--color-primary)" stopOpacity="0" />
                        </linearGradient>
                    </defs>
                    
                    <polygon 
                        points={`0,100 ${points} 100,100`}
                        fill="url(#equityGradient)"
                    />
                    
                    <polyline
                        points={points}
                        fill="none"
                        stroke="var(--color-primary)"
                        strokeWidth="0.5"
                        vectorEffect="non-scaling-stroke"
                    />
                </svg>
                
                <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[9px] text-text-muted font-mono">
                    <span>${max.toFixed(0)}</span>
                    <span>${min.toFixed(0)}</span>
                </div>
            </div>
        </div>
    );
}

// 品种表现表格
function SymbolPerformance({ trades }: { trades: Trade[] }) {
    const symbolStats = useMemo(() => {
        const stats: Record<string, { trades: number; profit: number; winRate: number }> = {};
        
        trades.forEach(t => {
            if (!stats[t.symbol]) {
                stats[t.symbol] = { trades: 0, profit: 0, winRate: 0 };
            }
            stats[t.symbol].trades++;
            stats[t.symbol].profit += t.profit;
        });
        
        Object.keys(stats).forEach(symbol => {
            const wins = trades.filter(t => t.symbol === symbol && t.profit > 0).length;
            stats[symbol].winRate = stats[symbol].trades > 0 ? (wins / stats[symbol].trades) * 100 : 0;
        });
        
        return Object.entries(stats)
            .sort((a, b) => b[1].profit - a[1].profit)
            .slice(0, 6)
            .map(([symbol, data]) => ({ symbol, ...data }));
    }, [trades]);
    
    if (symbolStats.length === 0) {
        return <div className="text-center text-text-muted py-4">暂无数据</div>;
    }
    
    return (
        <div className="space-y-2">
            {symbolStats.map(({ symbol, trades: count, profit, winRate }) => (
                <div key={symbol} className="flex items-center gap-3 p-2 bg-bg-subtle/50 rounded-lg hover:bg-bg-subtle transition-colors">
                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <span className="text-xs font-bold text-primary">{symbol.substring(0, 2)}</span>
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                            <span className="font-medium text-sm text-text-primary truncate">{symbol}</span>
                            <span className={cn("font-mono text-sm font-bold", profit >= 0 ? "text-success" : "text-danger")}>
                                {profit >= 0 ? '+' : ''}${profit.toFixed(2)}
                            </span>
                        </div>
                        <div className="flex items-center justify-between mt-0.5">
                            <span className="text-[10px] text-text-muted">{count} 笔</span>
                            <span className="text-[10px] text-text-muted">{winRate.toFixed(0)}% 胜率</span>
                        </div>
                        <div className="h-1 mt-1 bg-bg-base rounded-full overflow-hidden">
                            <div 
                                className={cn("h-full rounded-full transition-all", profit >= 0 ? "bg-success" : "bg-danger")}
                                style={{ width: `${Math.min(winRate, 100)}%` }}
                            />
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}

// 交易时段分布
function TradingHoursChart({ trades }: { trades: Trade[] }) {
    const hourlyData = useMemo(() => {
        const hours = Array(24).fill(0);
        trades.forEach(t => {
            const hour = new Date(t.openTime).getHours();
            hours[hour] += t.profit;
        });
        return hours;
    }, [trades]);
    
    return (
        <div className="h-full flex flex-col">
            <div className="flex-1">
                <MiniBarChart data={hourlyData} height={100} />
            </div>
            <div className="flex justify-between mt-2 text-[9px] text-text-muted font-mono">
                <span>00</span>
                <span>06</span>
                <span>12</span>
                <span>18</span>
                <span>24</span>
            </div>
        </div>
    );
}

// 盈亏分布饼图
function ProfitDistribution({ trades }: { trades: Trade[] }) {
    const stats = useMemo(() => {
        const wins = trades.filter(t => t.profit > 0);
        const losses = trades.filter(t => t.profit < 0);
        
        return {
            wins: wins.length,
            losses: losses.length,
            winRate: trades.length > 0 ? (wins.length / trades.length) * 100 : 0,
            avgWin: wins.length > 0 ? wins.reduce((s, t) => s + t.profit, 0) / wins.length : 0,
            avgLoss: losses.length > 0 ? losses.reduce((s, t) => s + t.profit, 0) / losses.length : 0,
        };
    }, [trades]);
    
    const winAngle = (stats.winRate / 100) * 360;
    
    return (
        <div className="flex items-center gap-4">
            <div className="relative w-24 h-24 shrink-0">
                <svg viewBox="0 0 100 100" className="transform -rotate-90">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="var(--color-danger)" strokeWidth="12" opacity="0.3" />
                    <circle 
                        cx="50" cy="50" r="40" 
                        fill="none" 
                        stroke="var(--color-success)" 
                        strokeWidth="12"
                        strokeDasharray={`${(winAngle / 360) * 251.2} 251.2`}
                        strokeLinecap="round"
                    />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center flex-col">
                    <span className="text-lg font-bold text-success">{stats.winRate.toFixed(0)}%</span>
                    <span className="text-[9px] text-text-muted">胜率</span>
                </div>
            </div>
            
            <div className="flex-1 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-success" />
                        <span className="text-text-secondary">盈利</span>
                    </div>
                    <span className="font-mono">{stats.wins} 笔</span>
                </div>
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-danger" />
                        <span className="text-text-secondary">亏损</span>
                    </div>
                    <span className="font-mono">{stats.losses} 笔</span>
                </div>
                <div className="pt-1.5 border-t border-white/5 space-y-1">
                    <div className="flex justify-between">
                        <span className="text-text-muted">均盈</span>
                        <span className="text-success font-mono">+${stats.avgWin.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-text-muted">均亏</span>
                        <span className="text-danger font-mono">${stats.avgLoss.toFixed(2)}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

// 最近交易列表
function RecentTrades({ trades }: { trades: Trade[] }) {
    const recentTrades = trades.slice(0, 8);
    
    if (recentTrades.length === 0) {
        return <div className="text-center text-text-muted py-4">暂无交易</div>;
    }
    
    return (
        <div className="space-y-1.5">
            {recentTrades.map((trade) => (
                <div key={trade.id} className="flex items-center gap-2 p-1.5 hover:bg-bg-subtle/50 rounded transition-colors">
                    <div className={cn(
                        "w-6 h-6 rounded flex items-center justify-center",
                        trade.type === 'BUY' ? "bg-success/10 text-success" : "bg-danger/10 text-danger"
                    )}>
                        {trade.type === 'BUY' ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                            <span className="font-medium text-xs truncate">{trade.symbol}</span>
                            <span className={cn(
                                "font-mono text-xs font-bold",
                                trade.profit >= 0 ? "text-success" : "text-danger"
                            )}>
                                {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-[9px] text-text-muted">{trade.volume} lots</span>
                            <span className="text-[9px] text-text-muted">
                                {new Date(trade.openTime).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })}
                            </span>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}

// 每日盈亏图表
function DailyPnLChart({ trades }: { trades: Trade[] }) {
    const dailyData = useMemo(() => {
        const days: Record<string, number> = {};
        
        trades.forEach(t => {
            const date = new Date(t.closeTime || t.openTime).toISOString().split('T')[0];
            days[date] = (days[date] || 0) + t.profit;
        });
        
        // 取最近30天
        return Object.entries(days)
            .sort((a, b) => a[0].localeCompare(b[0]))
            .slice(-30)
            .map(([, pnl]) => pnl);
    }, [trades]);
    
    if (dailyData.length === 0) {
        return <div className="h-full flex items-center justify-center text-text-muted text-sm">暂无数据</div>;
    }
    
    return (
        <div className="h-full flex flex-col">
            <div className="flex-1">
                <MiniBarChart data={dailyData} height={80} />
            </div>
            <div className="text-center mt-2 text-[9px] text-text-muted">最近 {dailyData.length} 天</div>
        </div>
    );
}

export default function AnalyticsPage() {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    
    const fetchTrades = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch('/api/analytics/trades?limit=1000&status=closed');
            if (!res.ok) throw new Error('Failed to fetch');
            const data = await res.json();
            setTrades(data.trades || []);
        } catch (e) {
            console.error(e);
            setError('加载数据失败');
        } finally {
            setIsLoading(false);
        }
    };
    
    useEffect(() => {
        fetchTrades();
    }, []);
    
    // 计算统计数据
    const stats = useMemo(() => {
        if (trades.length === 0) return null;
        
        const totalProfit = trades.reduce((s, t) => s + t.profit, 0);
        const wins = trades.filter(t => t.profit > 0);
        const losses = trades.filter(t => t.profit < 0);
        
        // 计算最大回撤
        let peak = 10000;
        let maxDrawdown = 0;
        let equity = 10000;
        
        const sorted = [...trades].sort((a, b) => 
            new Date(a.closeTime || a.openTime).getTime() - new Date(b.closeTime || b.openTime).getTime()
        );
        
        sorted.forEach(t => {
            equity += t.profit;
            if (equity > peak) peak = equity;
            const dd = peak > 0 ? (peak - equity) / peak * 100 : 0;
            if (dd > maxDrawdown) maxDrawdown = dd;
        });
        
        // 盈亏比
        const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + t.profit, 0) / wins.length : 0;
        const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((s, t) => s + t.profit, 0) / losses.length) : 1;
        const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;
        
        // Sharpe (简化)
        const returns = trades.map(t => t.profit / 100);
        const avgReturn = returns.reduce((s, r) => s + r, 0) / returns.length;
        const variance = returns.reduce((s, r) => s + Math.pow(r - avgReturn, 2), 0) / returns.length;
        const stdDev = Math.sqrt(variance);
        const sharpe = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;
        
        return {
            totalProfit,
            winRate: (wins.length / trades.length) * 100,
            maxDrawdown,
            profitFactor,
            sharpe,
            totalTrades: trades.length,
            totalVolume: trades.reduce((s, t) => s + t.volume, 0),
        };
    }, [trades]);
    
    if (isLoading) {
        return (
            <div className="h-full flex items-center justify-center">
                <Activity className="animate-spin text-primary" size={32} />
            </div>
        );
    }
    
    return (
        <div className="h-full flex flex-col gap-3 overflow-auto pb-4">
            {/* 页面标题 */}
            <div className="flex items-center justify-between shrink-0">
                <div>
                    <h1 className="text-xl font-bold tracking-tight">绩效分析</h1>
                    <p className="text-xs text-text-muted mt-0.5">
                        {trades.length > 0 ? `共 ${trades.length} 笔历史交易` : '暂无交易数据'}
                    </p>
                </div>
                <div className="flex gap-2">
                    <Button variant="ghost" size="sm" onClick={fetchTrades} disabled={isLoading}>
                        <RefreshCw size={14} className={cn("mr-1", isLoading && "animate-spin")} /> 刷新
                    </Button>
                    <Button variant="secondary" size="sm">
                        <Calendar size={14} className="mr-1" /> 筛选
                    </Button>
                </div>
            </div>
            
            {error && (
                <div className="bg-danger/10 text-danger text-sm p-3 rounded-lg">{error}</div>
            )}
            
            {/* 核心指标卡片 */}
            <div className="grid grid-cols-3 md:grid-cols-6 gap-2 shrink-0">
                <StatCard 
                    icon={DollarSign}
                    label="总盈亏"
                    value={`${stats?.totalProfit && stats.totalProfit >= 0 ? '+' : ''}$${stats?.totalProfit.toFixed(2) || '0'}`}
                    subValue={`${stats?.totalTrades || 0}笔`}
                    trend={stats?.totalProfit && stats.totalProfit >= 0 ? 'up' : 'down'}
                    color={stats?.totalProfit && stats.totalProfit >= 0 ? 'success' : 'danger'}
                />
                <StatCard 
                    icon={Target}
                    label="胜率"
                    value={`${stats?.winRate.toFixed(1) || 0}%`}
                    color="primary"
                />
                <StatCard 
                    icon={AlertTriangle}
                    label="最大回撤"
                    value={`${stats?.maxDrawdown.toFixed(1) || 0}%`}
                    color="warning"
                />
                <StatCard 
                    icon={Zap}
                    label="盈亏比"
                    value={`${stats?.profitFactor.toFixed(2) || 0}`}
                    color={stats?.profitFactor && stats.profitFactor >= 1 ? 'success' : 'danger'}
                />
                <StatCard 
                    icon={Award}
                    label="Sharpe"
                    value={stats?.sharpe.toFixed(2) || '0'}
                    color={stats?.sharpe && stats.sharpe >= 1 ? 'success' : 'warning'}
                />
                <StatCard 
                    icon={Activity}
                    label="总手数"
                    value={`${stats?.totalVolume.toFixed(2) || 0}`}
                    color="primary"
                />
                    </div>
            
            {/* 主要图表区域 */}
            <div className="grid grid-cols-12 gap-3 flex-1 min-h-0">
                {/* 资金曲线 */}
                <GlassCard className="col-span-12 lg:col-span-8 p-4">
                    <CardHeader className="px-0 pt-0 pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                            <Activity size={16} className="text-primary" />
                            资金曲线
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0 h-[calc(100%-40px)]">
                        <EquityCurve trades={trades} />
                    </CardContent>
                </GlassCard>
                
                {/* 盈亏分布 */}
                <GlassCard className="col-span-12 lg:col-span-4 p-4">
                    <CardHeader className="px-0 pt-0 pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                            <PieChart size={16} className="text-primary" />
                            盈亏分布
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                        <ProfitDistribution trades={trades} />
                    </CardContent>
                </GlassCard>
                
                {/* 品种表现 */}
                <GlassCard className="col-span-12 md:col-span-4 p-4 max-h-[300px] overflow-auto">
                    <CardHeader className="px-0 pt-0 pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                            <BarChart3 size={16} className="text-primary" />
                            品种表现
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                        <SymbolPerformance trades={trades} />
                    </CardContent>
                </GlassCard>
                
                {/* 每日盈亏 */}
                <GlassCard className="col-span-12 md:col-span-4 p-4">
                    <CardHeader className="px-0 pt-0 pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                            <Calendar size={16} className="text-primary" />
                            每日盈亏
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0 h-[calc(100%-40px)]">
                        <DailyPnLChart trades={trades} />
                    </CardContent>
                </GlassCard>
                
                {/* 最近交易 */}
                <GlassCard className="col-span-12 md:col-span-4 p-4 max-h-[300px] overflow-auto">
                    <CardHeader className="px-0 pt-0 pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                            <TrendingUp size={16} className="text-primary" />
                            最近交易
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                        <RecentTrades trades={trades} />
                    </CardContent>
                </GlassCard>
                
                {/* 交易时段 */}
                <GlassCard className="col-span-12 p-4">
                    <CardHeader className="px-0 pt-0 pb-2">
                        <CardTitle className="flex items-center gap-2 text-sm">
                            <Clock size={16} className="text-primary" />
                            交易时段分布 (按小时)
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0 h-[120px]">
                        <TradingHoursChart trades={trades} />
                    </CardContent>
                </GlassCard>
            </div>
        </div>
    );
}
