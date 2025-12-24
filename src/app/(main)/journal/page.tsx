"use client";

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { GlassCard, CardHeader, CardTitle, CardContent } from "@/components/ui/GlassCard";
import { Button } from "@/components/ui/Button";
import { ChevronLeft, ChevronRight, Calendar as CalendarIcon, Loader2, RefreshCw, TrendingUp, TrendingDown, Play, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBridgeStatus } from '@/hooks/useBridgeStatus';
import { OrderBookHeatmap } from "@/components/charts/OrderBookHeatmap";

interface Trade {
    id: string;
    symbol: string;
    type: string;
    profit: number;
    volume: number;
    entryPrice: number;
    exitPrice: number | null;
    openTime: string;
    closeTime: string | null;
    swap: number;
    commission: number;
    status?: string;
}

interface DayStats {
    date: string;
    count: number;
    pnl: number;
    wins: number;
    losses: number;
    trades: Trade[];
}

export default function JournalPage() {
    const [currentDate, setCurrentDate] = useState(new Date());
    const [trades, setTrades] = useState<Trade[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedDay, setSelectedDay] = useState<DayStats | null>(null);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [replayingTrade, setReplayingTrade] = useState<Trade | null>(null);
    
    // 实时持仓数据
    const { status, isConnected } = useBridgeStatus();
    const { positions } = status.last_mt5_update;
    
    // 计算今日浮动盈亏
    const todayFloatingPnl = positions.reduce((sum, p) => sum + p.pnl, 0);

    // 获取交易数据
    const fetchTrades = useCallback(async () => {
            const startOfMonth = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
            const endOfMonth = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0, 23, 59, 59);
            const queryStart = new Date(startOfMonth);
        queryStart.setDate(queryStart.getDate() - 7);

            try {
            const res = await fetch(`/api/analytics/trades?limit=1000&status=closed`);
            if (!res.ok) throw new Error('Failed to fetch');
            const data = await res.json();
            
            // 过滤当月数据
            const monthTrades = (data.trades || []).filter((t: Trade) => {
                const tradeDate = new Date(t.closeTime || t.openTime);
                return tradeDate >= queryStart && tradeDate <= endOfMonth;
            });
            
            setTrades(monthTrades);
            } catch (err) {
                console.error("Failed to fetch journal trades", err);
            } finally {
                setLoading(false);
            }
    }, [currentDate]);

    useEffect(() => {
        setLoading(true);
        fetchTrades();
    }, [fetchTrades]);
    
    // 自动刷新
    useEffect(() => {
        if (!autoRefresh) return;
        const interval = setInterval(fetchTrades, 10000); // 10秒刷新
        return () => clearInterval(interval);
    }, [autoRefresh, fetchTrades]);

    // 聚合数据
    const daysData = useMemo(() => {
        const map = new Map<string, DayStats>();
        trades.forEach(t => {
            const d = new Date(t.closeTime || t.openTime);
            const key = d.toLocaleDateString('en-CA');
            if (!map.has(key)) map.set(key, { date: key, count: 0, pnl: 0, wins: 0, losses: 0, trades: [] });
            const stats = map.get(key)!;
            stats.count++;
            stats.pnl += t.profit || 0;
            if ((t.profit || 0) > 0) stats.wins++;
            else if ((t.profit || 0) < 0) stats.losses++;
            stats.trades.push(t);
        });
        return map;
    }, [trades]);

    // 日历网格
    const calendarDays = useMemo(() => {
        const year = currentDate.getFullYear();
        const month = currentDate.getMonth();
        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
        const days = [];
        const startPadding = firstDay.getDay();

        for (let i = startPadding - 1; i >= 0; i--) {
            days.push({ date: new Date(year, month, -i), isCurrentMonth: false });
        }
        for (let i = 1; i <= lastDay.getDate(); i++) {
            days.push({ date: new Date(year, month, i), isCurrentMonth: true });
        }
        const remaining = 42 - days.length;
        for (let i = 1; i <= remaining; i++) {
            days.push({ date: new Date(year, month + 1, i), isCurrentMonth: false });
        }
        return days;
    }, [currentDate]);

    const changeMonth = (delta: number) => {
        const d = new Date(currentDate);
        d.setMonth(d.getMonth() + delta);
        setCurrentDate(d);
        setSelectedDay(null);
    };

    // 本月盈亏
    const monthPnL = useMemo(() => {
        let sum = 0;
        daysData.forEach((stats, key) => {
            const d = new Date(key);
            if (d.getMonth() === currentDate.getMonth() && d.getFullYear() === currentDate.getFullYear()) {
                sum += stats.pnl;
            }
        });
        return sum;
    }, [daysData, currentDate]);
    
    // 今日已平仓盈亏
    const todayKey = new Date().toLocaleDateString('en-CA');
    const todayStats = daysData.get(todayKey);
    const todayClosedPnl = todayStats?.pnl || 0;
    const todayTotalPnl = todayClosedPnl + todayFloatingPnl;
    
    // 合并今日实时数据到 daysData（用于日历显示）
    const enrichedDaysData = useMemo(() => {
        const enriched = new Map(daysData);
        const today = new Date().toLocaleDateString('en-CA');
        const existingToday = enriched.get(today) || { date: today, count: 0, pnl: 0, wins: 0, losses: 0, trades: [] };
        
        // 如果有浮动盈亏，添加到今日数据
        if (todayFloatingPnl !== 0 || positions.length > 0) {
            enriched.set(today, {
                ...existingToday,
                // 今日总盈亏 = 已平仓 + 浮动
                pnl: existingToday.pnl + todayFloatingPnl,
                // 持仓数量加入显示
                count: existingToday.count + positions.length,
            });
        }
        return enriched;
    }, [daysData, todayFloatingPnl, positions.length]);

    return (
        <div className="flex flex-col lg:flex-row h-full gap-4 overflow-hidden">
            {/* Calendar Area */}
            <div className="flex-1 flex flex-col gap-4 min-w-0">

                {/* Header */}
                <div className="flex items-center justify-between shrink-0">
                    <div>
                        <h1 className="text-xl font-bold tracking-tight flex items-center gap-3">
                            {currentDate.toLocaleString('zh-CN', { month: 'long', year: 'numeric' })}
                            <span className={cn("text-sm font-mono px-2 py-0.5 rounded bg-bg-card border border-white/5", monthPnL >= 0 ? "text-success" : "text-danger")}>
                                {monthPnL >= 0 ? "+" : ""}${monthPnL.toFixed(2)}
                            </span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button variant="ghost" size="icon" onClick={() => changeMonth(-1)}><ChevronLeft size={18} /></Button>
                        <Button variant="secondary" size="sm" onClick={() => setCurrentDate(new Date())}>今天</Button>
                        <Button variant="ghost" size="icon" onClick={() => changeMonth(1)}><ChevronRight size={18} /></Button>
                        <Button variant="ghost" size="sm" onClick={fetchTrades} disabled={loading}>
                            <RefreshCw size={14} className={cn("mr-1", loading && "animate-spin")} />
                        </Button>
                    </div>
                </div>

                {/* 今日实时盈亏卡片 */}
                <div className="grid grid-cols-3 gap-3 shrink-0">
                    <GlassCard className="p-3">
                        <div className="text-[10px] text-text-muted uppercase">今日已平仓</div>
                        <div className={cn("text-lg font-bold font-mono", todayClosedPnl >= 0 ? "text-success" : "text-danger")}>
                            {todayClosedPnl >= 0 ? "+" : ""}${todayClosedPnl.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-text-muted">{todayStats?.count || 0} 笔交易</div>
                    </GlassCard>
                    <GlassCard className="p-3">
                        <div className="text-[10px] text-text-muted uppercase flex items-center gap-1">
                            浮动盈亏
                            {isConnected && <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />}
                        </div>
                        <div className={cn("text-lg font-bold font-mono", todayFloatingPnl >= 0 ? "text-success" : "text-danger")}>
                            {todayFloatingPnl >= 0 ? "+" : ""}${todayFloatingPnl.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-text-muted">{positions.length} 持仓中</div>
                    </GlassCard>
                    <GlassCard className={cn("p-3", todayTotalPnl >= 0 ? "border-success/20" : "border-danger/20")}>
                        <div className="text-[10px] text-text-muted uppercase">今日总计</div>
                        <div className={cn("text-lg font-bold font-mono", todayTotalPnl >= 0 ? "text-success" : "text-danger")}>
                            {todayTotalPnl >= 0 ? "+" : ""}${todayTotalPnl.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-text-muted">实时更新</div>
                    </GlassCard>
                </div>

                {/* 日历网格 */}
                <GlassCard className="flex-1 p-4 flex flex-col overflow-hidden">
                    <div className="grid grid-cols-7 mb-2 border-b border-white/5 pb-2 shrink-0">
                        {['日', '一', '二', '三', '四', '五', '六'].map(d => (
                            <div key={d} className="text-center text-[10px] font-bold text-text-muted">周{d}</div>
                        ))}
                    </div>

                    {loading ? (
                        <div className="flex-1 flex items-center justify-center"><Loader2 className="animate-spin text-primary" size={32} /></div>
                    ) : (
                        <div className="flex-1 grid grid-cols-7 grid-rows-6 gap-1">
                            {calendarDays.map((cell, i) => {
                                const dateKey = cell.date.toLocaleDateString('en-CA');
                                const stats = enrichedDaysData.get(dateKey);
                                const isToday = cell.date.toDateString() === new Date().toDateString();
                                const isSelected = selectedDay?.date === dateKey;
                                const hasOpenPositions = isToday && positions.length > 0;

                                return (
                                    <div
                                        key={i}
                                        onClick={() => {
                                            if (stats) setSelectedDay(stats);
                                            else if (hasOpenPositions) {
                                                // 为今日创建一个虚拟的 stats 对象以显示持仓信息
                                                setSelectedDay({
                                                    date: dateKey,
                                                    count: positions.length,
                                                    pnl: todayFloatingPnl,
                                                    wins: 0,
                                                    losses: 0,
                                                    trades: positions.map(p => ({
                                                        id: String(p.ticket),
                                                        symbol: p.symbol,
                                                        type: p.type,
                                                        profit: p.pnl,
                                                        volume: p.volume,
                                                        entryPrice: p.open_price,
                                                        exitPrice: null,
                                                        openTime: new Date().toISOString(),
                                                        closeTime: null,
                                                        swap: p.swap,
                                                        commission: 0,
                                                        status: 'open'
                                                    }))
                                                });
                                            }
                                        }}
                                        className={cn(
                                            "rounded-lg border p-1.5 flex flex-col justify-between cursor-pointer transition-all hover:border-border-active",
                                            cell.isCurrentMonth ? "bg-white/[0.02]" : "bg-transparent opacity-20 pointer-events-none",
                                            isToday ? "border-primary shadow-[0_0_8px_rgba(37,99,235,0.2)]" : "border-transparent bg-bg-subtle",
                                            isSelected ? "ring-2 ring-primary z-10" : "",
                                            stats?.pnl! > 0 ? "bg-success/5" : stats?.pnl! < 0 ? "bg-danger/5" : ""
                                        )}
                                    >
                                        <div className="flex justify-between items-center">
                                            <span className={cn("text-[10px] font-medium", isToday ? "text-primary" : "text-text-muted")}>{cell.date.getDate()}</span>
                                            {stats && (
                                                <span className={cn("text-[8px] px-1 rounded", hasOpenPositions ? "bg-primary/20 text-primary" : "bg-bg-card")}>
                                                    {stats.count}{hasOpenPositions && " 活跃"}
                                                </span>
                                            )}
                                            {!stats && hasOpenPositions && (
                                                <span className="text-[8px] px-1 rounded bg-primary/20 text-primary animate-pulse">
                                                    {positions.length} 活跃
                                                </span>
                                            )}
                                        </div>
                                        {stats && (
                                            <div className={cn("text-[10px] font-mono font-bold text-right", stats.pnl >= 0 ? "text-success" : "text-danger")}>
                                                {stats.pnl > 0 ? "+" : ""}{stats.pnl.toFixed(0)}
                                            </div>
                                        )}
                                        {!stats && hasOpenPositions && (
                                            <div className={cn("text-[10px] font-mono font-bold text-right animate-pulse", todayFloatingPnl >= 0 ? "text-success" : "text-danger")}>
                                                {todayFloatingPnl > 0 ? "+" : ""}{todayFloatingPnl.toFixed(0)}
                                            </div>
                                        )}
                                    </div>
                                )
                            })}
                        </div>
                    )}
                </GlassCard>
            </div>

            {/* 侧边栏详情 */}
            <GlassCard className={cn(
                "w-[300px] flex flex-col transition-all duration-300 shrink-0 overflow-hidden",
                selectedDay ? "opacity-100" : "opacity-50 pointer-events-none hidden lg:flex"
            )}>
                {selectedDay ? (
                    <>
                        <CardHeader className="shrink-0 pb-2">
                            <CardTitle className="text-sm">{new Date(selectedDay.date).toLocaleDateString('zh-CN', { weekday: 'long', month: 'short', day: 'numeric' })}</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 overflow-y-auto space-y-3 p-4 pt-0">
                            <div className="grid grid-cols-2 gap-2">
                                <div className="p-2 bg-bg-base/50 rounded-lg">
                                    <div className="text-[10px] text-text-muted uppercase">净盈亏</div>
                                    <div className={cn("text-lg font-bold font-mono", selectedDay.pnl >= 0 ? "text-success" : "text-danger")}>
                                        ${selectedDay.pnl.toFixed(2)}
                                    </div>
                                </div>
                                <div className="p-2 bg-bg-base/50 rounded-lg">
                                    <div className="text-[10px] text-text-muted uppercase">胜率</div>
                                    <div className="text-lg font-bold font-mono text-text-primary">
                                        {selectedDay.count > 0 ? Math.round((selectedDay.wins / selectedDay.count) * 100) : 0}%
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-2 text-xs">
                                <span className="text-success">盈 {selectedDay.wins}</span>
                                <span className="text-text-muted">/</span>
                                <span className="text-danger">亏 {selectedDay.losses}</span>
                                <span className="text-text-muted">/</span>
                                <span>共 {selectedDay.count}</span>
                            </div>

                            <div className="space-y-1.5 pt-2 border-t border-white/5">
                                <h4 className="text-[10px] font-bold text-text-muted uppercase">
                                    {selectedDay.trades.some(t => t.status === 'open') ? '活跃持仓' : '交易记录'}
                                </h4>
                                {selectedDay.trades.length === 0 ? (
                                    <div className="text-xs text-text-muted text-center py-4">暂无交易记录</div>
                                ) : (
                                    selectedDay.trades.map(t => (
                                        <div key={t.id} className={cn(
                                            "p-2 rounded border flex justify-between items-center group",
                                            t.status === 'open' ? "bg-primary/5 border-primary/20" : "bg-white/5 border-white/5"
                                        )}>
                                        <div>
                                                <div className="font-bold text-xs text-text-primary flex items-center gap-1">
                                                    {t.symbol} 
                                                    <span className={cn("text-[9px]", t.type === 'BUY' ? "text-success" : "text-danger")}>
                                                        {t.type === 'BUY' ? '买' : '卖'}
                                                    </span>
                                                    {t.status === 'open' && (
                                                        <span className="text-[8px] px-1 bg-primary/20 text-primary rounded animate-pulse">活跃</span>
                                                    )}
                                                </div>
                                                <div className="text-[9px] text-text-muted font-mono">
                                                    {t.volume} lots · {new Date(t.openTime).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className={cn("font-mono text-sm font-bold", (t.profit || 0) >= 0 ? "text-success" : "text-danger")}>
                                                    {(t.profit || 0) >= 0 ? "+" : ""}${(t.profit || 0).toFixed(2)}
                                                </div>
                                                <button 
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setReplayingTrade(t);
                                                    }}
                                                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all text-primary"
                                                    title="Forensic Replay"
                                                >
                                                    <Play size={12} fill="currentColor" />
                                                </button>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </CardContent>
                    </>
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-text-muted p-6 text-center">
                        <CalendarIcon size={28} className="mb-3 opacity-20" />
                        <p className="text-xs">选择日期查看详情</p>
                    </div>
                )}
            </GlassCard>

            {/* Forensic Replay Modal */}
            {replayingTrade && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
                    <GlassCard className="w-full max-w-4xl h-[80vh] flex flex-col relative overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-white/10 bg-black/40">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center text-primary">
                                    <Play size={14} fill="currentColor" />
                                </div>
                                <div>
                                    <h2 className="text-lg font-bold flex items-center gap-2">
                                        交易回放 (Forensic Replay)
                                        <span className="text-xs px-2 py-0.5 rounded bg-white/10 font-mono text-text-secondary">#{replayingTrade.id}</span>
                                    </h2>
                                    <div className="text-xs text-text-muted flex items-center gap-2">
                                        <span>{replayingTrade.symbol}</span>
                                        <span>·</span>
                                        <span className={replayingTrade.type === 'BUY' ? "text-success" : "text-danger"}>{replayingTrade.type}</span>
                                        <span>·</span>
                                        <span>{new Date(replayingTrade.openTime).toLocaleString('zh-CN')}</span>
                                    </div>
                                </div>
                            </div>
                            <Button variant="ghost" size="icon" onClick={() => setReplayingTrade(null)}>
                                <X size={20} />
                            </Button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-4 grid grid-cols-12 gap-4">
                            {/* Left: Snapshot */}
                            <div className="col-span-8 flex flex-col gap-4">
                                <div className="bg-black rounded-lg border border-white/10 p-1 relative">
                                    <div className="absolute top-2 left-2 z-10 px-2 py-1 bg-black/60 backdrop-blur rounded text-[10px] text-text-muted border border-white/5">
                                        Execution Snapshot: T-0ms
                                    </div>
                                    <OrderBookHeatmap symbol={replayingTrade.symbol} height={400} />
                                </div>
                            </div>

                            {/* Right: AI Rationale */}
                            <div className="col-span-4 flex flex-col gap-4">
                                <div className="p-4 rounded-lg bg-white/5 border border-white/5 space-y-3">
                                    <h3 className="text-sm font-bold text-primary flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                                        AI 决策归因
                                    </h3>
                                    <div className="space-y-2 text-xs text-text-secondary">
                                        <div className="flex justify-between">
                                            <span>Model Confidence</span>
                                            <span className="text-white font-mono">87.4%</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span>Gradient Norm</span>
                                            <span className="text-success font-mono">1.2e-4 (Healthy)</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span>Wick Ratio</span>
                                            <span className="text-white font-mono">0.42</span>
                                        </div>
                                    </div>
                                    <div className="pt-2 border-t border-white/5">
                                        <p className="text-xs italic text-text-muted">
                                            "Market microstructure indicates strong buy wall support at {replayingTrade.entryPrice}. Volatility shock detected in previous 500ms window."
                                        </p>
                                    </div>
                                </div>

                                <div className="p-4 rounded-lg bg-white/5 border border-white/5 flex-1">
                                    <h3 className="text-sm font-bold text-text-muted mb-2">执行滑点分析</h3>
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-xs">
                                            <span className="text-text-muted">Expected Price</span>
                                            <span className="font-mono">{replayingTrade.entryPrice}</span>
                                        </div>
                                        <div className="flex justify-between text-xs">
                                            <span className="text-text-muted">Executed Price</span>
                                            <span className="font-mono">{replayingTrade.entryPrice}</span>
                                        </div>
                                        <div className="flex justify-between text-xs">
                                            <span className="text-text-muted">Slippage</span>
                                            <span className="font-mono text-success">0.00 pts</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </GlassCard>
                </div>
            )}
        </div>
    );
}
