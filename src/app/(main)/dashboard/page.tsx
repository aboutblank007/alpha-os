"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from "@/components/ui/Button";
import { MarketWatch } from "@/components/dashboard/MarketWatch";
import { TradingViewChart } from "@/components/charts/TradingViewChart";
import { GlassCard, CardHeader, CardTitle } from "@/components/ui/GlassCard";
import { Maximize2, MoreHorizontal, Plus, Activity, Zap, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";
import { useBridgeStatus } from "@/hooks/useBridgeStatus";

// 量子 HFT 组件
import { LatencyDisplay } from "@/components/dashboard/LatencyDisplay";
import { SystemVitals } from "@/components/dashboard/SystemVitals";
import { RiskGauge } from "@/components/dashboard/RiskGauge";
import { PanicButton } from "@/components/dashboard/PanicButton";
import { useQuantumSocket } from "@/hooks/useQuantumSocket";

interface AiSignal {
    action: string;
    confidence: number;
    symbol: string;
    price: number;
    timestamp?: number;
}

export default function DashboardPage() {
    const [selectedSymbol, setSelectedSymbol] = useState("BTCUSD");
    const [aiSignal, setAiSignal] = useState<AiSignal | null>(null);
    const [showQuantumPanel, setShowQuantumPanel] = useState(true);

    // 接入 Bridge 数据
    const { status, isConnected } = useBridgeStatus();
    const { account, positions } = status.last_mt5_update;

    // 量子 WebSocket 连接
    const { closeAllPositions, ticksPerSecond } = useQuantumSocket();

    // Fetch AI Signal
    useEffect(() => {
        const fetchAi = async () => {
            try {
                const res = await fetch('/api/ai/latest');
                if (res.ok) {
                    const data = await res.json();
                    setAiSignal(data);
                }
            } catch (e) {
                console.error(e);
            }
        };
        fetchAi();
        const interval = setInterval(fetchAi, 5000);
        return () => clearInterval(interval);
    }, []);

    // 计算浮动盈亏 (从持仓累加)
    const floatingPnl = positions.reduce((sum, p) => sum + p.pnl, 0);

    // 紧急平仓处理
    const handleCloseAll = useCallback(async () => {
        return closeAllPositions();
    }, [closeAllPositions]);

    return (
        <div className="h-full w-full flex flex-col gap-3 overflow-hidden">

            {/* Top Stats Row + System Status */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 min-h-[80px] shrink-0">
                {/* 系统状态卡片 */}
                <GlassCard className="flex flex-col justify-center px-4 relative overflow-hidden group col-span-1">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Cpu size={40} className="text-primary" />
                    </div>
                    <span className="text-xs text-text-muted uppercase tracking-wider font-semibold">系统状态</span>
                    <div className="flex items-center gap-3 mt-1">
                        <LatencyDisplay showLabel={false} />
                        <span className="text-xs text-text-muted font-mono">{ticksPerSecond} TPS</span>
                    </div>
                </GlassCard>

                {/* 原有统计卡片 */}
                {[
                    {
                        label: "总权益",
                        value: account ? `$${account.equity.toFixed(2)}` : "---",
                        change: account ? `Bal: $${account.balance.toFixed(2)}` : "---",
                        isPositive: true
                    },
                    {
                        label: "持仓盈亏",
                        value: `${floatingPnl >= 0 ? '+' : ''}$${floatingPnl.toFixed(2)}`,
                        change: isConnected ? "实时" : "断开",
                        isPositive: floatingPnl >= 0
                    },
                    {
                        label: "持仓数量",
                        value: positions.length.toString(),
                        change: positions.length > 0 ? "活跃" : "空仓",
                        isPositive: positions.length > 0
                    },
                    {
                        label: "AI 状态",
                        value: aiSignal?.action === 'BUY' ? '买入' : aiSignal?.action === 'SELL' ? '卖出' : '待机',
                        change: `${(aiSignal?.confidence || 0).toFixed(2)} 置信度`,
                        isPositive: true
                    },
                ].map((stat, i) => (
                    <GlassCard key={i} className="flex flex-col justify-center px-6 relative overflow-hidden group">
                        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                            <div className="w-16 h-16 bg-gradient-to-br from-primary to-transparent rounded-full blur-2xl" />
                        </div>
                        <span className="text-xs text-text-muted uppercase tracking-wider font-semibold">{stat.label}</span>
                        <div className="flex items-end gap-2 mt-1">
                            <span className={cn("text-2xl font-bold tracking-tight", stat.isPositive ? "text-text-primary" : "text-text-primary")}>{stat.value}</span>
                            <span className={cn("text-xs mb-1.5 font-medium", stat.label === "AI 状态" ? "text-primary" : stat.isPositive ? "text-success" : "text-danger")}>{stat.change}</span>
                        </div>
                    </GlassCard>
                ))}
            </div>

            {/* Main Workspace (Grid) */}
            <div className="flex-1 grid grid-cols-12 gap-3 min-h-0 overflow-hidden">

                {/* Left: Market Watch */}
                <div className="col-span-12 md:col-span-3 lg:col-span-2 h-full overflow-hidden">
                    <MarketWatch onSymbolSelect={setSelectedSymbol} />
                </div>

                {/* Center: Chart */}
                <div className={cn(
                    "col-span-12 h-full flex flex-col gap-3 overflow-hidden",
                    showQuantumPanel ? "md:col-span-6 lg:col-span-7" : "md:col-span-9 lg:col-span-10"
                )}>
                    <GlassCard className="flex-1 relative group overflow-hidden">
                        {/* Chart Header Overlay */}
                        <div className="absolute top-4 left-4 z-10 flex items-center gap-4 bg-bg-card/50 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/5">
                            <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded bg-warning text-black flex items-center justify-center font-bold text-xs">{selectedSymbol.substring(0, 1)}</div>
                                <span className="font-bold text-sm tracking-wide">{selectedSymbol}</span>
                            </div>
                            <div className="h-4 w-px bg-white/10" />
                            <div className="flex gap-2">
                                <span className="text-xs text-text-secondary cursor-pointer hover:text-primary transition-colors">15m</span>
                                <span className="text-xs text-primary font-bold cursor-pointer">1H</span>
                                <span className="text-xs text-text-secondary cursor-pointer hover:text-primary transition-colors">4H</span>
                            </div>
                        </div>

                        <div className="absolute top-4 right-4 z-10 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 bg-bg-card/50 backdrop-blur"
                                onClick={() => setShowQuantumPanel(!showQuantumPanel)}
                            >
                                <Activity size={14} />
                            </Button>
                            <Button variant="ghost" size="icon" className="h-8 w-8 bg-bg-card/50 backdrop-blur"><Maximize2 size={14} /></Button>
                            <Button variant="ghost" size="icon" className="h-8 w-8 bg-bg-card/50 backdrop-blur"><MoreHorizontal size={14} /></Button>
                        </div>

                        <div className="w-full h-full">
                            <TradingViewChart symbol={selectedSymbol} />
                        </div>
                    </GlassCard>
                </div>

                {/* Right: Quantum Control Panel */}
                {showQuantumPanel && (
                    <div className="col-span-12 md:col-span-3 lg:col-span-3 h-full flex flex-col gap-3 overflow-hidden">

                        {/* System Vitals */}
                        <GlassCard className="p-4 overflow-auto">
                            <div className="flex items-center gap-2 mb-3">
                                <Activity size={16} className="text-primary" />
                                <span className="text-xs font-bold text-text-muted uppercase">系统监控</span>
                            </div>
                            <SystemVitals />
                        </GlassCard>

                        {/* AI Insight Card */}
                        <GlassCard className="p-5 relative overflow-hidden flex flex-col justify-center">
                            <div className="absolute top-0 right-0 p-6 opacity-5">
                                <Zap size={80} />
                            </div>
                            <div className="flex items-center gap-2 mb-2">
                                <Activity size={16} className="text-primary" />
                                <span className="text-xs font-bold text-text-muted uppercase">神经网络引擎</span>
                            </div>

                            {aiSignal ? (
                                <div className="z-10">
                                    <div className="flex items-baseline gap-3">
                                        <span className={cn("text-3xl font-bold tracking-tight",
                                            aiSignal.action === 'BUY' ? "text-success" :
                                                aiSignal.action === 'SELL' ? "text-danger" : "text-text-muted"
                                        )}>
                                            {aiSignal.action === 'BUY' ? '买入' : aiSignal.action === 'SELL' ? '卖出' : aiSignal.action}
                                        </span>
                                        <span className="text-sm font-mono text-text-secondary">{aiSignal.symbol}</span>
                                    </div>
                                    <div className="mt-2 flex items-center gap-2">
                                        <div className="h-1.5 flex-1 bg-bg-subtle rounded-full overflow-hidden">
                                            <div className="h-full bg-primary" style={{ width: `${aiSignal.confidence * 100}%` }} />
                                        </div>
                                        <span className="text-xs font-mono font-bold text-primary">{(aiSignal.confidence * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="mt-2 text-[10px] text-text-secondary font-mono">
                                        最后更新: {new Date(aiSignal.timestamp || Date.now()).toLocaleTimeString()}
                                    </div>
                                </div>
                            ) : (
                                <div className="flex items-center gap-2 text-text-muted">
                                    <Activity className="animate-pulse" size={16} /> 连接中...
                                </div>
                            )}
                        </GlassCard>

                        {/* Risk Gauge */}
                        <GlassCard className="p-4">
                            <div className="flex items-center gap-2 mb-2">
                                <Activity size={16} className="text-primary" />
                                <span className="text-xs font-bold text-text-muted uppercase">风险仪表</span>
                            </div>
                            <RiskGauge
                                currentLeverage={positions.length > 0 ? 2.5 : 0}
                                winProbability={aiSignal?.confidence || 0.5}
                                odds={1.5}
                            />
                        </GlassCard>

                        {/* Panic Button */}
                        <div className="mt-auto">
                            <PanicButton
                                onCloseAll={handleCloseAll}
                                disabled={positions.length === 0}
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Bottom Panel (Tabs for Positions) */}
            <GlassCard className="min-h-[180px] max-h-[240px] shrink-0 flex flex-col overflow-hidden">
                <div className="flex items-center gap-6 px-4 border-b border-white/5 h-12">
                    {['持仓', '挂单', '历史', '日志'].map((tab, i) => (
                        <button key={tab} className={`text-xs font-medium h-full border-b-2 transition-colors px-2 ${i === 0 ? 'border-primary text-white shadow-[0_4px_12px_-4px_var(--color-primary)]' : 'border-transparent text-text-muted hover:text-text-secondary'}`}>
                            {tab}
                        </button>
                    ))}
                    <div className="flex-1" />
                    <Button variant="ghost" size="sm" className="h-7 text-xs"><Plus size={12} className="mr-1" /> 添加标签</Button>
                </div>
                <div className="flex-1 overflow-auto">
                    {positions.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-text-muted text-sm">
                            无持仓
                        </div>
                    ) : (
                        <div className="w-full text-left border-collapse">
                            <div className="sticky top-0 bg-bg-card z-10 grid grid-cols-7 px-4 py-2 text-xs font-bold text-text-muted border-b border-white/5 uppercase">
                                <div>Ticket</div>
                                <div>Symbol</div>
                                <div>Type</div>
                                <div>Volume</div>
                                <div>Price</div>
                                <div>Profit</div>
                                <div>Swap</div>
                            </div>
                            {positions.map((p) => (
                                <div key={p.ticket} className="grid grid-cols-7 px-4 py-2 text-xs border-b border-white/5 hover:bg-white/5 transition-colors">
                                    <div className="font-mono text-text-secondary">{p.ticket}</div>
                                    <div className="font-bold text-text-primary">{p.symbol}</div>
                                    <div className={cn("font-bold", p.type === 'BUY' ? "text-long" : "text-short")}>{p.type}</div>
                                    <div className="font-mono">{p.volume}</div>
                                    <div className="font-mono">{p.open_price}</div>
                                    <div className={cn("font-mono font-bold", p.pnl >= 0 ? "text-success" : "text-danger")}>{p.pnl.toFixed(2)}</div>
                                    <div className="font-mono text-text-secondary">{p.swap.toFixed(2)}</div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </GlassCard>
        </div>
    );
}
