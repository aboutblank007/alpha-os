"use client";

import { useEffect, useMemo, useRef, useState } from 'react';
import { TrendingUp, DollarSign, Target, Activity, Calendar, ArrowRight, Share2, FileDown } from 'lucide-react';
import { StatCard } from '@/components/Card';
import { EquityCurve } from '@/components/EquityCurve';
import { RecentTrades } from '@/components/RecentTrades';
import { SymbolPerformance } from '@/components/SymbolPerformance';
import { OngoingOrders } from '@/components/OngoingOrders';
import { TradingInsights } from '@/components/TradingInsights';
import { SentimentAnalysis } from '@/components/SentimentAnalysis';
import { RiskAlerts } from '@/components/RiskAlerts';
import { supabase, type Trade } from '@/lib/supabase';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { TradingViewChart } from '@/components/charts/TradingViewChart';
import { MarketWatch } from '@/components/MarketWatch';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors, DragOverlay, defaultDropAnimationSideEffects, DragStartEvent, DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, rectSortingStrategy } from '@dnd-kit/sortable';
import { SortableItem } from '@/components/dashboard/SortableItem';
import { useTradeStore } from '@/store/useTradeStore';
import { useMarketStore } from '@/store/useMarketStore';
import { ErrorBoundary } from '@/components/ErrorBoundary'; // Import ErrorBoundary

interface DashboardStats {
    totalPnL: number;
    winRate: number;
    totalTrades: number;
    profitFactor: number;
    maxDrawdown?: number;
    executionEfficiency?: number;
}

type Period = '1W' | '1M' | '3M' | 'YTD' | 'ALL';
type ConfirmKind = 'export' | 'share' | 'reset' | null;
interface TagItem { id: string; date: string; label: string }

export default function DashboardPage() {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [stats, setStats] = useState<DashboardStats>({
        totalPnL: 0,
        winRate: 0,
        totalTrades: 0,
        profitFactor: 0,
    });
    const [equityData, setEquityData] = useState<Array<{ date: string; equity: number }>>([]);
    const [loading, setLoading] = useState(true);
    const [period, setPeriod] = useState<Period>('1M');
    const [overlays, setOverlays] = useState<{ ema: boolean; bb: boolean }>({ ema: false, bb: false });
    const [tags, setTags] = useState<TagItem[]>([]);
    const [confirm, setConfirm] = useState<ConfirmKind>(null);
    const [workspace, setWorkspace] = useState<string>('默认工作区');

    // Default layout with new components
    const [layout, setLayout] = useState<string[]>(() => {
        const saved = typeof window !== 'undefined' ? localStorage.getItem('alphaos_layout_v4') : null;
        return saved ? JSON.parse(saved) : ['market', 'marketWatch', 'chart', 'symbols', 'orders', 'insights', 'sentiment', 'alerts', 'recent'];
    });

    const [activeId, setActiveId] = useState<string | null>(null);
    const [selectedSymbol, setSelectedSymbol] = useState<string>('EUR_USD'); // State for selected chart symbol

    const account = useTradeStore(state => state.account);
    const isConnected = useMarketStore(state => state.isConnected);

    const handleTrade = async (symbol: string, side: 'BUY' | 'SELL') => {
        try {
            const res = await fetch('/api/bridge/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: side,
                    symbol: symbol.replace('/', '').replace('_', ''), // Normalize symbol
                    volume: 0.01 // Fixed volume for MVP
                })
            });

            const data = await res.json();
            if (!res.ok || data.error) {
                throw new Error(data.error || 'Trade failed');
            }
            console.log('Trade executed:', data);
            // Optional: Refresh trades or show toast
            fetchTrades();
        } catch (e) {
            console.error('Trade execution error:', e);
            alert(`Trade Failed: ${e instanceof Error ? e.message : 'Unknown error'}`);
        }
    };

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                distance: 8, // 8px movement required to start drag, prevents accidental drags on clicks
            },
        }),
        useSensor(KeyboardSensor, {
            coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    const logsRef = useRef<Array<{ time: number; action: string }>>([]);

    // Derived State
    const openTrades = useMemo(() => trades.filter(t => t.status === 'open'), [trades]);
    const closedTrades = useMemo(() => trades.filter(t => t.status === 'closed'), [trades]);

    useEffect(() => {
        fetchTrades();

        const channel = supabase
            .channel('realtime trades')
            .on('postgres_changes', { event: '*', schema: 'public', table: 'trades' }, (_payload) => {
                console.log('Supabase Realtime Update:', _payload);
                fetchTrades();
            })
            .subscribe();

        return () => {
            supabase.removeChannel(channel);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (typeof window === 'undefined') return;
        const onKey = (e: KeyboardEvent) => {
            if (e.metaKey && e.key.toLowerCase() === 'e') setConfirm('export');
            if (e.metaKey && e.key.toLowerCase() === 's') setConfirm('share');
            if (e.metaKey && e.key.toLowerCase() === 'r') setConfirm('reset');
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, []);

    async function fetchTrades() {
        try {
            const { data, error } = await supabase
                .from('trades')
                .select('*')
                .order('created_at', { ascending: false });

            if (error) throw error;

            if (data) {
                setTrades(data);
                const closed = data.filter(t => t.status === 'closed');
                calculateStats(closed, data.length); // Pass total count (active + closed)

                const sortedTrades = [...closed].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
                generateEquityCurve(sortedTrades);
            }
        } catch (error: unknown) {
            const err = error as Record<string, unknown>;
            const errorMsg = `
╔═══════════════════════════════════════════════════════════════╗
║ ❌ Supabase 查询失败 ║
╚═══════════════════════════════════════════════════════════════╝
错误详情:
• message: ${err?.message || '未知错误'}
• code: ${err?.code || '无代码'}
• details: ${err?.details || '无详情'}
• hint: ${err?.hint || '无提示'}

完整错误对象: ${JSON.stringify(error, null, 2)}

🔍 可能的原因:
1. Supabase 项目已暂停（免费计划会自动暂停）
2. trades 表不存在
3. RLS 策略配置错误
4. API 密钥已过期

📝 解决方法:
1. 访问 http://localhost:3000/debug 查看详细诊断
2. 登录 https://app.supabase.com 检查项目状态
3. 如项目暂停，点击 Resume 恢复

🔗 快速诊断: http://localhost:3000/debug
            `;
            console.error(errorMsg);
        } finally {
            setLoading(false);
        }
    }

    function calculateStats(closedTrades: Trade[], totalCount: number) {
        const wins = closedTrades.filter(t => t.pnl_net > 0).length;
        const totalPnL = closedTrades.reduce((sum, t) => sum + t.pnl_net, 0);
        const totalWins = closedTrades.filter(t => t.pnl_net > 0).reduce((sum, t) => sum + t.pnl_net, 0);
        const totalLosses = Math.abs(closedTrades.filter(t => t.pnl_net < 0).reduce((sum, t) => sum + t.pnl_net, 0));

        // Compute Max Drawdown from equity curve built from closed trades
        let peak = 10000;
        let equity = 10000;
        let mdd = 0;
        closedTrades
            .slice()
            .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
            .forEach(t => {
                equity += t.pnl_net;
                if (equity > peak) peak = equity;
                const dd = peak > 0 ? (peak - equity) / peak : 0;
                if (dd > mdd) mdd = dd;
            });

        // Simple execution efficiency proxy: combine PF and WinRate into 0-100 scale
        const pf = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? 2 : 0;
        const efficiency = Math.min(100, Math.round(((wins / (closedTrades.length || 1)) * 0.6 + Math.min(pf / 2, 1) * 0.4) * 100));

        setStats({
            totalPnL,
            winRate: closedTrades.length > 0 ? (wins / closedTrades.length) * 100 : 0,
            totalTrades: totalCount,
            profitFactor: totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? 999 : 0,
            maxDrawdown: mdd * 100,
            executionEfficiency: efficiency,
        });
    }

    function generateEquityCurve(trades: Trade[]) {
        let cumulative = 10000; // Initial balance
        const curve = trades.map(trade => {
            cumulative += trade.pnl_net;
            return {
                date: new Date(trade.created_at).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }),
                equity: cumulative,
            };
        });
        setEquityData(curve);
    }

    // Risk alerts logic moved to component, but we need to trigger audio here if needed or move audio to component?
    // The AudioContext logic was in useEffect dependent on alerts.
    // For now, let's keep the sound effect logic simple or move it. 
    // Since we want to keep logic unchanged, I'll calculate alerts just for the sound effect.
    const activeAlerts = useMemo(() => {
        const list = [];
        if (stats.winRate < 45) list.push('warning');
        if (stats.profitFactor < 1.2 && stats.profitFactor > 0) list.push('warning');
        if (stats.totalPnL < -500) list.push('danger');
        return list;
    }, [stats]);

    useEffect(() => {
        if (activeAlerts.includes('danger')) {
            try {
                const AC = (window as unknown as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext }).AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
                if (!AC) return;
                const ctx = new AC();
                const o = ctx.createOscillator();
                const g = ctx.createGain();
                o.type = 'sine';
                o.frequency.value = 880;
                g.gain.value = 0.02;
                o.connect(g);
                g.connect(ctx.destination);
                o.start();
                setTimeout(() => o.stop(), 300);
            } catch { }
        }
    }, [activeAlerts]);

    function toggleOverlay(k: 'ema' | 'bb') {
        setOverlays(prev => ({ ...prev, [k]: !prev[k] }));
    }

    function addTag(date: string, label: string) {
        const newTag = { id: Date.now().toString(), date, label };
        setTags(prev => [...prev, newTag]);
        logsRef.current.push({ time: Date.now(), action: `add-tag:${date}:${label}` });
        localStorage.setItem('alphaos_logs', JSON.stringify(logsRef.current));
    }

    function exportCSV() {
        const headers = ['id', 'created_at', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl_net', 'status'];
        const rows = trades.map(t => [t.id, t.created_at, t.symbol, t.side, t.entry_price, t.exit_price ?? '', t.quantity, t.pnl_net, t.status]);
        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'trades.csv'; a.click();
        URL.revokeObjectURL(url);
        logsRef.current.push({ time: Date.now(), action: 'export-csv' });
        localStorage.setItem('alphaos_logs', JSON.stringify(logsRef.current));
    }

    async function share() {
        const data = { title: 'AlphaOS Dashboard', text: 'My trading dashboard snapshot', url: window.location.href };
        const nav = navigator as Navigator & { share?: (p: { title: string; text: string; url: string }) => Promise<void> };
        if (nav.share) {
            try { await nav.share(data); } catch { }
        } else {
            await navigator.clipboard.writeText(data.url);
        }
        logsRef.current.push({ time: Date.now(), action: 'share' });
        localStorage.setItem('alphaos_logs', JSON.stringify(logsRef.current));
    }

    function resetLayout() {
        const defaultLayout = ['market', 'chart', 'symbols', 'orders', 'insights', 'sentiment', 'alerts', 'recent'];
        setLayout(defaultLayout);
        localStorage.setItem('alphaos_layout_v3', JSON.stringify(defaultLayout));
        logsRef.current.push({ time: Date.now(), action: 'reset-layout' });
        localStorage.setItem('alphaos_logs', JSON.stringify(logsRef.current));
    }

    function handleDragStart(event: DragStartEvent) {
        setActiveId(event.active.id as string);
    }

    function handleDragEnd(event: DragEndEvent) {
        const { active, over } = event;

        if (over && active.id !== over.id) {
            setLayout((items) => {
                const oldIndex = items.indexOf(active.id as string);
                const newIndex = items.indexOf(over.id as string);
                const newLayout = arrayMove(items, oldIndex, newIndex);
                localStorage.setItem('alphaos_layout_v4', JSON.stringify(newLayout));
                return newLayout;
            });
        }

        setActiveId(null);
    }

    const dropAnimation = {
        sideEffects: defaultDropAnimationSideEffects({
            styles: {
                active: {
                    opacity: '0.5',
                },
            },
        }),
    };

    const renderWidget = (key: string, isOverlay = false) => {
        // 根据组件类型定义响应式列跨度
        const getColSpan = (componentKey: string) => {
            if (isOverlay) return 'col-span-1'; // Overlay always fits content or fixed width? Actually overlay mimics the item.
            // For overlay, we might want to force a specific width or let it inherit. 
            // But grid classes won't work well in overlay portal unless we copy the grid context or set explicit dimensions.
            // For simplicity, the overlay will just render the content.

            switch (componentKey) {
                case 'market':
                    return 'col-span-1 md:col-span-2 lg:col-span-3';
                case 'marketWatch':
                    return 'col-span-1 md:col-span-1 lg:col-span-1'; // Fits next to chart
                case 'chart':
                    return 'col-span-1 md:col-span-2 lg:col-span-3'; // Equity Curve
                case 'orders':
                case 'recent':
                    return 'col-span-1 md:col-span-2 lg:col-span-4';
                default:
                    return 'col-span-1';
            }
        };

        const getHeight = (componentKey: string) => {
            switch (componentKey) {
                case 'market':
                case 'marketWatch':
                    return 'h-[400px] md:h-[500px]';
                case 'chart':
                case 'symbols':
                    return 'h-[350px] md:h-[500px]';
                case 'orders':
                case 'recent':
                    return 'h-[350px] md:h-[400px]';
                case 'insights':
                case 'sentiment':
                case 'alerts':
                    return 'h-[250px] md:h-[300px]';
                default:
                    return 'h-[250px] md:h-[300px]';
            }
        };

        const colSpanClass = getColSpan(key);
        const heightClass = getHeight(key);
        const baseClass = `bg-transparent ${colSpanClass} ${heightClass}`;
        // If overlay, we might want to remove col-span and just give it a fixed width or similar to the source.
        // But for now let's keep it simple.

        const content = (() => {
            switch (key) {
                case 'market':
                    return <TradingViewChart className="h-full" height={500} initialSymbol={selectedSymbol} key={selectedSymbol} />; // Add key to force re-render on symbol change
                case 'chart':
                    return (
                        <EquityCurve
                            data={equityData}
                            period={period}
                            overlays={overlays}
                            tags={tags}
                            onPeriodChange={setPeriod}
                            onToggleOverlay={toggleOverlay}
                            onAddTag={() => addTag(equityData.slice(-1)[0]?.date ?? '', '策略标记')}
                        />
                    );
                case 'symbols':
                    return <SymbolPerformance trades={closedTrades} />;
                case 'orders':
                    return <OngoingOrders />;
                case 'recent':
                    return <RecentTrades trades={closedTrades} />;
                case 'insights':
                    return <TradingInsights trades={trades} />;
                case 'sentiment':
                    return <SentimentAnalysis trades={trades} />;
                case 'marketWatch':
                    return <MarketWatch isConnected={isConnected} onTrade={handleTrade} onSymbolSelect={setSelectedSymbol} />;
                case 'alerts':
                    return <RiskAlerts stats={stats} onResetLayout={() => setConfirm('reset')} />;
                default:
                    return null;
            }
        })();

        return (
            <div className={`${isOverlay ? 'h-full w-full' : baseClass} ${isOverlay ? 'shadow-2xl scale-105 cursor-grabbing' : ''}`}>
                {/* Wrap content in ErrorBoundary */}
                <ErrorBoundary>
                {content}
                </ErrorBoundary>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="max-w-screen-3xl mx-auto space-y-8 animate-pulse">
                <div className="h-48 rounded-3xl bg-white/5"></div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    {[1, 2, 3, 4].map(i => (
                        <div key={i} className="h-32 rounded-2xl bg-white/5"></div>
                    ))}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[500px]">
                    <div className="lg:col-span-3 rounded-2xl bg-white/5"></div>
                    <div className="rounded-2xl bg-white/5"></div>
                </div>
            </div>
        );
    }

    return (
        <div className="max-w-screen-3xl mx-auto pb-12 md:pb-20 space-y-4 md:space-y-8">
            {/* Premium Welcome Banner */}
            <div className="relative overflow-hidden rounded-2xl md:rounded-3xl glass-panel-strong p-6 md:p-12 animate-fade-in-up border border-white/10 shadow-2xl">
                {/* Dynamic Background Mesh - Hidden on mobile for performance */}
                <div className="hidden md:block absolute top-0 right-0 w-[800px] h-[800px] bg-accent-primary/20 rounded-full blur-[120px] -translate-y-1/2 translate-x-1/3 pointer-events-none mix-blend-screen"></div>
                <div className="hidden md:block absolute bottom-0 left-0 w-[600px] h-[600px] bg-accent-secondary/20 rounded-full blur-[100px] translate-y-1/3 -translate-x-1/3 pointer-events-none mix-blend-screen"></div>
                <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10 md:opacity-20 pointer-events-none"></div>

                <div className="relative z-10 flex flex-col gap-6 md:gap-8">
                    <div className="space-y-4 md:space-y-6">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-accent-primary backdrop-blur-md shadow-sm">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-primary opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-accent-primary"></span>
                            </span>
                            {workspace}
                        </div>

                        <div className="space-y-2">
                            <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold text-white tracking-tight-custom text-balance drop-shadow-lg">
                                欢迎回来，<span className="text-gradient">交易员</span>
                            </h1>
                            <p className="text-sm md:text-lg text-slate-400 max-w-xl leading-relaxed font-light">
                                您的投资组合今天表现良好。您有 <span className="text-white font-medium border-b border-accent-success/50">{openTrades.length} 个活跃信号</span>，胜率正在上升。
                            </p>
                        </div>
                    </div>

                    <div className="flex gap-2 md:gap-3 flex-wrap">
                        <button className="group flex items-center gap-2 px-4 md:px-5 py-2 md:py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 transition-all text-xs md:text-sm font-medium text-slate-300 hover:text-white backdrop-blur-sm">
                            <Calendar size={16} className="text-slate-400 group-hover:text-white transition-colors md:w-[18px] md:h-[18px]" />
                            <span>本月</span>
                        </button>
                        <Button variant="primary" rightIcon={<ArrowRight size={16} />} className="btn-premium shadow-lg shadow-accent-primary/20 text-xs md:text-sm px-4 md:px-5 py-2 md:py-3">新建分析</Button>
                        <Button variant="secondary" leftIcon={<Share2 size={16} />} onClick={() => setConfirm('share')} className="backdrop-blur-sm text-xs md:text-sm px-4 md:px-5 py-2 md:py-3">分享</Button>
                        <Button variant="secondary" leftIcon={<FileDown size={16} />} onClick={() => setConfirm('export')} className="backdrop-blur-sm text-xs md:text-sm px-4 md:px-5 py-2 md:py-3">导出</Button>

                        <div className="hidden md:flex gap-1 bg-black/40 p-1 rounded-xl border border-white/5 backdrop-blur-md">
                            {['默认工作区', '分析', '策略'].map(w => (
                                <button
                                    key={w}
                                    onClick={() => setWorkspace(w)}
                                    className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${workspace === w ? 'bg-white/10 text-white shadow-sm' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                                >
                                    {w}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Bento Grid Layout with dnd-kit */}
            <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragStart={handleDragStart}
                onDragEnd={handleDragEnd}
            >
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 auto-rows-min">
                    {/* Stats - Fixed */}
                    <div className="col-span-1 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
                        <StatCard
                            label="净资产"
                            value={`$${(account?.equity ?? 0).toFixed(2)}`}
                            trend={stats.totalPnL > 0 ? 8.0 : -5.0}
                            icon={<DollarSign size={24} />}
                            subValue={`余额: $${(account?.balance ?? 0).toFixed(2)}`}
                        />
                    </div>
                    <div className="col-span-1 animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
                        <StatCard
                            label="胜率"
                            value={`${stats.winRate.toFixed(1)}%`}
                            trend={2.5}
                            icon={<Target size={24} />}
                            subValue="最近 20 笔交易"
                        />
                    </div>
                    <div className="col-span-1 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
                        <StatCard
                            label="最大回撤"
                            value={`${(stats.maxDrawdown ?? 0).toFixed(1)}%`}
                            trend={Number((-(stats.maxDrawdown ?? 0)).toFixed(2))}
                            icon={<Activity size={24} />}
                            subValue="基于已平仓交易"
                        />
                    </div>
                    <div className="col-span-1 animate-fade-in-up" style={{ animationDelay: '0.25s' }}>
                        <StatCard
                            label="执行效率"
                            value={`${(stats.executionEfficiency ?? 0).toFixed(0)}%`}
                            icon={<TrendingUp size={24} />}
                            subValue="综合评分"
                        />
                    </div>

                    <SortableContext items={layout} strategy={rectSortingStrategy}>
                        {layout.map((key) => (
                            <SortableItem key={key} id={key} className={renderWidget(key).props.className}>
                                {renderWidget(key).props.children}
                            </SortableItem>
                        ))}
                    </SortableContext>
                </div>

                <DragOverlay dropAnimation={dropAnimation}>
                    {activeId ? renderWidget(activeId, true) : null}
                </DragOverlay>
            </DndContext>

            {/* Modal */}
            <Modal
                open={confirm !== null}
                onOpenChange={(o) => setConfirm(o ? confirm : null)}
                title={confirm === 'export' ? '导出数据' : confirm === 'share' ? '分享仪表板' : '重置布局'}
                footer={
                    <>
                        <Button variant="secondary" onClick={() => setConfirm(null)}>取消</Button>
                        {confirm === 'export' && <Button onClick={() => { exportCSV(); setConfirm(null); }}>导出 CSV</Button>}
                        {confirm === 'share' && <Button onClick={() => { share(); setConfirm(null); }}>复制链接</Button>}
                        {confirm === 'reset' && <Button onClick={() => { resetLayout(); setConfirm(null); }}>重置</Button>}
                    </>
                }
            >
                {confirm === 'export' && <p className="text-slate-300">将最近的交易导出为 CSV 文件。</p>}
                {confirm === 'share' && <p className="text-slate-300">分享您的仪表板快照链接。</p>}
                {confirm === 'reset' && <p className="text-slate-300">恢复默认的仪表板布局。</p>}
            </Modal>
        </div>
    );
}
