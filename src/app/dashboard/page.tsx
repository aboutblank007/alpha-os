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
interface TagItem { date: string; label: string }

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
    const [overlays, setOverlays] = useState<{ ema: boolean; bb: boolean }>({ ema: true, bb: false });
    const [tags, setTags] = useState<TagItem[]>([]);
    const [confirm, setConfirm] = useState<ConfirmKind>(null);
    const [workspace, setWorkspace] = useState<string>('默认工作区');
    
    // Default layout with new components
    const [layout, setLayout] = useState<string[]>(() => {
        const saved = typeof window !== 'undefined' ? localStorage.getItem('alphaos_layout_v2') : null;
        return saved ? JSON.parse(saved) : ['chart', 'symbols', 'orders', 'insights', 'sentiment', 'alerts', 'recent'];
    });
    
    const logsRef = useRef<Array<{ time: number; action: string }>>([]);

    // Derived State
    const openTrades = useMemo(() => trades.filter(t => t.status === 'open'), [trades]);
    const closedTrades = useMemo(() => trades.filter(t => t.status === 'closed'), [trades]);

    useEffect(() => {
        fetchTrades();

        const channel = supabase
            .channel('realtime trades')
            .on('postgres_changes', { event: '*', schema: 'public', table: 'trades' }, (_payload) => {
                fetchTrades();
            })
            .subscribe();

        return () => {
            supabase.removeChannel(channel);
        };
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
        } catch (error: any) {
            const errorMsg = `
╔═══════════════════════════════════════════════════════════════╗
║ ❌ Supabase 查询失败 ║
╚═══════════════════════════════════════════════════════════════╝
错误详情:
• message: ${error?.message || '未知错误'}
• code: ${error?.code || '无代码'}
• details: ${error?.details || '无详情'}
• hint: ${error?.hint || '无提示'}

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
            } catch {}
        }
    }, [activeAlerts]);

    function toggleOverlay(k: 'ema' | 'bb') {
        setOverlays(prev => ({ ...prev, [k]: !prev[k] }));
    }

    function addTag(date: string, label: string) {
        setTags(prev => [...prev, { date, label }]);
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
            try { await nav.share(data); } catch {}
        } else {
            await navigator.clipboard.writeText(data.url);
        }
        logsRef.current.push({ time: Date.now(), action: 'share' });
        localStorage.setItem('alphaos_logs', JSON.stringify(logsRef.current));
    }

    function resetLayout() {
        const defaultLayout = ['chart', 'symbols', 'orders', 'insights', 'sentiment', 'alerts', 'recent'];
        setLayout(defaultLayout);
        localStorage.setItem('alphaos_layout_v2', JSON.stringify(defaultLayout));
        logsRef.current.push({ time: Date.now(), action: 'reset-layout' });
        localStorage.setItem('alphaos_logs', JSON.stringify(logsRef.current));
    }

    function onDragStart(e: React.DragEvent<HTMLDivElement>, key: string) {
        e.dataTransfer.setData('text/plain', key);
    }
    function onDrop(e: React.DragEvent<HTMLDivElement>, target: string) {
        const src = e.dataTransfer.getData('text/plain');
        if (!src || src === target) return;
        const order = [...layout];
        const si = order.indexOf(src), ti = order.indexOf(target);
        if (si === -1 || ti === -1) return;
        order.splice(si, 1);
        order.splice(ti, 0, src);
        setLayout(order);
        localStorage.setItem('alphaos_layout_v2', JSON.stringify(order));
    }
    function allowDrop(e: React.DragEvent<HTMLDivElement>) { e.preventDefault(); }

    if (loading) {
        return (
            <div className="max-w-[1600px] mx-auto space-y-8 animate-pulse">
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
        <div className="max-w-[1600px] mx-auto pb-20 space-y-8">
            {/* Premium Welcome Banner */}
            <div className="relative overflow-hidden rounded-3xl glass-panel-strong p-8 md:p-10 animate-fade-in-up">
                <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-accent-primary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3 pointer-events-none"></div>
                <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-accent-secondary/10 rounded-full blur-3xl translate-y-1/3 -translate-x-1/3 pointer-events-none"></div>

                <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div className="space-y-4 max-w-2xl">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-accent-primary">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-primary opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-accent-primary"></span>
                            </span>
                            {workspace}
                        </div>
                        <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight-custom text-balance">
                            欢迎回来，<span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">交易员</span>
                        </h1>
                        <p className="text-lg text-slate-400 max-w-xl leading-relaxed">
                            您的投资组合今天表现良好。您有 <span className="text-white font-medium">{openTrades.length} 个活跃信号</span>，胜率正在上升。
                        </p>
                    </div>

                    <div className="flex gap-3 flex-wrap">
                        <button className="group flex items-center gap-2 px-5 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 transition-all text-sm font-medium text-slate-300 hover:text-white">
                            <Calendar size={18} className="text-slate-400 group-hover:text-white transition-colors" />
                            <span>本月</span>
                        </button>
                        <Button variant="primary" rightIcon={<ArrowRight size={18} />}>新建分析</Button>
                        <Button variant="secondary" leftIcon={<Share2 size={18} />} onClick={() => setConfirm('share')}>分享</Button>
                        <Button variant="secondary" leftIcon={<FileDown size={18} />} onClick={() => setConfirm('export')}>导出</Button>
                        <div className="flex gap-1 bg-white/5 p-1 rounded-lg">
                            {['默认工作区', '分析', '策略'].map(w => (
                                <button key={w} onClick={() => setWorkspace(w)} className={`px-3 py-1 text-xs rounded ${workspace === w ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>{w}</button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Bento Grid Layout with drag & drop */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 auto-rows-min">
                {/* Stats - Fixed */}
                <div className="col-span-1 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
                    <StatCard
                        label="净盈亏"
                        value={`$${stats.totalPnL.toFixed(2)}`}
                        trend={stats.totalPnL > 0 ? 8.0 : -5.0}
                        icon={<DollarSign size={24} />}
                        subValue="今日 +$1,240"
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
                        trend={-(stats.maxDrawdown ?? 0)}
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

                {layout.map((key, index) => {
                    const getCommonProps = (colSpanClass: string, heightClass: string) => ({
                        draggable: true,
                        onDragStart: (e: React.DragEvent<HTMLDivElement>) => onDragStart(e, key),
                        onDragOver: allowDrop,
                        onDrop: (e: React.DragEvent<HTMLDivElement>) => onDrop(e, key),
                        className: `animate-fade-in-up bg-transparent ${colSpanClass} ${heightClass}`
                    });

                    // 根据组件类型定义响应式列跨度
                    const getColSpan = (componentKey: string) => {
                        switch (componentKey) {
                            case 'chart':
                                // 图表组件：移动端全宽，中等屏幕全宽，大屏幕占3列
                                return 'col-span-1 md:col-span-2 lg:col-span-3';
                            case 'orders':
                            case 'recent':
                                // 订单和交易记录：移动端全宽，中等屏幕全宽，大屏幕占4列（全宽）
                                return 'col-span-1 md:col-span-2 lg:col-span-4';
                            case 'symbols':
                            case 'insights':
                            case 'sentiment':
                            case 'alerts':
                                // 其他组件：移动端全宽，中等屏幕1列，大屏幕1列
                                return 'col-span-1';
                            default:
                                return 'col-span-1';
                        }
                    };

                    const getHeight = (componentKey: string) => {
                        switch (componentKey) {
                            case 'chart':
                            case 'symbols':
                                return 'h-[500px]';
                            case 'orders':
                            case 'recent':
                                return 'h-[400px]';
                            case 'insights':
                            case 'sentiment':
                            case 'alerts':
                                return 'h-[300px]';
                            default:
                                return 'h-[300px]';
                        }
                    };

                    const colSpanClass = getColSpan(key);
                    const heightClass = getHeight(key);

                    switch (key) {
                        case 'chart':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <EquityCurve data={equityData} period={period} overlays={overlays} tags={tags} onPeriodChange={setPeriod} onToggleOverlay={toggleOverlay} />
                                    <div className="mt-3 flex items-center gap-2">
                                        <Button variant="outline" onClick={() => addTag(equityData.slice(-1)[0]?.date ?? '', '策略标记')}>添加策略标记</Button>
                                    </div>
                                </div>
                            );
                        case 'symbols':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <SymbolPerformance trades={closedTrades} />
                                </div>
                            );
                        case 'orders':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <OngoingOrders orders={openTrades} />
                                </div>
                            );
                        case 'recent':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <RecentTrades trades={closedTrades} />
                                </div>
                            );
                         case 'insights':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <TradingInsights trades={trades} />
                                </div>
                            );
                        case 'sentiment':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <SentimentAnalysis trades={trades} />
                                </div>
                            );
                        case 'alerts':
                            return (
                                <div key={key} {...getCommonProps(colSpanClass, heightClass)}>
                                    <RiskAlerts stats={stats} onResetLayout={() => setConfirm('reset')} />
                                </div>
                            );
                        default:
                            return null;
                    }
                })}
            </div>

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
