"use client";

import { useMemo } from 'react';
import { useSignalStore } from '@/store/useSignalStore';
import { Bot, Activity, Brain, Zap, TrendingUp, TrendingDown, Minus, ShieldAlert } from 'lucide-react';

export function AiMarketMonitor() {
    const { signals } = useSignalStore();

    // 1. Calculate System Status (Pulse)
    // If last signal is recent (< 5 mins), system is "Active". Otherwise "Standby".
    const systemStatus = useMemo(() => {
        if (signals.length === 0) return 'standby';
        const lastSignalTime = new Date(signals[0].created_at).getTime();
        const diffMinutes = (Date.now() - lastSignalTime) / 1000 / 60;
        return diffMinutes < 5 ? 'active' : 'standby';
    }, [signals]);

    // 2. Calculate Market Mood (Bullish/Bearish Ratio of recent signals)
    const marketMood = useMemo(() => {
        if (signals.length === 0) return { label: 'Neutral', score: 50, color: 'text-slate-400' };

        // Analyze last 20 signals
        const recent = signals.slice(0, 20);
        const buys = recent.filter(s => s.action === 'BUY').length;
        const sells = recent.filter(s => s.action === 'SELL').length;
        const total = buys + sells;

        if (total === 0) return { label: 'Neutral', score: 50, color: 'text-slate-400' };

        const buyRatio = (buys / total) * 100;

        if (buyRatio > 60) return { label: 'Bullish', score: buyRatio, color: 'text-accent-success' };
        if (buyRatio < 40) return { label: 'Bearish', score: buyRatio, color: 'text-accent-danger' };
        return { label: 'Mixed', score: buyRatio, color: 'text-accent-warning' };
    }, [signals]);

    // 3. Parse Latest Insight
    const latestInsight = useMemo(() => {
        if (signals.length === 0) return null;
        const s = signals[0];

        // Parse comment for AI Metadata
        // Format: "Comment | Original: BUY | Auto: Executed (AI: 0.85)"
        const aiMatch = s.comment?.match(/AI:\s*(\d+\.?\d*)/);
        const confidence = aiMatch ? parseFloat(aiMatch[1]) : 0;
        const isSkipped = s.comment?.includes('Skipped') || s.comment?.includes('failed');
        const isExecuted = s.comment?.includes('Executed') || s.status === 'processed';

        return {
            symbol: s.symbol,
            action: s.action,
            confidence,
            status: isSkipped ? 'Skipped' : isExecuted ? 'Executed' : 'Pending',
            reason: isSkipped ? 'Risk Controls' : 'High Confidence', // Simplified inference
            time: new Date(s.created_at).toLocaleTimeString()
        };
    }, [signals]);

    return (
        <div className="glass-panel rounded-xl p-4 h-full flex flex-col relative overflow-hidden ring-1 ring-white/10">
            {/* Background Decor */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-accent-primary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>

            {/* Header */}
            <div className="flex items-center justify-between mb-4 z-10">
                <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${systemStatus === 'active' ? 'bg-accent-primary/20 text-accent-primary' : 'bg-slate-800 text-slate-500'}`}>
                        <Bot size={18} />
                    </div>
                    <div>
                        <h3 className="text-base font-bold text-white leading-none">AI Monitor</h3>
                        <div className="flex items-center gap-1.5 mt-1">
                            <span className={`relative flex h-2 w-2`}>
                                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${systemStatus === 'active' ? 'bg-accent-success' : 'bg-slate-500'}`}></span>
                                <span className={`relative inline-flex rounded-full h-2 w-2 ${systemStatus === 'active' ? 'bg-accent-success' : 'bg-slate-500'}`}></span>
                            </span>
                            <span className="text-[10px] uppercase tracking-wider font-medium text-slate-400">
                                {systemStatus === 'active' ? 'System Online' : 'Standby'}
                            </span>
                        </div>
                    </div>
                </div>
                {/* CPU/Latency Mockup */}
                <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500 bg-black/20 px-2 py-1 rounded-md border border-white/5">
                    <div className="flex items-center gap-1">
                        <Activity size={10} />
                        <span>24ms</span>
                    </div>
                    <div className="w-px h-3 bg-white/10"></div>
                    <div className="flex items-center gap-1">
                        <Brain size={10} />
                        <span>v2.4</span>
                    </div>
                </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-2 gap-3 flex-1 min-h-0 z-10">

                {/* Left: Latest Decision */}
                <div className="col-span-1 flex flex-col gap-2 p-3 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                    <div className="flex items-center justify-between">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Latest Decision</span>
                        {latestInsight && (
                            <span className="text-[10px] text-slate-500">{latestInsight.time}</span>
                        )}
                    </div>

                    {latestInsight ? (
                        <div className="flex-1 flex flex-col justify-between">
                            <div>
                                <div className="flex items-baseline gap-1.5">
                                    <span className="text-sm font-bold text-white">{latestInsight.symbol}</span>
                                    <span className={`text-xs font-bold ${latestInsight.action === 'BUY' ? 'text-accent-success' : 'text-accent-danger'}`}>
                                        {latestInsight.action}
                                    </span>
                                </div>
                                <div className="flex items-center gap-1 mt-1">
                                    <div className={`text-[10px] px-1.5 py-0.5 rounded border flex items-center gap-1 w-fit ${latestInsight.status === 'Executed'
                                            ? 'bg-accent-success/10 border-accent-success/20 text-accent-success'
                                            : latestInsight.status === 'Skipped'
                                                ? 'bg-accent-danger/10 border-accent-danger/20 text-accent-danger'
                                                : 'bg-slate-700/50 border-slate-600 text-slate-300'
                                        }`}>
                                        {latestInsight.status === 'Executed' && <Zap size={10} />}
                                        {latestInsight.status === 'Skipped' && <ShieldAlert size={10} />}
                                        {latestInsight.status}
                                    </div>
                                </div>
                            </div>

                            <div className="mt-2 pt-2 border-t border-white/5">
                                <div className="flex justify-between items-center mb-1">
                                    <span className="text-[10px] text-slate-400">Confidence</span>
                                    <span className="text-[10px] font-mono text-accent-primary">{(latestInsight.confidence * 100).toFixed(0)}%</span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-accent-primary to-purple-500 rounded-full transition-all duration-500"
                                        style={{ width: `${latestInsight.confidence * 100}%` }}
                                    ></div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex-1 flex items-center justify-center text-xs text-slate-500 italic">
                            No recent signals
                        </div>
                    )}
                </div>

                {/* Right: Market Mood Gauge */}
                <div className="col-span-1 flex flex-col gap-2 p-3 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                    <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Market Mood</span>

                    <div className="flex-1 flex flex-col items-center justify-center gap-2">
                        {/* Gauge Visual */}
                        <div className="relative w-16 h-8 overflow-hidden">
                            <div className="absolute top-0 left-0 w-16 h-16 rounded-full border-4 border-slate-700 box-border"></div>
                            <div
                                className={`absolute top-0 left-0 w-16 h-16 rounded-full border-4 border-t-transparent border-r-transparent border-l-transparent box-border transition-transform duration-700 ease-out custom-gauge-rotate origin-center`}
                                style={{
                                    borderColor: marketMood.color === 'text-accent-success' ? '#10b981' : marketMood.color === 'text-accent-danger' ? '#ef4444' : '#f59e0b',
                                    transform: `rotate(${(marketMood.score / 100) * 180 - 45}deg)` // Simplified rotation logic for visual
                                    // Actually better to just use a semi-circle SVG or similar, but let's use text for robustness first
                                }}
                            ></div>
                            {/* Simple alternative: Icon */}
                        </div>

                        <div className={`text-2xl font-bold ${marketMood.color}`}>
                            {marketMood.label === 'Bullish' && <TrendingUp size={28} />}
                            {marketMood.label === 'Bearish' && <TrendingDown size={28} />}
                            {marketMood.label === 'Neutral' && <Minus size={28} />}
                            {marketMood.label === 'Mixed' && <Activity size={28} />}
                        </div>

                        <div className="text-center">
                            <div className={`text-xs font-bold ${marketMood.color}`}>{marketMood.label}</div>
                            <div className="text-[10px] text-slate-500 mt-0.5">Sentiment Score: {marketMood.score.toFixed(0)}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
