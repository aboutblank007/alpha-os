"use client";

import { useState, useMemo } from 'react';
import { Trade } from '@/lib/supabase';
import { PerformanceMetrics } from './PerformanceMetrics';
import { EquityChart } from './EquityChart';
import { SymbolBreakdown } from './SymbolBreakdown';
import { ExecutionQuality } from './ExecutionQuality';
import { Filter, Calendar, Layers } from 'lucide-react';

interface AnalyticsPanelProps {
    trades: Trade[];
}

type TimeRange = '1W' | '1M' | '3M' | 'YTD' | 'ALL';

export function AnalyticsPanel({ trades }: AnalyticsPanelProps) {
    const [timeRange, setTimeRange] = useState<TimeRange>('1M');
    const [selectedSymbol, setSelectedSymbol] = useState<string>('ALL');

    // -- 1. Filter Logic --
    const filteredTrades = useMemo(() => {
        const now = new Date();
        let cutoff = new Date(0); // Default ALL

        switch (timeRange) {
            case '1W':
                cutoff = new Date(now.setDate(now.getDate() - 7));
                break;
            case '1M':
                cutoff = new Date(now.setMonth(now.getMonth() - 1));
                break;
            case '3M':
                cutoff = new Date(now.setMonth(now.getMonth() - 3));
                break;
            case 'YTD':
                cutoff = new Date(new Date().getFullYear(), 0, 1);
                break;
        }

        return trades.filter(t => {
            const tradeDate = new Date(t.created_at);
            const matchesTime = timeRange === 'ALL' || tradeDate >= cutoff;
            const matchesSymbol = selectedSymbol === 'ALL' || t.symbol === selectedSymbol;
            return matchesTime && matchesSymbol && t.status === 'closed';
        });
    }, [trades, timeRange, selectedSymbol]);

    // Unique symbols for filter
    const symbols = useMemo(() => {
        const unique = new Set(trades.map(t => t.symbol));
        return ['ALL', ...Array.from(unique)];
    }, [trades]);

    if (!trades || trades.length === 0) {
        return (
            <div className="glass-panel p-6 flex flex-col items-center justify-center h-full text-slate-400">
                <Layers size={48} className="mb-4 opacity-50" />
                <p>暂无交易数据，开始交易以查看分析。</p>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col gap-4 overflow-y-auto pr-2">
            {/* Header / Filters */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-4 glass-panel rounded-xl">
                <div className="flex items-center gap-2">
                    <div className="p-2 bg-accent-primary/20 rounded-lg text-accent-primary">
                        <Filter size={20} />
                    </div>
                    <h2 className="text-lg font-bold text-white">交易分析看板</h2>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                    {/* Symbol Select */}
                    <div className="relative">
                        <select
                            value={selectedSymbol}
                            onChange={(e) => setSelectedSymbol(e.target.value)}
                            className="appearance-none bg-slate-800 border border-white/10 text-xs text-white rounded-md px-3 py-1.5 focus:outline-none focus:border-accent-primary pr-8"
                        >
                            {symbols.map(s => (
                                <option key={s} value={s}>{s === 'ALL' ? '所有品种' : s}</option>
                            ))}
                        </select>
                        {/* Custom arrow could go here */}
                    </div>

                    {/* Time Range Pills */}
                    <div className="flex bg-slate-800 p-1 rounded-md border border-white/5">
                        {(['1W', '1M', '3M', 'YTD', 'ALL'] as TimeRange[]).map((range) => (
                            <button
                                key={range}
                                onClick={() => setTimeRange(range)}
                                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${timeRange === range
                                        ? 'bg-accent-primary text-white shadow-sm'
                                        : 'text-slate-400 hover:text-white'
                                    }`}
                            >
                                {range}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Top Row: KPI Cards */}
            <PerformanceMetrics trades={filteredTrades} />

            {/* Main Charts Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-[400px]">
                {/* Equity Curve (2/3 width) */}
                <div className="lg:col-span-2 glass-panel rounded-xl p-4 min-h-[350px]">
                    <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                        <Calendar size={16} /> 资金曲线 (Equity Curve)
                    </h3>
                    <EquityChart trades={filteredTrades} />
                </div>

                {/* Symbol Breakdown (1/3 width) */}
                <div className="lg:col-span-1 glass-panel rounded-xl p-4 min-h-[350px]">
                    <SymbolBreakdown trades={filteredTrades} />
                </div>
            </div>

            {/* Bottom Row: Execution Quality */}
            <div className="glass-panel rounded-xl p-4 min-h-[300px]">
                <ExecutionQuality trades={filteredTrades} />
            </div>
        </div>
    );
}
