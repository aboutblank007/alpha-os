"use client";

import { Trade } from '@/lib/supabase';
// Tooltip removed as we use custom hover
import { useMemo } from 'react';

interface SentimentHeatmapProps {
    trades: Trade[];
}

export function SentimentHeatmap({ trades }: SentimentHeatmapProps) {
    // Generate last 30 days
    const days = useMemo(() => {
        const result = [];
        const today = new Date();
        for (let i = 29; i >= 0; i--) {
            const d = new Date(today);
            d.setDate(d.getDate() - i);
            result.push(d.toISOString().split('T')[0]); // YYYY-MM-DD
        }
        return result;
    }, []);

    // Group trades by day and calculate average emotion score
    const dailyData = useMemo(() => {
        const map = new Map<string, { score: number; count: number; pnl: number }>();
        
        trades.forEach(t => {
            const date = t.created_at.split('T')[0];
            if (!map.has(date)) {
                map.set(date, { score: 0, count: 0, pnl: 0 });
            }
            const entry = map.get(date)!;
            // Mock emotion score if missing (1-10)
            const score = t.emotion_score ?? (t.pnl_net > 0 ? 8 : 3); 
            entry.score += score;
            entry.pnl += t.pnl_net;
            entry.count += 1;
        });

        return map;
    }, [trades]);

    const getIntensityClass = (score: number) => {
        if (score >= 9) return 'bg-accent-success'; // Very Happy
        if (score >= 7) return 'bg-accent-success/70'; // Happy
        if (score >= 5) return 'bg-slate-500'; // Neutral
        if (score >= 3) return 'bg-accent-danger/70'; // Sad/Frustrated
        return 'bg-accent-danger'; // Angry/Tilt
    };

    return (
        <div className="glass-panel p-6 rounded-xl">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">情绪热力图</h3>
                <div className="flex gap-2 text-xs text-slate-500">
                    <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-accent-danger"></div>焦虑</div>
                    <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-slate-500"></div>平静</div>
                    <div className="flex items-center gap-1"><div className="w-3 h-3 rounded bg-accent-success"></div>自信</div>
                </div>
            </div>

            <div className="grid grid-cols-10 md:grid-cols-15 lg:grid-cols-30 gap-2">
                {days.map(day => {
                    const data = dailyData.get(day);
                    const avgScore = data ? data.score / data.count : null;
                    const dateObj = new Date(day);
                    
                    return (
                        <div key={day} className="flex flex-col gap-1 group relative">
                            <div 
                                className={`w-full pt-[100%] rounded-md transition-all hover:scale-110 cursor-pointer ${
                                    avgScore !== null ? getIntensityClass(avgScore) : 'bg-white/5'
                                }`}
                                title={`${day}: ${avgScore ? `Score ${avgScore.toFixed(1)}` : 'No trades'}`}
                            ></div>
                            {/* Custom Tooltip on Hover could go here */}
                            <div className="opacity-0 group-hover:opacity-100 absolute bottom-full mb-2 left-1/2 -translate-x-1/2 whitespace-nowrap bg-black/90 text-xs px-2 py-1 rounded text-slate-200 pointer-events-none z-10 border border-white/10">
                                {dateObj.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })}
                                {data && (
                                    <div className="font-mono mt-0.5">
                                        Score: {avgScore?.toFixed(1)}<br/>
                                        PnL: ${data.pnl.toFixed(0)}
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
            <div className="mt-4 text-xs text-slate-400 text-center">
                最近 30 天的情绪与交易表现关联分析
            </div>
        </div>
    );
}

