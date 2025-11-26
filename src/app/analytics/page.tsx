"use client";

import { useState, useEffect } from 'react';
// createClient unused
// Actually, let's use the helper if possible, but page.tsx is client side, so createClientComponentClient is better if using auth, 
// but we are using a simple global client in src/lib/supabase.ts for this project structure.
import { supabase, Trade } from '@/lib/supabase';
import { MaeMfeScatterChart } from '@/components/charts/MaeMfeScatterChart';
import { SentimentHeatmap } from '@/components/SentimentHeatmap';
import { StrategyBreakdown } from '@/components/StrategyBreakdown';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function AnalyticsPage() {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchTrades = async () => {
            try {
                const { data, error } = await supabase
                    .from('trades')
                    .select('*')
                    .eq('status', 'closed')
                    .order('created_at', { ascending: false });
                
                if (error) throw error;
                if (data) setTrades(data);
            } catch (e) {
                console.error('Error fetching trades:', e);
            } finally {
                setLoading(false);
            }
        };

        fetchTrades();
    }, []);

    if (loading) {
        return <div className="flex items-center justify-center h-screen text-slate-500">Loading Analytics...</div>;
    }

    return (
        <div className="max-w-screen-3xl mx-auto pb-20 space-y-8 p-6 md:p-12">
            {/* Header */}
            <div className="flex items-center gap-4">
                <Link href="/dashboard" className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
                    <ArrowLeft size={20} />
                </Link>
                <div>
                    <h1 className="text-2xl md:text-3xl font-bold text-white">数据智能分析 (Preview)</h1>
                    <p className="text-slate-400 text-sm">Phase 3: Data Intelligence</p>
                </div>
            </div>

            {/* Top Row: MAE/MFE & Strategy */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-auto lg:h-[500px]">
                <div className="lg:col-span-2 glass-panel p-6 rounded-xl flex flex-col">
                    <h3 className="text-lg font-semibold text-white mb-4">执行分析 (MAE/MFE)</h3>
                    <div className="flex-1 min-h-0">
                        <MaeMfeScatterChart trades={trades} height={400} />
                    </div>
                </div>
                <div className="lg:col-span-1">
                    <StrategyBreakdown trades={trades} />
                </div>
            </div>

            {/* Bottom Row: Sentiment Heatmap */}
            <div>
                <SentimentHeatmap trades={trades} />
            </div>
        </div>
    );
}
