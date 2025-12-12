"use client";

import { useState, useEffect } from 'react';
import { supabase, Trade } from '@/lib/supabase';
import { AnalyticsPanel } from '@/components/analytics/AnalyticsPanel';
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
        <div className="max-w-screen-3xl mx-auto h-[calc(100vh-80px)] p-6 md:p-8 flex flex-col gap-6">
            {/* Header */}
            <div className="flex items-center gap-4 shrink-0">
                <Link href="/dashboard" className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
                    <ArrowLeft size={20} />
                </Link>
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                        深度洞察分析
                        <span className="text-xs px-2 py-0.5 rounded-full bg-accent-primary/20 text-accent-primary border border-accent-primary/20">v2.5.0</span>
                    </h1>
                    <p className="text-slate-400 text-sm">AI Performance & Execution Quality</p>
                </div>
            </div>

            {/* Main Content Area - Full height */}
            <div className="flex-1 min-h-0">
                <AnalyticsPanel trades={trades} />
            </div>
        </div>
    );
}
