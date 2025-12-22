import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function GET() {
    try {
        // Fetch recent trades (last 7 days)
        const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();

        const { data: trades, error: tradesError } = await supabase
            .from('training_signals')
            .select('*')
            .gte('timestamp', sevenDaysAgo)
            .not('result_profit', 'is', null);

        if (tradesError) throw tradesError;

        // Fetch recent scans (decisions) to see activity
        const { data: scans, error: scansError } = await supabase
            .from('market_scans')
            .select('action, timestamp')
            .gte('timestamp', sevenDaysAgo)
            .limit(1000); // Limit to avoid heavy payload

        if (scansError) throw scansError;

        // Calculate Stats
        const totalTrades = trades?.length || 0;
        const wins = trades?.filter(t => t.result_profit > 0).length || 0;
        const totalPnL = trades?.reduce((sum, t) => sum + (t.result_profit || 0), 0) || 0;

        // Win Rate
        const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 0;

        // Decision Distribution (Combine trades + scans)
        // Note: trades have 'action', scans have 'action'
        const tradeActions = trades?.map(t => t.action) || [];
        const scanActions = scans?.map(s => s.action) || [];
        const allActions = [...tradeActions, ...scanActions];

        const actionCounts = allActions.reduce((acc, action) => {
            acc[action] = (acc[action] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

        return NextResponse.json({
            performance: {
                totalTrades,
                winRate,
                totalPnL
            },
            distribution: actionCounts,
            recent: trades?.slice(0, 5) // Last 5 trades
        });

    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
