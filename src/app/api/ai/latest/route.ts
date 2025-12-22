import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";

// Init Supabase (Server Side)
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!; // Prefer Service Role if available for server
const supabase = createClient(supabaseUrl, supabaseKey);

export async function GET() {
    try {
        // Fetch latest signal (Union of training_signals and market_scans? Just active signals usually)
        // Let's check training_signals (executions) and market_scans (non-executions)
        // We want the absolute latest decision regardless of action.

        // Parallel fetch is complex for "latest of two tables".
        // Let's just fetch training_signals for "Last Trade" and market_scans for "Latest Scan".
        // Or assumes 'market_scans' has the high frequency loop data.

        const { data: scanData } = await supabase
            .from('market_scans')
            .select('*')
            .order('timestamp', { ascending: false })
            .limit(1)
            .single();

        const { data: tradeData } = await supabase
            .from('training_signals')
            .select('*')
            .order('timestamp', { ascending: false })
            .limit(1)
            .single();

        // Compare timestamps
        let latest = null;
        if (scanData && tradeData) {
            const scanTime = new Date(scanData.timestamp).getTime();
            const tradeTime = new Date(tradeData.timestamp).getTime();
            latest = scanTime > tradeTime ? scanData : tradeData;
        } else {
            latest = scanData || tradeData;
        }

        if (!latest) {
            return NextResponse.json({
                action: 'WAIT',
                confidence: 0,
                symbol: '---',
                timestamp: new Date().toISOString()
            });
        }

        // Parse features if needed, or just return top level
        let features: Record<string, any> = {};
        try {
            if (typeof latest.ai_features === 'string') {
                features = JSON.parse(latest.ai_features);
            } else {
                features = latest.ai_features || {};
            }
        } catch (e) { }

        return NextResponse.json({
            action: latest.action,
            confidence: latest.ai_features?.ai_score || 0, // Fallback if column missing, check json
            symbol: latest.symbol,
            price: latest.signal_price,
            timestamp: latest.timestamp,
            // Extract some features for display if available
            regime: features['regime'] || 'Unknown',
        });

    } catch (error) {
        return NextResponse.json({ error: "Failed to fetch AI stats" }, { status: 500 });
    }
}
