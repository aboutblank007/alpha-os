import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { mt5Client, MT5Candle } from '@/lib/mt5-client';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

interface Trade {
    entry_price: number;
    exit_price: number;
    side: string;
    entry_time: string;
    exit_time: string;
    pnl_net: number;
    id: string;
    symbol: string;
}

function calculateMaeMfe(trade: Trade, candles: MT5Candle[]) {
    let maxProfit = 0;
    let maxLoss = 0;
    const entryPrice = trade.entry_price;
    const isLong = trade.side === 'buy';

    const entryTime = new Date(trade.entry_time).getTime();
    const exitTime = new Date(trade.exit_time).getTime();

    for (const candle of candles) {
        // Convert MT5 timestamp (seconds) to ms
        const candleTime = candle.time * 1000;

        // Filter candles strictly within trade duration
        // Note: With range-based fetching, candles should be mostly relevant, but precision check is good.
        // However, allow small buffer if needed? No, strict is better for MAE/MFE.
        if (candleTime < entryTime || candleTime > exitTime) continue;

        const high = candle.high;
        const low = candle.low;

        if (isLong) {
            // Long: High is Profit, Low is Loss
            const profitHigh = high - entryPrice;
            const lossLow = low - entryPrice; // Negative value

            if (profitHigh > maxProfit) maxProfit = profitHigh;
            if (lossLow < maxLoss) maxLoss = lossLow;
        } else {
            // Short: Low is Profit, High is Loss
            const profitLow = entryPrice - low;
            const lossHigh = entryPrice - high; // Negative value

            if (profitLow > maxProfit) maxProfit = profitLow;
            if (lossHigh < maxLoss) maxLoss = lossHigh;
        }
    }

    // Convert Price Difference to Dollar Value (Approximate)
    let ratio = 1;
    const priceDiff = isLong ? (trade.exit_price - entryPrice) : (entryPrice - trade.exit_price);

    if (Math.abs(priceDiff) > 0.000001 && trade.pnl_net) {
        ratio = Math.abs(trade.pnl_net) / Math.abs(priceDiff);
    } else {
        // Fallback if PnL is 0 or price diff is 0
        // Use quantity as rough scalar if symbol seems standard
        // ratio = trade.quantity; 
    }

    // Use ratio to convert price diff to currency value
    return {
        mae: maxLoss * ratio,
        mfe: maxProfit * ratio
    };
}

export async function POST() {
    try {
        // 1. Get batch of trades needing enrichment
        const { data: trades } = await supabase
            .from('trades')
            .select('*')
            .eq('status', 'closed')
            .or('mae.is.null,mae.eq.0')
            .not('entry_time', 'is', null)
            .not('exit_time', 'is', null)
            .order('exit_time', { ascending: false }) // Newest first
            .limit(5); // Process 5 at a time to manage load

        if (!trades || trades.length === 0) {
            return NextResponse.json({ message: 'No trades to enrich' });
        }

        const results = [];

        for (const trade of trades) {
            try {
                const entryTime = new Date(trade.entry_time);
                const exitTime = new Date(trade.exit_time);

                // Add buffer (e.g., 1 hour before and after) to ensure we catch the peaks
                // even if clock sync is slightly off
                const from = new Date(entryTime.getTime() - 3600000);
                const to = new Date(exitTime.getTime() + 3600000);

                // Fetch History using new Range Support
                // We don't need 'count' anymore, but pass dummy value
                const candles = await mt5Client.getHistory(trade.symbol, 'M1', 0, from, to);

                if (candles && candles.length > 0) {
                    const { mae, mfe } = calculateMaeMfe(trade, candles);

                    // Valid numbers check
                    if (!isNaN(mae) && !isNaN(mfe)) {
                        await supabase
                            .from('trades')
                            .update({ mae, mfe })
                            .eq('id', trade.id);

                        results.push({ id: trade.id, status: 'updated', mae, mfe });
                    } else {
                        results.push({ id: trade.id, status: 'failed', reason: 'calc error' });
                    }
                } else {
                    results.push({ id: trade.id, status: 'failed', reason: 'no candle data' });
                }
            } catch (e: unknown) {
                const errorMessage = e instanceof Error ? e.message : String(e);
                console.error(`Enrich error for trade ${trade.id}:`, errorMessage);
                results.push({ id: trade.id, status: 'error', error: errorMessage });
            }
        }

        return NextResponse.json({
            processed: results.filter(r => r.status === 'updated').length,
            results
        });

    } catch (error: unknown) {
        console.error('Enrich API Error:', error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: errorMessage }, { status: 500 });
    }
}
