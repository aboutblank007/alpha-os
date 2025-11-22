import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return NextResponse.json({}, { headers: corsHeaders });
}

export async function POST(request: Request) {
  try {
    if (request.method === 'OPTIONS') {
      return NextResponse.json({}, { headers: corsHeaders });
    }

    const body = await request.json();
    const positions = body.positions || [];
    
    // 1. Fetch all currently OPEN trades from DB
    const { data: dbTrades, error: fetchError } = await supabase
        .from('trades')
        .select('*')
        .eq('status', 'open');

    if (fetchError) throw fetchError;

    // 2. Compare Snapshot (positions) with DB (dbTrades)
    
    // Map for quick lookup
    // Key could be "Symbol-Side" (Simple hedging support)
    // Note: If multiple trades of same symbol/side exist, this logic needs to be robust.
    // For V1, we assume one aggregated position per symbol/side as typical in some views, 
    // OR we try to match by closest quantity/price.
    // SIMPLIFICATION: We match by Symbol + Side. 
    
    const processedDbIds = new Set();

    for (const pos of positions) {
        // Find matching DB trade
        // We look for a trade with same symbol, same side, and roughly same quantity?
        // Actually, the snapshot shows TOTAL position for that symbol.
        // If DB has multiple open trades for XAUUSD Buy, and snapshot shows 1 big XAUUSD Buy,
        // we should probably aggregate DB trades or update them all?
        // Let's assume for now 1 DB trade = 1 Table Row (simplest case).
        
        const match = dbTrades?.find(t => 
            t.symbol === pos.symbol && 
            t.side === pos.side && 
            !processedDbIds.has(t.id)
        );

        if (match) {
            // EXISTING POSITION: Update PnL and Price
            processedDbIds.add(match.id);
            
            const { error: updateError } = await supabase
                .from('trades')
                .update({
                    pnl_net: pos.pnl_net,
                    // If quantity changed externally (partial close elsewhere?), we might update it too
                    quantity: pos.quantity,
                    // Update current price? We don't store 'current_price' column yet, 
                    // but we could update 'exit_price' tentatively or just pnl.
                })
                .eq('id', match.id);
                
            if (updateError) console.error('Error updating trade:', updateError);

        } else {
            // NEW POSITION DETECTED (that wasn't in DB)
            // Insert it!
            const { error: insertError } = await supabase
                .from('trades')
                .insert([{
                    symbol: pos.symbol,
                    side: pos.side,
                    quantity: pos.quantity,
                    entry_price: pos.entry_price,
                    status: 'open',
                    pnl_net: pos.pnl_net,
                    // We don't know initial time, so it defaults to NOW
                }]);
            
            if (insertError) console.error('Error inserting new trade:', insertError);
        }
    }

    // 3. Detect Closed Trades
    // Any trade in DB that was NOT matched in the snapshot should be CLOSED
    // (Unless the snapshot is empty/partial? We assume snapshot is complete list of open positions)
    
    if (dbTrades) {
        for (const dbTrade of dbTrades) {
            if (!processedDbIds.has(dbTrade.id)) {
                // This trade is in DB as OPEN, but not in current positions snapshot.
                // Means it's closed!
                console.log(`[SYNC] Marking trade ${dbTrade.id} (${dbTrade.symbol}) as closed.`);
                
                const { error: closeError } = await supabase
                    .from('trades')
                    .update({
                        status: 'closed',
                        // We don't have exact exit price/time from snapshot since it's gone.
                        // We can leave exit_price null or use last known.
                        // Ideally we set closed_at = now.
                    })
                    .eq('id', dbTrade.id);
                
                if (closeError) console.error('Error closing trade:', closeError);
            }
        }
    }

    return NextResponse.json({ success: true }, { status: 200, headers: corsHeaders });
  } catch (error: unknown) {
    console.error('API Sync error:', error);
    const message = error instanceof Error ? error.message : 'Internal Server Error';
    return NextResponse.json(
      { error: message },
      { status: 500, headers: corsHeaders }
    );
  }
}
