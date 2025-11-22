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
    
    // Basic validation
    // Note: entry_price can be 0 for MKT orders, so we check for undefined/null instead of truthiness
    if (!body.symbol || !body.side || body.entry_price === undefined || body.entry_price === null || !body.quantity) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400, headers: corsHeaders }
      );
    }

    // Check for existing open position to close
    // Logic: Same symbol, opposite side, status='open'
    const oppositeSide = body.side === 'buy' ? 'sell' : 'buy';
    
    // console.log(`[API DEBUG] Trying to close position: Symbol=${body.symbol}, Side=${body.side} (looking for ${oppositeSide}), Qty=${body.quantity}`);

    const { data: openTrades, error: fetchError } = await supabase
      .from('trades')
      .select('*')
      .eq('symbol', body.symbol)
      .eq('side', oppositeSide)
      .eq('status', 'open')
      .order('created_at', { ascending: true }); // FIFO (First In First Out)

    if (fetchError) {
        console.error('Error fetching open trades:', fetchError);
    } else {
        // console.log(`[API DEBUG] Found ${openTrades?.length || 0} matching open trades.`);
    }

    let remainingQuantity = body.quantity;
    let tradeProcessed = false;

    // If matching open trade found, close it (partial or full)
    if (openTrades && openTrades.length > 0) {
        for (const openTrade of openTrades) {
            if (remainingQuantity <= 0) break;

            const closeQty = Math.min(openTrade.quantity, remainingQuantity);
            
            // Calculate PnL
            // Buy to Open, Sell to Close: (Exit - Entry) * Qty
            // Sell to Open, Buy to Close: (Entry - Exit) * Qty
            let pnl = 0;
            const entryPrice = openTrade.entry_price;
            // If current order price is 0 (MKT), use entry price to avoid wild PnL, 
            // OR ideally we should wait for price update. For now, if 0, result is 0 PnL.
            const exitPrice = body.entry_price || entryPrice; 

            if (openTrade.side === 'buy') { // Closing a Long
                pnl = (exitPrice - entryPrice) * closeQty;
            } else { // Closing a Short
                pnl = (entryPrice - exitPrice) * closeQty;
            }

            // Update the existing trade
            // Note: simplified logic. If partial close, we ideally split the trade.
            // Here we assume full match or just update the status for simplicity first.
            if (Math.abs(openTrade.quantity - closeQty) < 0.0001) {
                // console.log(`[API DEBUG] Full match found! Updating trade ID: ${openTrade.id}`);
                // Full match
                const { error: updateError } = await supabase
                    .from('trades')
                    .update({
                        status: 'closed',
                        exit_price: exitPrice,
                        pnl_net: pnl,
                        pnl_gross: pnl,
                    })
                    .eq('id', openTrade.id);
                
                if (updateError) console.error('Error updating trade:', updateError);
                remainingQuantity -= closeQty;
                tradeProcessed = true;
            } else {
                // Partial match: The open trade is larger than the closing order
                // Logic:
                // 1. Update the EXISTING open trade to reduce its quantity (it stays open)
                // 2. Insert a NEW 'closed' trade representing the portion that was just closed
                
                // console.log(`[API DEBUG] Partial match. Splitting trade ${openTrade.id}. Open: ${openTrade.quantity} -> ${openTrade.quantity - closeQty}. Closing: ${closeQty}`);

                const newOpenQty = openTrade.quantity - closeQty;
                
                // 1. Reduce quantity of existing open trade
                const { error: updateError } = await supabase
                    .from('trades')
                    .update({ quantity: newOpenQty })
                    .eq('id', openTrade.id);

                if (updateError) {
                    console.error('Error updating partial trade:', updateError);
                } else {
                    // 2. Insert the closed portion
                    const { error: insertError } = await supabase
                        .from('trades')
                        .insert([
                            {
                                symbol: openTrade.symbol,
                                side: openTrade.side,
                                entry_price: openTrade.entry_price,
                                exit_price: exitPrice,
                                quantity: closeQty,
                                pnl_net: pnl,
                                pnl_gross: pnl,
                                commission: (openTrade.commission || 0) * (closeQty / openTrade.quantity), // Pro-rated commission
                                status: 'closed',
                                notes: `Partial close of ${openTrade.id}`,
                                strategies: openTrade.strategies,
                                account_id: openTrade.account_id
                            }
                        ]);
                    
                    if (insertError) console.error('Error inserting closed partial trade:', insertError);
                }

                remainingQuantity -= closeQty; // Should be 0 now if closeQty was remainingQuantity
                tradeProcessed = true;
            }
        }
    }

    // If no matching open trade found, or quantity remains, insert as new open position
    if (!tradeProcessed || remainingQuantity > 0) {
        const { data, error } = await supabase
          .from('trades')
          .insert([
            {
              symbol: body.symbol,
              side: body.side,
              entry_price: body.entry_price,
              exit_price: body.exit_price,
              quantity: remainingQuantity, // Insert remaining qty
              pnl_net: body.pnl_net || 0,
              pnl_gross: body.pnl_gross || 0,
              commission: body.commission || 0,
              status: body.status || 'open',
              notes: body.notes,
              strategies: body.strategies,
            },
          ])
          .select();

        if (error) {
            console.error('Supabase error:', error);
            return NextResponse.json({ error: error.message }, { status: 500, headers: corsHeaders });
        }
        return NextResponse.json({ data }, { status: 201, headers: corsHeaders });
    }

    return NextResponse.json({ message: 'Trade processed/closed' }, { status: 200, headers: corsHeaders });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500, headers: corsHeaders }
    );
  }
}
