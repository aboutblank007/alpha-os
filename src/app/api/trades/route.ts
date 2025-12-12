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

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status');
    const page = parseInt(searchParams.get('page') || '0');
    const pageSize = parseInt(searchParams.get('pageSize') || '50');

    // Calculate range for pagination
    const from = page * pageSize;
    const to = from + pageSize - 1;

    let query = supabase
      .from('trades')
      .select('*', { count: 'exact' }) // Request exact count
      .order('entry_time', { ascending: false }); // Newest first by default for recent trades

    if (status) {
      query = query.eq('status', status);
    }

    const { data, error, count } = await query.range(from, to);

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json({ error: error.message }, { status: 500, headers: corsHeaders });
    }

    return NextResponse.json({
      data,
      count,
      page,
      pageSize
    }, { status: 200, headers: corsHeaders });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500, headers: corsHeaders });
  }
}

export async function POST(request: Request) {
  try {
    if (request.method === 'OPTIONS') {
      return NextResponse.json({}, { headers: corsHeaders });
    }

    const body = await request.json();

    // Basic validation
    if (!body.symbol || !body.side || body.entry_price === undefined || body.entry_price === null || !body.quantity) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400, headers: corsHeaders }
      );
    }

    // Check for existing open position to close
    const oppositeSide = body.side === 'buy' ? 'sell' : 'buy';

    const { data: openTrades, error: fetchError } = await supabase
      .from('trades')
      .select('*')
      .eq('symbol', body.symbol)
      .eq('side', oppositeSide)
      .eq('status', 'open')
      .order('created_at', { ascending: true }); // FIFO

    if (fetchError) {
      console.error('Error fetching open trades:', fetchError);
    }

    let remainingQuantity = body.quantity;
    let tradeProcessed = false;

    // If matching open trade found, close it (partial or full)
    if (openTrades && openTrades.length > 0) {
      for (const openTrade of openTrades) {
        if (remainingQuantity <= 0) break;

        const closeQty = Math.min(openTrade.quantity, remainingQuantity);

        let pnl = 0;
        const entryPrice = openTrade.entry_price;
        const exitPrice = body.entry_price || entryPrice;

        if (openTrade.side === 'buy') { // Closing a Long
          pnl = (exitPrice - entryPrice) * closeQty;
        } else { // Closing a Short
          pnl = (entryPrice - exitPrice) * closeQty;
        }

        // Full match
        if (Math.abs(openTrade.quantity - closeQty) < 0.0001) {
          const { error: updateError } = await supabase
            .from('trades')
            .update({
              status: 'closed',
              exit_price: exitPrice,
              pnl_net: pnl,
              pnl_gross: pnl,
              mae: body.mae,
              mfe: body.mfe,
            })
            .eq('id', openTrade.id);

          if (updateError) console.error('Error updating trade:', updateError);
          remainingQuantity -= closeQty;
          tradeProcessed = true;
        } else {
          // Partial match
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
                  commission: (openTrade.commission || 0) * (closeQty / openTrade.quantity),
                  status: 'closed',
                  notes: `Partial close of ${openTrade.id}`,
                  strategies: openTrade.strategies,
                  account_id: openTrade.account_id,
                  mae: body.mae,
                  mfe: body.mfe
                }
              ]);

            if (insertError) console.error('Error inserting closed partial trade:', insertError);
          }

          remainingQuantity -= closeQty;
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
            quantity: remainingQuantity,
            pnl_net: body.pnl_net || 0,
            pnl_gross: body.pnl_gross || 0,
            commission: body.commission || 0,
            status: body.status || 'open',
            notes: body.notes,
            strategies: body.strategies,
            mae: body.mae,
            mfe: body.mfe
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
