import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { parseThinkMarketsCSV, reconstructTrades } from '@/lib/import-utils';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { trades, csvContent, source } = body;

    let formattedTrades: Record<string, unknown>[] = [];
    const skipped: string[] = [];
    const errors: string[] = [];

    if (source === 'thinkmarkets' && csvContent) {
      try {
        // 1. Parse CSV
        const parsedRows = parseThinkMarketsCSV(csvContent);
        // 2. Reconstruct Trades (FIFO)
        const reconstructed = reconstructTrades(parsedRows);

        // 3. Format for DB
        formattedTrades = reconstructed.map(t => ({
          symbol: t.symbol,
          side: t.side,
          quantity: t.quantity,
          entry_price: t.entryPrice,
          exit_price: t.exitPrice,
          entry_time: t.entryTime.toISOString(),
          exit_time: t.exitTime.toISOString(),
          pnl_net: t.pnl,
          commission: t.commission,
          swap: t.swap,
          status: t.status,
          external_order_id: t.external_order_id,
          notes: t.notes || `Linked Orders: ${t.related_order_ids.join(', ')}`,
          source: 'import_thinkmarkets',
          created_at: t.exitTime.toISOString() // Use exit time as record creation time roughly
        }));

      } catch (e: unknown) {
        console.error("ThinkMarkets Import Error:", e);
        const errorMessage = e instanceof Error ? e.message : String(e);
        return NextResponse.json({ error: 'ThinkMarkets Parsing Error: ' + errorMessage }, { status: 400 });
      }
    } else if (trades && Array.isArray(trades)) {
      // --- GENERIC LOGIC (Legacy Support) ---
      const externalOrderIds = trades
        .map((t: Record<string, unknown>) => t.external_order_id)
        .filter((id: unknown) => id && String(id).trim());

      let existingOrderIds = new Set<string>();
      if (externalOrderIds.length > 0) {
        const { data: existingTrades } = await supabase
          .from('trades')
          .select('external_order_id')
          .in('external_order_id', externalOrderIds);

        if (existingTrades) {
          existingOrderIds = new Set(existingTrades.map(t => t.external_order_id));
        }
      }

      for (let i = 0; i < trades.length; i++) {
        const trade = trades[i];

        if (trade.external_order_id && existingOrderIds.has(String(trade.external_order_id))) {
          skipped.push(`Row ${i + 2}: Order ${trade.external_order_id} exists`);
          continue;
        }

        if (!trade.symbol || !trade.side || !trade.quantity || !trade.entry_price) {
          errors.push(`Row ${i + 2}: Missing fields`);
          continue;
        }

        try {
          const side = String(trade.side).toLowerCase();
          if (side !== 'buy' && side !== 'sell') continue;

          let status = 'closed';
          if (trade.status) {
            const s = String(trade.status).toLowerCase();
            if (s.includes('open') || s.includes('pending') || s.includes('未成交')) status = 'open';
          } else if (!trade.exit_price && trade.entry_price) {
            status = 'open';
          }

          formattedTrades.push({
            symbol: String(trade.symbol).toUpperCase(),
            side: side,
            entry_price: parseFloat(trade.entry_price),
            exit_price: trade.exit_price ? parseFloat(trade.exit_price) : null,
            quantity: parseFloat(trade.quantity),
            pnl_net: trade.pnl_net ? parseFloat(trade.pnl_net) : 0,
            commission: trade.commission ? parseFloat(trade.commission) : 0,
            swap: trade.swap ? parseFloat(trade.swap) : 0,
            status: status,
            notes: trade.notes || 'Imported via CSV',
            created_at: trade.date ? new Date(trade.date).toISOString() : new Date().toISOString(),
            external_order_id: trade.external_order_id ? String(trade.external_order_id).trim() : null,
            source: 'import_generic'
          });
        } catch (e: unknown) {
          const errorMessage = e instanceof Error ? e.message : String(e);
          errors.push(`Row ${i + 2}: ${errorMessage}`);
        }
      }
    } else {
      return NextResponse.json({ error: 'No valid data provided' }, { status: 400 });
    }

    // --- COMMON INSERTION LOGIC ---
    if (formattedTrades.length === 0) {
      return NextResponse.json({
        error: 'No valid trades to import',
        skipped: skipped.length > 0 ? skipped : undefined,
        details: errors
      }, { status: 400 });
    }

    // Check duplicates for ThinkMarkets trades (batch check)
    if (source === 'thinkmarkets') {
      const extIds = formattedTrades.map(t => t.external_order_id).filter(id => id);
      if (extIds.length > 0) {
        // Supabase `in` limit is roughly 65535 params, safe here
        const { data: existing } = await supabase.from('trades').select('external_order_id').in('external_order_id', extIds);
        const existingSet = new Set(existing?.map(t => t.external_order_id));

        const initialCount = formattedTrades.length;
        formattedTrades = formattedTrades.filter(t => !existingSet.has(t.external_order_id));

        if (formattedTrades.length < initialCount) {
          skipped.push(`${initialCount - formattedTrades.length} duplicate records skipped`);
        }
      }
    }

    if (formattedTrades.length === 0) {
      return NextResponse.json({ success: true, count: 0, skipped: skipped.length, message: 'All duplicates skipped' });
    }

    // Insert in batches of 100 to avoid payload limits
    const BATCH_SIZE = 100;
    let totalInserted = 0;

    for (let i = 0; i < formattedTrades.length; i += BATCH_SIZE) {
      const batch = formattedTrades.slice(i, i + BATCH_SIZE);
      const { error } = await supabase.from('trades').insert(batch);
      if (error) {
        console.error('Batch insert error:', error);
        throw error;
      }
      totalInserted += batch.length;
    }

    return NextResponse.json({
      success: true,
      count: totalInserted,
      skipped: skipped.length,
      skippedDetails: skipped,
      errors: errors.length > 0 ? errors : undefined
    });

  } catch (error: unknown) {
    console.error('Import API error:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'Internal Server Error: ' + errorMessage },
      { status: 500 }
    );
  }
}
