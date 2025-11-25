import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { mt5Client } from '@/lib/mt5-client';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

interface Discrepancy {
    type: string;
    trade_id?: string;
    ticket?: string;
    symbol: string;
    reason: string;
}

export async function GET(request: Request) {
    // Secure this endpoint
    const authHeader = request.headers.get('authorization');
    if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
        // Allow local debugging if CRON_SECRET is not set or for dev
        if (process.env.NODE_ENV !== 'development') {
            return new NextResponse('Unauthorized', { status: 401 });
        }
    }

    try {
        // 1. Get MT5 Status
        const status = await mt5Client.getStatus();
        if (!status || !status.last_mt5_update || !status.last_mt5_update.positions) {
            return NextResponse.json({ error: 'Failed to fetch MT5 status' }, { status: 500 });
        }

        const mt5Positions = status.last_mt5_update.positions;
        
        // 2. Get Supabase Open Trades
        const { data: dbTrades, error } = await supabase
            .from('trades')
            .select('*')
            .eq('status', 'open');

        if (error) throw error;

        const discrepancies: Discrepancy[] = [];

        // 3. Compare
        // Check for trades in DB that are NOT in MT5 (Ghost trades in DB)
        // We match by external_order_id (Position ID) or Ticket if stored
        
        // Create a map of MT5 positions for O(1) lookup
        // Assuming ticket is unique. We might store ticket as external_ticket or in notes.
        // In our schema, we have external_order_id which usually maps to Position ID.
        const mt5PositionIds = new Set(mt5Positions.map((p: { ticket: number }) => p.ticket.toString())); 

        for (const dbTrade of dbTrades) {
            // Try to find matching MT5 position
            // We look for external_order_id (Position ID) or external_ticket
            let matchFound = false;
            
            // Check via external_order_id (Preferred)
            if (dbTrade.external_order_id && mt5PositionIds.has(dbTrade.external_order_id)) {
                matchFound = true;
            } 
            // Fallback: Check via notes if ticket is embedded
            else if (dbTrade.notes && mt5Positions.some((p: { ticket: number }) => dbTrade.notes.includes(p.ticket.toString()))) {
                matchFound = true;
            }

            if (!matchFound) {
                discrepancies.push({
                    type: 'GHOST_IN_DB',
                    trade_id: dbTrade.id,
                    symbol: dbTrade.symbol,
                    reason: 'Open in DB but not found in MT5'
                });
                
                // Auto-fix option: Mark as closed? Or just alert?
                // For safety, we just log/alert first.
            }
        }

        // Check for trades in MT5 that are NOT in DB (Missing in DB)
        for (const mt5Pos of mt5Positions) {
            const ticket = mt5Pos.ticket.toString();
            const existsInDb = dbTrades.some(t => 
                t.external_order_id === ticket || 
                (t.notes && t.notes.includes(ticket))
            );

            if (!existsInDb) {
                discrepancies.push({
                    type: 'MISSING_IN_DB',
                    ticket: ticket,
                    symbol: mt5Pos.symbol,
                    reason: 'Open in MT5 but not found in DB'
                });
            }
        }

        return NextResponse.json({
            status: 'checked',
            mt5_count: mt5Positions.length,
            db_count: dbTrades.length,
            discrepancies,
            timestamp: new Date().toISOString()
        });

    } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: errorMessage }, { status: 500 });
    }
}
