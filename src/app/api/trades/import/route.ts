import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { trades } = body;

    if (!trades || !Array.isArray(trades) || trades.length === 0) {
      return NextResponse.json(
        { error: 'No trades data provided' },
        { status: 400 }
      );
    }

    const formattedTrades = [];
    const errors = [];
    const skipped = [];
    
    // 收集所有的外部订单ID用于批量查询
    const externalOrderIds = trades
      .map(t => t.external_order_id)
      .filter(id => id && String(id).trim());

    // 如果有外部订单ID,查询已存在的订单
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
        
        // 检查是否重复
        if (trade.external_order_id && existingOrderIds.has(String(trade.external_order_id))) {
            skipped.push(`Row ${i + 2}: 订单 ${trade.external_order_id} 已存在,跳过导入`);
            continue;
        }
        
        // Essential validation
        if (!trade.symbol || !trade.side || !trade.quantity || !trade.entry_price) {
            errors.push(`Row ${i + 2}: Missing required fields (Symbol, Side, Quantity, Entry Price)`);
            continue;
        }

        try {
            const side = String(trade.side).toLowerCase();
            if (side !== 'buy' && side !== 'sell') {
                errors.push(`Row ${i + 2}: Invalid side '${trade.side}' (must be 'buy' or 'sell')`);
                continue;
            }

            // 规范化状态值 - 将各种状态映射到 'open' 或 'closed'
            let status = 'closed'; // 默认为已完成,因为大多数导入的是历史交易
            if (trade.status) {
              const statusLower = String(trade.status).toLowerCase();
              // 检查是否为未成交/开仓状态
              if (statusLower.includes('open') || 
                  statusLower.includes('pending') ||
                  statusLower.includes('未成交') ||
                  statusLower.includes('待处理')) {
                status = 'open';
              }
              // 其他所有状态(已成交、已取消、完成等)都视为 closed
            } else if (trade.exit_price) {
              // 如果有平仓价,确认是已平仓
              status = 'closed';
            } else if (!trade.exit_price && trade.entry_price) {
              // 如果只有开仓价没有平仓价,可能是持仓中
              status = 'open';
            }

            const formattedTrade: any = {
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
            };

            // 添加外部订单ID(如果存在)
            if (trade.external_order_id && String(trade.external_order_id).trim()) {
              formattedTrade.external_order_id = String(trade.external_order_id).trim();
            }

            formattedTrades.push(formattedTrade);
        } catch (e: any) {
            errors.push(`Row ${i + 2}: Formatting error - ${e.message}`);
        }
    }

    if (formattedTrades.length === 0) {
        return NextResponse.json(
            { 
              error: 'No valid trades found to import', 
              details: errors,
              skipped: skipped.length > 0 ? skipped : undefined
            },
            { status: 400 }
        );
    }

    const { data, error } = await supabase
      .from('trades')
      .insert(formattedTrades)
      .select();

    if (error) {
      console.error('Supabase import error:', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({ 
      success: true, 
      count: data.length,
      skipped: skipped.length,
      data,
      errors: errors.length > 0 ? errors : undefined,
      skippedDetails: skipped.length > 0 ? skipped : undefined
    });

  } catch (error: any) {
    console.error('Import API error:', error);
    return NextResponse.json(
      { error: 'Internal Server Error: ' + error.message },
      { status: 500 }
    );
  }
}
