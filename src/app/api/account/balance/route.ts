import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// GET - 获取账户余额和净资产
export async function GET() {
  try {
    // 获取账户信息（使用第一个账户）
    const { data: account, error: accountError } = await supabase
      .from('accounts')
      .select('*')
      .limit(1)
      .single();

    if (accountError) {
      console.error('获取账户信息错误:', accountError);
      // 如果没有账户，返回默认值
      return NextResponse.json({
        initial_balance: 0,
        current_balance: 0,
        net_asset: 0,
        total_pnl: 0,
        open_positions_pnl: 0,
      });
    }

    const initialBalance = parseFloat(account.initial_balance) || 0;

    // 获取所有已平仓交易的总盈亏
    const { data: closedTrades, error: closedError } = await supabase
      .from('trades')
      .select('pnl_net')
      .eq('status', 'closed');

    if (closedError) throw closedError;

    const totalClosedPnl = closedTrades?.reduce(
      (sum, trade) => sum + (parseFloat(trade.pnl_net) || 0),
      0
    ) || 0;

    // 获取所有持仓交易的浮动盈亏
    const { data: openTrades, error: openError } = await supabase
      .from('trades')
      .select('pnl_net')
      .eq('status', 'open');

    if (openError) throw openError;

    const totalOpenPnl = openTrades?.reduce(
      (sum, trade) => sum + (parseFloat(trade.pnl_net) || 0),
      0
    ) || 0;

    // 计算净资产 = 初始本金 + 已实现盈亏 + 浮动盈亏
    const netAsset = initialBalance + totalClosedPnl + totalOpenPnl;
    const totalPnl = totalClosedPnl + totalOpenPnl;

    // 更新账户的当前余额
    await supabase
      .from('accounts')
      .update({ current_balance: netAsset })
      .eq('id', account.id);

    return NextResponse.json({
      initial_balance: parseFloat(initialBalance.toFixed(2)),
      current_balance: parseFloat(netAsset.toFixed(2)),
      net_asset: parseFloat(netAsset.toFixed(2)),
      total_pnl: parseFloat(totalPnl.toFixed(2)),
      total_closed_pnl: parseFloat(totalClosedPnl.toFixed(2)),
      open_positions_pnl: parseFloat(totalOpenPnl.toFixed(2)),
      open_positions_count: openTrades?.length || 0,
    });
  } catch (error: unknown) {
    console.error('获取账户余额错误:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: '获取账户余额失败: ' + errorMessage },
      { status: 500 }
    );
  }
}

