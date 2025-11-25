import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// GET - 获取每日交易统计
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get('date');
    const startDate = searchParams.get('startDate');
    const endDate = searchParams.get('endDate');

    if (date) {
      // 获取特定日期的统计
      const { data: trades, error } = await supabase
        .from('trades')
        .select('pnl_net, status')
        .gte('created_at', `${date}T00:00:00Z`)
        .lt('created_at', `${date}T23:59:59Z`)
        .eq('status', 'closed');

      if (error) throw error;

      const totalTrades = trades?.length || 0;
      const winningTrades = trades?.filter(t => t.pnl_net > 0).length || 0;
      const totalPnl = trades?.reduce((sum, t) => sum + (parseFloat(t.pnl_net) || 0), 0) || 0;

      return NextResponse.json({
        stats: {
          date,
          total_trades: totalTrades,
          winning_trades: winningTrades,
          total_pnl: parseFloat(totalPnl.toFixed(2)),
          win_rate: totalTrades > 0 ? parseFloat(((winningTrades / totalTrades) * 100).toFixed(1)) : 0,
        },
      });
    } else if (startDate && endDate) {
      // 获取日期范围的每日统计
      const { data: trades, error } = await supabase
        .from('trades')
        .select('created_at, pnl_net, status')
        .gte('created_at', `${startDate}T00:00:00Z`)
        .lte('created_at', `${endDate}T23:59:59Z`)
        .eq('status', 'closed')
        .order('created_at', { ascending: true });

      if (error) throw error;

      // 按日期分组统计
      interface DailyStats {
        date: string;
        total_trades: number;
        winning_trades: number;
        total_pnl: number;
      }
      const dailyMap = new Map<string, DailyStats>();

      trades?.forEach((trade) => {
        const tradeDate = trade.created_at.split('T')[0];
        const pnl = parseFloat(trade.pnl_net) || 0;

        if (!dailyMap.has(tradeDate)) {
          dailyMap.set(tradeDate, {
            date: tradeDate,
            total_trades: 0,
            winning_trades: 0,
            total_pnl: 0,
          });
        }

        const stats = dailyMap.get(tradeDate)!;
        stats.total_trades += 1;
        if (pnl > 0) stats.winning_trades += 1;
        stats.total_pnl += pnl;
      });

      // 转换为数组并格式化
      const stats = Array.from(dailyMap.values()).map((stat) => ({
        ...stat,
        total_pnl: parseFloat(stat.total_pnl.toFixed(2)),
        win_rate: stat.total_trades > 0
          ? parseFloat(((stat.winning_trades / stat.total_trades) * 100).toFixed(1))
          : 0,
      }));

      return NextResponse.json({ stats });
    } else {
      return NextResponse.json(
        { error: '请提供 date 或 startDate/endDate 参数' },
        { status: 400 }
      );
    }
  } catch (error: unknown) {
    console.error('获取交易统计错误:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: '获取交易统计失败: ' + errorMessage },
      { status: 500 }
    );
  }
}

