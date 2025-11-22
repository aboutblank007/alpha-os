import { NextResponse } from 'next/server';
import { getCurrentPrices, convertToOandaInstrument } from '@/lib/oanda';

/**
 * 获取实时价格 API
 * GET /api/prices?symbols=USDJPY,EURUSD,XAUUSD
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbolsParam = searchParams.get('symbols');

    if (!symbolsParam) {
      return NextResponse.json(
        { error: '缺少 symbols 参数' },
        { status: 400 }
      );
    }

    // 解析品种列表
    const symbols = symbolsParam.split(',').map(s => s.trim().toUpperCase());

    if (symbols.length === 0) {
      return NextResponse.json(
        { error: '品种列表不能为空' },
        { status: 400 }
      );
    }

    // 检查 OANDA API 配置
    if (!process.env.OANDA_API_KEY || !process.env.OANDA_ACCOUNT_ID) {
      console.warn('OANDA API 未配置，返回模拟价格');
      
      // 如果没有配置 OANDA API，返回模拟价格
      const mockPrices: Record<string, number> = {};
      symbols.forEach(symbol => {
        // 生成基于品种的模拟价格
        if (symbol.includes('JPY')) {
          mockPrices[symbol] = 150 + Math.random() * 10;
        } else if (symbol.includes('XAU')) {
          mockPrices[symbol] = 4000 + Math.random() * 100;
        } else {
          mockPrices[symbol] = 1.1 + Math.random() * 0.1;
        }
      });

      return NextResponse.json({
        prices: mockPrices,
        timestamp: new Date().toISOString(),
        source: 'mock',
      });
    }

    // 从 OANDA 获取真实价格
    const prices = await getCurrentPrices(symbols);

    return NextResponse.json({
      prices,
      timestamp: new Date().toISOString(),
      source: 'oanda',
    });

  } catch (error: any) {
    console.error('获取价格失败:', error);
    
    return NextResponse.json(
      { 
        error: '获取价格失败', 
        message: error.message,
        details: error.stack 
      },
      { status: 500 }
    );
  }
}

/**
 * 获取蜡烛图数据 API
 * POST /api/prices
 * Body: { instrument: 'USD_JPY', granularity: 'M1', count: 500 }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { instrument, granularity = 'M1', count = 500 } = body;

    if (!instrument) {
      return NextResponse.json(
        { error: '缺少 instrument 参数' },
        { status: 400 }
      );
    }

    // 检查 OANDA API 配置
    if (!process.env.OANDA_API_KEY || !process.env.OANDA_ACCOUNT_ID) {
      return NextResponse.json(
        { error: 'OANDA API 未配置' },
        { status: 503 }
      );
    }

    // 动态导入 getCandles 函数
    const { getCandles } = await import('@/lib/oanda');
    
    const oandaInstrument = convertToOandaInstrument(instrument);
    const { candles } = await getCandles(oandaInstrument, granularity, count);

    // 转换为前端友好的格式
    const formattedCandles = candles.map(candle => ({
      time: new Date(candle.time).getTime() / 1000, // TradingView 使用秒级时间戳
      open: parseFloat(candle.mid.o),
      high: parseFloat(candle.mid.h),
      low: parseFloat(candle.mid.l),
      close: parseFloat(candle.mid.c),
      volume: candle.volume,
    }));

    return NextResponse.json({
      instrument: oandaInstrument,
      candles: formattedCandles,
      granularity,
      count: formattedCandles.length,
    });

  } catch (error: any) {
    console.error('获取蜡烛图数据失败:', error);
    
    return NextResponse.json(
      { 
        error: '获取蜡烛图数据失败', 
        message: error.message 
      },
      { status: 500 }
    );
  }
}

