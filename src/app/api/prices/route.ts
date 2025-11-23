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
    if (!process.env.OANDA_API_KEY || !process.env.OANDA_ACCOUNT_ID || process.env.OANDA_ENVIRONMENT === 'mock') {
      console.warn('OANDA API 未配置或处于模拟模式，返回模拟价格');
      
      // 如果没有配置 OANDA API，返回模拟价格
      const mockPrices: Record<string, number> = {};
      symbols.forEach(symbol => {
        // 生成基于品种的模拟价格
        if (symbol.includes('JPY')) {
          mockPrices[symbol] = 150 + Math.random() * 10;
        } else if (symbol.includes('XAU')) {
          mockPrices[symbol] = 2600 + Math.random() * 100;
        } else if (symbol.includes('BTC')) {
          mockPrices[symbol] = 95000 + Math.random() * 2000;
        } else {
          mockPrices[symbol] = 1.05 + Math.random() * 0.1;
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
 * 生成模拟 K 线数据
 */
function generateMockCandles(instrument: string, granularity: string, count: number) {
  const candles = [];
  let price = 1.0800; // 默认 EURUSD 价格
  
  if (instrument.includes('JPY')) price = 154.00;
  if (instrument.includes('XAU')) price = 2700.00;
  if (instrument.includes('BTC')) price = 98000.00;
  if (instrument.includes('GBP')) price = 1.2600;

  // 根据时间周期确定时间间隔（秒）
  let intervalSeconds = 300; // M5
  if (granularity === 'M1') intervalSeconds = 60;
  if (granularity === 'M15') intervalSeconds = 900;
  if (granularity === 'H1') intervalSeconds = 3600;
  if (granularity === 'D') intervalSeconds = 86400;

  const now = Math.floor(Date.now() / 1000);
  const volatility = price * 0.0005; // 0.05% 波动

  for (let i = count; i > 0; i--) {
    const time = now - (i * intervalSeconds);
    
    const change = (Math.random() - 0.5) * volatility * 2;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * volatility;
    const low = Math.min(open, close) - Math.random() * volatility;
    
    candles.push({
      time: time,
      open: parseFloat(open.toFixed(5)),
      high: parseFloat(high.toFixed(5)),
      low: parseFloat(low.toFixed(5)),
      close: parseFloat(close.toFixed(5)),
      volume: Math.floor(Math.random() * 100),
    });

    price = close;
  }
  
  return candles;
}

/**
 * 获取蜡烛图数据 API
 * POST /api/prices
 * Body: { instrument: 'USD_JPY', granularity: 'M1', count: 500 }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { instrument, granularity = 'M5', count = 200 } = body;

    if (!instrument) {
      return NextResponse.json(
        { error: '缺少 instrument 参数' },
        { status: 400 }
      );
    }

    const oandaInstrument = instrument.includes('_') ? instrument : convertToOandaInstrument(instrument);

    // 检查 OANDA API 配置，如果未配置或为 mock 模式，返回模拟数据
    if (!process.env.OANDA_API_KEY || !process.env.OANDA_ACCOUNT_ID || process.env.OANDA_ENVIRONMENT === 'mock') {
      console.warn('OANDA API 未配置或处于模拟模式，生成模拟 K 线数据');
      
      const mockCandles = generateMockCandles(oandaInstrument, granularity, count);
      
      return NextResponse.json({
        instrument: oandaInstrument,
        candles: mockCandles,
        granularity,
        count: mockCandles.length,
        source: 'mock'
      });
    }

    // 动态导入 getCandles 函数
    const { getCandles } = await import('@/lib/oanda');
    
    try {
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
        source: 'oanda'
      });
    } catch (oandaError) {
      console.warn('OANDA API 调用失败，降级为模拟数据:', oandaError);
      // 降级处理：如果 OANDA 失败，也返回模拟数据
      const mockCandles = generateMockCandles(oandaInstrument, granularity, count);
      
      return NextResponse.json({
        instrument: oandaInstrument,
        candles: mockCandles,
        granularity,
        count: mockCandles.length,
        source: 'mock-fallback'
    });
    }

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
