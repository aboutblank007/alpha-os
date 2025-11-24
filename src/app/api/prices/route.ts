import { NextResponse } from 'next/server';
import { mt5Client } from '@/lib/mt5-client';
import { env } from '@/env';

/**
 * 符号格式转换工具函数
 */
function normalizeSymbol(symbol: string): string {
    // EUR_USD -> EURUSD, USD_JPY -> USDJPY
    return symbol.replace('_', '');
}

function addUnderscoreToSymbol(symbol: string): string {
    // EURUSD -> EUR_USD (simplified)
    if (symbol.length === 6) {
        return `${symbol.slice(0, 3)}_${symbol.slice(3)}`;
    }
    return symbol;
}

/**
 * 映射图表周期到 MT5 时间框架
 */
function mapGranularityToMT5(g: string): string {
    if (g === 'S5' || g === 'S10' || g === 'S15' || g === 'S30') return 'M1'; // 秒级降级为 M1
    if (g === 'M1') return 'M1';
    if (g === 'M2') return 'M2';
    if (g === 'M3') return 'M3';
    if (g === 'M4') return 'M4';
    if (g === 'M5') return 'M5';
    if (g === 'M6') return 'M6';
    if (g === 'M10') return 'M10';
    if (g === 'M12') return 'M12';
    if (g === 'M15') return 'M15';
    if (g === 'M20') return 'M20';
    if (g === 'M30') return 'M30';
    if (g === 'H1') return 'H1';
    if (g === 'H2') return 'H2';
    if (g === 'H3') return 'H3';
    if (g === 'H4') return 'H4';
    if (g === 'H6') return 'H6';
    if (g === 'H8') return 'H8';
    if (g === 'H12') return 'H12';
    if (g === 'D' || g === 'D1') return 'D1';
    if (g === 'W' || g === 'W1') return 'W1';
    if (g === 'M' || g === 'MN1') return 'MN1';
    return 'M5';
}

/**
 * 获取实时价格 API
 * GET /api/prices?symbols=USDJPY,EURUSD,XAUUSD
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbolsParam = searchParams.get('symbols');

    if (!symbolsParam) {
      return NextResponse.json({ error: '缺少 symbols 参数' }, { status: 400 });
    }

    // 解析品种列表
    const symbols = symbolsParam.split(',').map(s => s.trim().toUpperCase());

    if (symbols.length === 0) {
      return NextResponse.json({ error: '品种列表不能为空' }, { status: 400 });
    }

    // 优先尝试 MT5 Bridge
    try {
        const status = await mt5Client.getStatus();
        
        if (status && status.symbol_prices) {
            const mt5Prices = status.symbol_prices;
            const prices: Record<string, number> = {};
            
            let foundAny = false;
            symbols.forEach(s => {
                const cleanSymbol = normalizeSymbol(s); // USD_JPY -> USDJPY
                const data = mt5Prices[cleanSymbol] || mt5Prices[s];
                
                if (data) {
                    // 使用中间价
                    prices[cleanSymbol] = (data.bid + data.ask) / 2;
                    foundAny = true;
                } else {
                    // 如果 MT5 没有该品种数据，尝试生成模拟数据防止前端报错
                    // 或者保留 undefined 让前端处理
                    // 这里为了平滑过渡，对未找到的品种不做处理，留给 Mock 兜底
                }
            });

            if (foundAny) {
                return NextResponse.json({
                    prices,
                    timestamp: new Date().toISOString(),
                    source: 'mt5-bridge',
                });
            }
        }
    } catch (error) {
        console.warn('MT5 Bridge 连接失败，尝试使用其他源:', error);
    }

    // Mock Fallback Logic
    console.warn('使用模拟数据作为兜底');
    const mockPrices: Record<string, number> = {};
    symbols.forEach(symbol => {
        const clean = normalizeSymbol(symbol);
        if (clean.includes('JPY')) {
            mockPrices[clean] = 150 + Math.random() * 10;
        } else if (clean.includes('XAU')) {
            mockPrices[clean] = 2600 + Math.random() * 100;
        } else if (clean.includes('BTC')) {
            mockPrices[clean] = 95000 + Math.random() * 2000;
        } else {
            mockPrices[clean] = 1.05 + Math.random() * 0.1;
        }
    });

    return NextResponse.json({
        prices: mockPrices,
        timestamp: new Date().toISOString(),
        source: 'mock-fallback',
    });

  } catch (error: any) {
    console.error('获取价格失败:', error);
    return NextResponse.json(
      { error: '获取价格失败', message: error.message },
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
      return NextResponse.json({ error: '缺少 instrument 参数' }, { status: 400 });
    }

    const mt5Symbol = normalizeSymbol(instrument); // USD_JPY -> USDJPY
    const mt5Timeframe = mapGranularityToMT5(granularity);

    // 优先尝试 MT5 Bridge
    try {
        const candles = await mt5Client.getHistory(mt5Symbol, mt5Timeframe, count);
        
        // 转换为前端友好格式 (MT5 Bridge 返回的已经是 time(秒), open, high, low, close, tick_volume)
        const formattedCandles = candles.map(candle => ({
            time: candle.time, // 秒级时间戳
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.tick_volume,
        }));

        return NextResponse.json({
            instrument: mt5Symbol,
            candles: formattedCandles,
            granularity,
            count: formattedCandles.length,
            source: 'mt5-bridge'
        });

    } catch (mt5Error) {
        console.warn('MT5 Bridge 获取历史数据失败，降级为模拟数据:', mt5Error);
    }

    // Mock Fallback
    const mockCandles = generateMockCandles(mt5Symbol, granularity, count);
    return NextResponse.json({
        instrument: mt5Symbol,
        candles: mockCandles,
        granularity,
        count: mockCandles.length,
        source: 'mock-fallback'
    });

  } catch (error: any) {
    console.error('获取蜡烛图数据失败:', error);
    return NextResponse.json(
      { error: '获取蜡烛图数据失败', message: error.message },
      { status: 500 }
    );
  }
}
