import { env } from '../env';

/**
 * OANDA API 客户端
 * 用于获取实时外汇和商品价格
 */

const BASE_URL = env.OANDA_ENVIRONMENT === 'live'
  ? 'https://api-fxtrade.oanda.com'
  : 'https://api-fxpractice.oanda.com';

const STREAM_URL = env.OANDA_ENVIRONMENT === 'live'
  ? 'https://stream-fxtrade.oanda.com'
  : 'https://stream-fxpractice.oanda.com';

/**
 * 将常见的交易品种代码转换为 OANDA 格式
 * 例如: USDJPY -> USD_JPY, XAUUSD -> XAU_USD
 */
export function convertToOandaInstrument(symbol: string): string {
  symbol = symbol.toUpperCase();

  // 特殊处理黄金
  if (symbol === 'XAUUSD') return 'XAU_USD';

  // 特殊处理白银
  if (symbol === 'XAGUSD') return 'XAG_USD';

  // 处理外汇对 (6个字符)
  if (symbol.length === 6) {
    return `${symbol.substring(0, 3)}_${symbol.substring(3, 6)}`;
  }

  return symbol;
}

/**
 * 将 OANDA 格式转换回常见格式
 * 例如: USD_JPY -> USDJPY
 */
export function convertFromOandaInstrument(instrument: string): string {
  return instrument.replace('_', '');
}

/**
 * OANDA API 请求配置
 */
interface OandaRequestConfig {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: any;
}

/**
 * 发送 OANDA API 请求
 */
async function oandaRequest(endpoint: string, config: OandaRequestConfig = {}) {
  if (!env.OANDA_API_KEY) {
    console.warn('OANDA_API_KEY not configured, skipping request');
    return null; // Or throw specific error that UI can handle
  }

  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    'Authorization': `Bearer ${env.OANDA_API_KEY}`,
    'Content-Type': 'application/json',
    ...config.headers,
  };

  try {
    // 添加超时控制（10秒）
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(url, {
      method: config.method || 'GET',
      headers,
      body: config.body ? JSON.stringify(config.body) : undefined,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OANDA API 错误 [${response.status}]: ${error}`);
    }

    return response.json();
  } catch (error: any) {
    // 增强错误信息
    if (error.name === 'AbortError') {
      throw new Error(`OANDA API 请求超时（10秒）- 可能的原因：网络连接问题或需要 VPN。URL: ${BASE_URL}`);
    }

    if (error.cause?.code === 'ENOTFOUND') {
      throw new Error(`无法连接到 OANDA API (${BASE_URL}) - 请检查网络连接。如在中国大陆，可能需要 VPN。`);
    }

    if (error.cause?.code === 'ECONNREFUSED') {
      throw new Error(`OANDA API 拒绝连接 (${BASE_URL}) - 请检查防火墙设置。`);
    }

    // 重新抛出原始错误
    throw error;
  }
}

/**
 * 价格数据接口
 */
export interface OandaPrice {
  instrument: string;
  time: string;
  bids: { price: string; liquidity: number }[];
  asks: { price: string; liquidity: number }[];
  closeoutBid: string;
  closeoutAsk: string;
  status: string;
}

/**
 * 获取实时价格
 * @param instruments 交易品种数组，例如 ['USD_JPY', 'EUR_USD', 'XAU_USD']
 */
export async function getPricing(instruments: string[]): Promise<{ prices: OandaPrice[] }> {
  if (!env.OANDA_ACCOUNT_ID) {
    return { prices: [] };
  }

  const instrumentsParam = instruments.join(',');
  const endpoint = `/v3/accounts/${env.OANDA_ACCOUNT_ID}/pricing?instruments=${instrumentsParam}`;

  return oandaRequest(endpoint);
}

/**
 * 蜡烛图数据接口
 */
export interface OandaCandle {
  time: string;
  bid: { o: string; h: string; l: string; c: string };
  mid: { o: string; h: string; l: string; c: string };
  ask: { o: string; h: string; l: string; c: string };
  volume: number;
  complete: boolean;
}

/**
 * K线周期类型
 */
export type CandleGranularity =
  | 'S5' | 'S10' | 'S15' | 'S30'  // 秒级
  | 'M1' | 'M2' | 'M4' | 'M5' | 'M10' | 'M15' | 'M30'  // 分钟级
  | 'H1' | 'H2' | 'H3' | 'H4' | 'H6' | 'H8' | 'H12'  // 小时级
  | 'D' | 'W' | 'M';  // 日、周、月

/**
 * 获取历史蜡烛图数据
 * @param instrument 交易品种，例如 'USD_JPY'
 * @param granularity K线周期，默认 'M1' (1分钟)
 * @param count 数量，默认 500
 */
export async function getCandles(
  instrument: string,
  granularity: CandleGranularity = 'M1',
  count: number = 500
): Promise<{ candles: OandaCandle[] }> {
  const endpoint = `/v3/instruments/${instrument}/candles?granularity=${granularity}&count=${count}`;
  const response = await oandaRequest(endpoint);
  return response || { candles: [] };
}

/**
 * 获取账户信息
 */
export async function getAccountInfo() {
  if (!env.OANDA_ACCOUNT_ID) {
    return null;
  }

  const endpoint = `/v3/accounts/${env.OANDA_ACCOUNT_ID}`;
  return oandaRequest(endpoint);
}

/**
 * 获取多个品种的当前价格（简化版）
 * 返回 { symbol: price } 格式的对象
 */
export async function getCurrentPrices(symbols: string[]): Promise<Record<string, number>> {
  try {
    const oandaInstruments = symbols.map(convertToOandaInstrument);
    const { prices } = await getPricing(oandaInstruments);

    const result: Record<string, number> = {};

    prices.forEach((price) => {
      const symbol = convertFromOandaInstrument(price.instrument);
      // 使用中间价 (bid + ask) / 2
      const bid = parseFloat(price.closeoutBid || price.bids[0]?.price || '0');
      const ask = parseFloat(price.closeoutAsk || price.asks[0]?.price || '0');
      result[symbol] = (bid + ask) / 2;
    });

    return result;
  } catch (error) {
    console.error('获取 OANDA 价格失败:', error);
    throw error;
  }
}

