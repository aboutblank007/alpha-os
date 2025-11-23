import { env } from '@/env';

export interface MT5Candle {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    tick_volume: number;
}

export interface MT5HistoryResponse {
    request_id: string;
    symbol: string;
    timeframe: string;
    count: number;
    data: MT5Candle[];
}

export class MT5Client {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    /**
     * 获取历史 K 线数据
     * @param symbol 交易品种 (e.g. "EURUSD")
     * @param timeframe 时间周期 (e.g. "M1", "H1")
     * @param count 数量 (max 1000)
     */
    async getHistory(symbol: string, timeframe: string, count: number = 100): Promise<MT5Candle[]> {
        try {
            const url = new URL(`${this.baseUrl}/history`);
            url.searchParams.append('symbol', symbol);
            url.searchParams.append('timeframe', timeframe);
            url.searchParams.append('count', count.toString());

            const response = await fetch(url.toString(), {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                // 设置较长的超时，因为需要等待 EA 响应
                signal: AbortSignal.timeout(30000) 
            });

            if (!response.ok) {
                throw new Error(`MT5 Bridge Error: ${response.status} ${response.statusText}`);
            }

            const result: MT5HistoryResponse = await response.json();
            return result.data;
        } catch (error) {
            console.error('MT5 getHistory failed:', error);
            throw error;
        }
    }

    /**
     * 获取当前市场状态（包含所有活跃品种报价）
     */
    async getStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/status`, {
                next: { revalidate: 1 } // Next.js 缓存控制
            });
            if (!response.ok) throw new Error('Failed to get status');
            return await response.json();
        } catch (error) {
            console.error('MT5 getStatus failed:', error);
            return null;
        }
    }
}

export const mt5Client = new MT5Client(env.TRADING_BRIDGE_API_URL || 'http://localhost:8000');

