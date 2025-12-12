import { env } from '@/env';
import { z } from 'zod';

// Define Zod schemas for validation
const MT5CandleSchema = z.object({
    time: z.number(),
    open: z.number(),
    high: z.number(),
    low: z.number(),
    close: z.number(),
    tick_volume: z.number(),
});

const MT5HistoryResponseSchema = z.object({
    request_id: z.string(),
    symbol: z.string(),
    timeframe: z.string(),
    count: z.number(),
    data: z.array(MT5CandleSchema),
});

export type MT5Candle = z.infer<typeof MT5CandleSchema>;
export type MT5HistoryResponse = z.infer<typeof MT5HistoryResponseSchema>;

export class MT5Client {
    private baseUrl: string;
    private retryDelay: number = 1000;
    private maxRetries: number = 3;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    /**
     * Execute a fetch request with exponential backoff
     */
    private async fetchWithRetry(url: string, options: RequestInit, retries = 0): Promise<Response> {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                // Only retry on 5xx errors or specific 4xx that might be transient
                if (response.status >= 500 || response.status === 429) {
                    throw new Error(`Server Error: ${response.status}`);
                }
                return response; // Return 4xx errors directly without retry (e.g., 404, 400)
            }
            return response;
        } catch (error) {
            if (retries < this.maxRetries) {
                const delay = this.retryDelay * Math.pow(2, retries); // 1s, 2s, 4s
                console.warn(`MT5 Client: Request failed, retrying in ${delay}ms... (${retries + 1}/${this.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.fetchWithRetry(url, options, retries + 1);
            }
            throw error;
        }
    }

    /**
     * 获取历史 K 线数据
     * @param symbol 交易品种 (e.g. "EURUSD")
     * @param timeframe 时间周期 (e.g. "M1", "H1")
     * @param count 数量 (max 1000)
     * @param from 开始时间 (可选)
     * @param to 结束时间 (可选)
     */
    async getHistory(symbol: string, timeframe: string, count: number = 100, from?: Date, to?: Date): Promise<MT5Candle[]> {
        try {
            const url = new URL(`${this.baseUrl}/history`);
            url.searchParams.append('symbol', symbol);
            url.searchParams.append('timeframe', timeframe);

            if (from && to) {
                // Convert to unix timestamp (seconds)
                url.searchParams.append('from_ts', Math.floor(from.getTime() / 1000).toString());
                url.searchParams.append('to_ts', Math.floor(to.getTime() / 1000).toString());
            } else {
                url.searchParams.append('count', count.toString());
            }

            const response = await this.fetchWithRetry(url.toString(), {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                // 设置较长的超时，因为需要等待 EA 响应
                signal: AbortSignal.timeout(65000)
            });

            if (!response.ok) {
                throw new Error(`MT5 Bridge Error: ${response.status} ${response.statusText}`);
            }

            const rawData = await response.json();

            // Validate response data with Zod
            const result = MT5HistoryResponseSchema.parse(rawData);

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
            const response = await this.fetchWithRetry(`${this.baseUrl}/status`, {
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
