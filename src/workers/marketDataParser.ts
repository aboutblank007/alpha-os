/**
 * 市场数据解析 Web Worker
 * 
 * 目的: 在独立线程中解析高频 CSV Tick 数据，避免阻塞主 UI 线程
 * 设计: 响应 M2 Pro 异构核心调度，将解析任务调度至 E-Cores
 */

import type { MarketTick, WorkerInputMessage, WorkerOutputMessage } from '@/types/quantum';

/**
 * 解析单条 CSV Tick 数据
 * 格式: TICK,timestamp,symbol,bid,ask,volume,wick_ratio,vol_density,vol_shock[,spread,tick_rate,bid_ask_imbalance]
 */
function parseTickCSV(csvLine: string): MarketTick | null {
    try {
        const parts = csvLine.trim().split(',');

        if (parts.length < 9 || parts[0] !== 'TICK') {
            return null;
        }

        return {
            type: 'TICK',
            timestamp: parseInt(parts[1], 10),
            symbol: parts[2],
            bid: parseFloat(parts[3]),
            ask: parseFloat(parts[4]),
            volume: parseInt(parts[5], 10),
            wickRatio: parseFloat(parts[6]),
            volumeDensity: parseFloat(parts[7]),
            volumeShock: parseFloat(parts[8]),
            // 可选扩展字段
            spread: parts[9] ? parseInt(parts[9], 10) : undefined,
            tickRate: parts[10] ? parseInt(parts[10], 10) : undefined,
            bidAskImbalance: parts[11] ? parseFloat(parts[11]) : undefined,
        };
    } catch (error) {
        console.error('[Worker] Parse error:', error);
        return null;
    }
}

/**
 * 批量解析 CSV 数据
 */
function parseBatch(lines: string[]): MarketTick[] {
    const results: MarketTick[] = [];

    for (const line of lines) {
        const tick = parseTickCSV(line);
        if (tick) {
            results.push(tick);
        }
    }

    return results;
}

// Worker 消息处理
self.onmessage = (event: MessageEvent<WorkerInputMessage>) => {
    const { type, data } = event.data;

    try {
        switch (type) {
            case 'PARSE_TICK': {
                const tick = parseTickCSV(data as string);
                const response: WorkerOutputMessage = tick
                    ? { type: 'TICK_PARSED', data: tick }
                    : { type: 'ERROR', error: 'Invalid tick format' };
                self.postMessage(response);
                break;
            }

            case 'PARSE_BATCH': {
                const ticks = parseBatch(data as string[]);
                const response: WorkerOutputMessage = {
                    type: 'BATCH_PARSED',
                    data: ticks,
                };
                self.postMessage(response);
                break;
            }

            default:
                self.postMessage({
                    type: 'ERROR',
                    error: `Unknown message type: ${type}`,
                } as WorkerOutputMessage);
        }
    } catch (error) {
        self.postMessage({
            type: 'ERROR',
            error: error instanceof Error ? error.message : 'Unknown error',
        } as WorkerOutputMessage);
    }
};

// 通知主线程 Worker 已就绪
self.postMessage({ type: 'READY' });

export { };
