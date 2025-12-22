import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { MarketTick } from '@/types/quantum';

export interface MarketPrice {
  bid: number;
  ask: number;
  last_seen: number;
}

interface MarketStore {
  // 连接状态
  isConnected: boolean;
  latency: number;
  lastUpdate: Date | null;

  // 市场数据
  activeSymbols: string[];
  symbolPrices: Record<string, MarketPrice>;
  chartPeriod: string | null;

  // 高频瞬态数据 (用于直接 DOM 更新，不触发 React 渲染)
  lastTick: MarketTick | null;
  tickCount: number;

  // Actions
  setConnectionStatus: (isConnected: boolean, latency: number) => void;
  updateMarketData: (activeSymbols: string[], symbolPrices: Record<string, MarketPrice>, chartPeriod?: string) => void;
  setLastUpdate: (date: Date) => void;

  // 高频更新 (瞬态)
  setTransientTick: (tick: MarketTick) => void;
  incrementTickCount: () => void;
}

export const useMarketStore = create<MarketStore>()(
  subscribeWithSelector((set) => ({
    // 初始状态
    isConnected: false,
    latency: 0,
    lastUpdate: null,
    activeSymbols: [],
    symbolPrices: {},
    chartPeriod: null,
    lastTick: null,
    tickCount: 0,

    setConnectionStatus: (isConnected, latency) => set({ isConnected, latency }),

    updateMarketData: (activeSymbols, symbolPrices, chartPeriod) => set((state) => ({
      activeSymbols,
      symbolPrices: { ...state.symbolPrices, ...symbolPrices },
      ...(chartPeriod ? { chartPeriod } : {}),
    })),

    setLastUpdate: (date) => set({ lastUpdate: date }),

    // 高频瞬态更新
    setTransientTick: (tick) => set({
      lastTick: tick,
      latency: Date.now() - tick.timestamp,
    }),

    incrementTickCount: () => set((state) => ({
      tickCount: state.tickCount + 1,
    })),
  }))
);

// 导出类型以便外部使用 subscribe
export type { MarketStore };
