import { create } from 'zustand';

export interface MarketPrice {
  bid: number;
  ask: number;
  last_seen: number;
}

interface MarketStore {
  isConnected: boolean;
  latency: number | null;
  lastUpdate: Date | null;
  activeSymbols: string[];
  symbolPrices: Record<string, MarketPrice>;
  chartPeriod: string | null;

  // Actions
  setConnectionStatus: (isConnected: boolean, latency: number | null) => void;
  updateMarketData: (activeSymbols: string[], symbolPrices: Record<string, MarketPrice>, chartPeriod?: string) => void;
  setLastUpdate: (date: Date) => void;
}

export const useMarketStore = create<MarketStore>((set) => ({
  isConnected: false,
  latency: null,
  lastUpdate: null,
  activeSymbols: [],
  symbolPrices: {},
  chartPeriod: null,

  setConnectionStatus: (isConnected, latency) => set({ isConnected, latency }),

  updateMarketData: (activeSymbols, symbolPrices, chartPeriod) => set((state) => {
    // Optimistic update or merge if needed, but simple replacement is usually fine for snapshot data
    // We could merge symbolPrices if we wanted to keep stale prices for other symbols
    return {
      activeSymbols,
      symbolPrices: { ...state.symbolPrices, ...symbolPrices },
      ...(chartPeriod ? { chartPeriod } : {})
    };
  }),

  setLastUpdate: (date) => set({ lastUpdate: date }),
}));

