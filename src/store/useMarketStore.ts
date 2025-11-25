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
  
  // Actions
  setConnectionStatus: (isConnected: boolean, latency: number | null) => void;
  updateMarketData: (activeSymbols: string[], symbolPrices: Record<string, MarketPrice>) => void;
  setLastUpdate: (date: Date) => void;
}

export const useMarketStore = create<MarketStore>((set) => ({
  isConnected: false,
  latency: null,
  lastUpdate: null,
  activeSymbols: [],
  symbolPrices: {},

  setConnectionStatus: (isConnected, latency) => set({ isConnected, latency }),
  
  updateMarketData: (activeSymbols, symbolPrices) => set((state) => {
    // Optimistic update or merge if needed, but simple replacement is usually fine for snapshot data
    // We could merge symbolPrices if we wanted to keep stale prices for other symbols
    return { 
      activeSymbols, 
      symbolPrices: { ...state.symbolPrices, ...symbolPrices } 
    };
  }),
  
  setLastUpdate: (date) => set({ lastUpdate: date }),
}));

