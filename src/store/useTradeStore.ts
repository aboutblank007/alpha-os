import { create } from 'zustand';
import { Trade } from '@/lib/supabase';

export interface MT5Account {
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
}

export interface MT5Position {
  ticket: number;
  symbol: string;
  type: 'BUY' | 'SELL';
  volume: number;
  open_price: number;
  current_price: number;
  sl: number;
  tp: number;
  pnl: number;
  swap: number;
  comment?: string;
}

interface TradeStore {
  // MT5 Realtime Data
  account: MT5Account | null;
  positions: MT5Position[];
  
  // Supabase Data
  historyTrades: Trade[];
  isLoadingHistory: boolean;
  
  // Actions
  updateAccountInfo: (account: MT5Account) => void;
  updatePositions: (positions: MT5Position[]) => void;
  setHistoryTrades: (trades: Trade[]) => void;
  setLoadingHistory: (loading: boolean) => void;
  addHistoryTrade: (trade: Trade) => void;
}

export const useTradeStore = create<TradeStore>((set) => ({
  account: null,
  positions: [],
  historyTrades: [],
  isLoadingHistory: false,

  updateAccountInfo: (account) => set({ account }),
  updatePositions: (positions) => set({ positions }),
  
  setHistoryTrades: (trades) => set({ historyTrades: trades }),
  setLoadingHistory: (loading) => set({ isLoadingHistory: loading }),
  
  addHistoryTrade: (trade) => set((state) => ({ 
    historyTrades: [trade, ...state.historyTrades] 
  })),
}));

