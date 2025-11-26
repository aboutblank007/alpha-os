import { create } from 'zustand';

export interface Signal {
    id: string;
    created_at: string;
    symbol: string;
    action: 'BUY' | 'SELL';
    price: number;
    sl: number;
    tp: number;
    status: string;
    source: string;
    comment?: string;
}

interface SignalStore {
    signals: Signal[];
    isHistoryOpen: boolean;
    unreadCount: number;
    
    setSignals: (signals: Signal[]) => void;
    addSignal: (signal: Signal) => void;
    toggleHistory: () => void;
    setHistoryOpen: (open: boolean) => void;
    markAllAsRead: () => void;
    clearHistory: () => void;
}

export const useSignalStore = create<SignalStore>((set) => ({
    signals: [],
    isHistoryOpen: false,
    unreadCount: 0,

    setSignals: (signals) => set({ signals, unreadCount: signals.length }), // Initially all loaded might be considered "read" or "unread" depending on logic. Let's assume setSignals is for history load.
    
    addSignal: (signal) => set((state) => ({ 
        signals: [signal, ...state.signals],
        unreadCount: state.unreadCount + 1
    })),
    
    toggleHistory: () => set((state) => ({ isHistoryOpen: !state.isHistoryOpen })),
    
    setHistoryOpen: (open) => set({ isHistoryOpen: open }),
    
    markAllAsRead: () => set({ unreadCount: 0 }),
    
    clearHistory: () => set({ signals: [], unreadCount: 0 })
}));

