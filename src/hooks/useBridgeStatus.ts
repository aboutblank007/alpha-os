import { useMarketStore } from '@/store/useMarketStore';
import { useTradeStore } from '@/store/useTradeStore';
import { useShallow } from 'zustand/react/shallow';

// Re-export types for compatibility
export type { MT5Account, MT5Position } from '@/store/useTradeStore';

export interface BridgeStatus {
    bridge_status: 'connected' | 'disconnected';
    last_mt5_update: {
        account: import('@/store/useTradeStore').MT5Account | null;
        positions: import('@/store/useTradeStore').MT5Position[];
    };
    active_symbols: string[];
    symbol_prices: Record<string, { bid: number; ask: number; last_seen: number }>;
    pending_commands: number;
    last_trade: unknown;
}

export function useBridgeStatus() {
    // Note: pollInterval is ignored here as polling is handled by useBridgeSync in AppShell
    
    const { isConnected, latency, lastUpdate, activeSymbols, symbolPrices } = useMarketStore(
        useShallow(state => ({
            isConnected: state.isConnected,
            latency: state.latency,
            lastUpdate: state.lastUpdate,
            activeSymbols: state.activeSymbols,
            symbolPrices: state.symbolPrices
        }))
    );

    const { account, positions } = useTradeStore(
        useShallow(state => ({
            account: state.account,
            positions: state.positions
        }))
    );

    // Construct compatibility object
    const status: BridgeStatus = {
        bridge_status: isConnected ? 'connected' : 'disconnected',
        last_mt5_update: {
            account,
            positions
        },
        active_symbols: activeSymbols,
        symbol_prices: symbolPrices,
        pending_commands: 0, // Not currently tracked in store, can add if needed
        last_trade: null // Not currently tracked in store
    };

    return {
        status,
        isConnected,
        latency,
        lastUpdate,
        activeSymbols,
        symbolPrices,
    };
}
