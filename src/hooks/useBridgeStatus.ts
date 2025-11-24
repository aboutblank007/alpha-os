import { useState, useEffect, useRef } from 'react';

export interface BridgeStatus {
    bridge_status: 'connected' | 'disconnected';
    last_mt5_update: any;
    active_symbols: string[];
    symbol_prices: Record<string, any>;
    pending_commands: number;
    last_trade: any;
    positions?: any[];
    account?: {
        balance: number;
        equity: number;
        margin: number;
        free_margin: number;
    };
}

export function useBridgeStatus(pollInterval = 1000) {
    const [status, setStatus] = useState<BridgeStatus | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [latency, setLatency] = useState<number | null>(null);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

    const pollRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        const fetchStatus = async () => {
            const start = Date.now();
            try {
                const res = await fetch('/api/bridge/status');
                const data = await res.json();
                const end = Date.now();

                setLatency(end - start);
                setStatus(data);
                setIsConnected(data.bridge_status === 'connected');
                setLastUpdate(new Date());
            } catch (error) {
                console.error('Bridge status poll failed:', error);
                setIsConnected(false);
                setLatency(null);
            }
        };

        fetchStatus(); // Initial fetch
        pollRef.current = setInterval(fetchStatus, pollInterval);

        return () => {
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, [pollInterval]);

    return {
        status,
        isConnected,
        latency,
        lastUpdate,
        activeSymbols: status?.active_symbols || [],
        symbolPrices: status?.symbol_prices || {},
    };
}
