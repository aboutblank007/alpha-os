import { useEffect, useRef } from 'react';
import { useMarketStore } from '@/store/useMarketStore';
import { useTradeStore } from '@/store/useTradeStore';

export function useBridgeSync(pollInterval = 1000) {
  const { setConnectionStatus, updateMarketData, setLastUpdate } = useMarketStore();
  const { updateAccountInfo, updatePositions } = useTradeStore();

  const pollRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      const start = Date.now();
      try {
        const res = await fetch('/api/bridge/status');
        const data = await res.json();
        const end = Date.now();

        // Update Market Store
        setConnectionStatus(data.bridge_status === 'connected', end - start);
        if (data.active_symbols) {
          const period = data.last_mt5_update?.period;
          updateMarketData(data.active_symbols, data.symbol_prices || {}, period);
        }
        setLastUpdate(new Date());

        // Update Trade Store
        if (data.last_mt5_update) {
          if (data.last_mt5_update.account) {
            updateAccountInfo(data.last_mt5_update.account);
          }
          if (data.last_mt5_update.positions) {
            updatePositions(data.last_mt5_update.positions);
          }
        }

      } catch (error) {
        console.error('Bridge status sync failed:', error);
        setConnectionStatus(false, null);
      }
    };

    fetchStatus(); // Initial fetch
    pollRef.current = setInterval(fetchStatus, pollInterval);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [pollInterval, setConnectionStatus, updateMarketData, setLastUpdate, updateAccountInfo, updatePositions]);
}

