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
        // 注意：/api/bridge/status 在断连时也会返回 200，但会携带 bridge_status=disconnected。
        // 不能仅凭 fetch 成功就认为连接正常，否则会出现 "Waiting for EA data..." 的误导提示。
        const bridgeConnected = data?.bridge_status ? data.bridge_status !== 'disconnected' : true;
        setConnectionStatus(!!bridgeConnected, end - start);
        if (bridgeConnected && data.active_symbols) {
          const period = data.last_mt5_update?.period;
          updateMarketData(data.active_symbols, data.symbol_prices || {}, period);
        }
        setLastUpdate(new Date());

        // Update Trade Store
        // The API returns a flat object (account, positions, symbols...)
        // But the error mock returns nested last_mt5_update. We should handle both.
        const payload = (data.last_mt5_update && typeof data.last_mt5_update === "object") ? data.last_mt5_update : data;

        if (bridgeConnected && payload.account) {
          updateAccountInfo(payload.account);
        }
        if (bridgeConnected && payload.positions) {
          updatePositions(payload.positions);
        }

      } catch (error) {
        console.error('Bridge status sync failed:', error);
        setConnectionStatus(false, 0);
      }
    };

    fetchStatus(); // Initial fetch
    pollRef.current = setInterval(fetchStatus, pollInterval);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [pollInterval, setConnectionStatus, updateMarketData, setLastUpdate, updateAccountInfo, updatePositions]);
}
