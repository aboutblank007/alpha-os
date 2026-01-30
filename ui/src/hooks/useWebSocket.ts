import { useEffect, useRef, useCallback, useState } from 'react';

// 动态获取 WebSocket URL，支持局域网访问与 HTTPS
const getWsUrl = () => {
    const host = window.location.hostname || 'localhost';
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${host}:8765`;
};
const MAX_RECONNECT_DELAY = 30000; // 最大重连延迟 30 秒
const INITIAL_RECONNECT_DELAY = 1000; // 初始重连延迟 1 秒

type ConnectionState = 'connected' | 'connecting' | 'disconnected' | 'error';

export function useWebSocket(onMessage: (message: any) => void) {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);
    const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setConnectionState('connecting');
        
        try {
            const wsUrl = getWsUrl();
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[WS] Connected to AlphaOS');
                setConnectionState('connected');
                reconnectDelayRef.current = INITIAL_RECONNECT_DELAY; // 重置重连延迟
            };

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    onMessage?.(message);
                } catch (e) {
                    console.warn('[WS] Failed to parse message:', e);
                }
            };

            ws.onclose = (event) => {
                console.log('[WS] Disconnected:', event.code, event.reason);
                setConnectionState('disconnected');
                scheduleReconnect();
            };

            ws.onerror = (error) => {
                console.error('[WS] Error:', error);
                setConnectionState('error');
            };
        } catch (error) {
            console.error('[WS] Failed to create connection:', error);
            setConnectionState('error');
            scheduleReconnect();
        }
    }, [onMessage]);

    const scheduleReconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        const delay = reconnectDelayRef.current;
        console.log(`[WS] Reconnecting in ${delay}ms...`);

        reconnectTimeoutRef.current = setTimeout(() => {
            // 指数退避
            reconnectDelayRef.current = Math.min(
                reconnectDelayRef.current * 2,
                MAX_RECONNECT_DELAY
            );
            connect();
        }, delay);
    }, [connect]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setConnectionState('disconnected');
    }, []);

    const sendPing = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
        }
    }, []);

    // 初始连接
    useEffect(() => {
        connect();
        
        // 心跳检测
        const pingInterval = setInterval(sendPing, 25000);

        return () => {
            clearInterval(pingInterval);
            disconnect();
        };
    }, [connect, disconnect, sendPing]);

    return {
        connectionState,
        reconnect: connect,
        disconnect,
    };
}

export default useWebSocket;
