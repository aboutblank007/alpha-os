import React, { createContext, useContext, useReducer, useCallback, useEffect, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { alphaOSReducer, initialState } from '../store/reducer';
import { ActionTypes, AlphaOSState } from '../store/types';
import { RuntimeSnapshot } from '../shared/ws/types';

interface AlphaOSContextValue extends AlphaOSState {
    connectionState: string;
    reconnect: () => void;
    runtime: RuntimeSnapshot; // Alias for runtimeSnapshot for convenience
}

const AlphaOSContext = createContext<AlphaOSContextValue | null>(null);

export const AlphaOSProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [state, dispatch] = useReducer(alphaOSReducer, initialState);

    // 跟踪前一个状态值用于事件检测
    const prevBarsRef = useRef(0);
    const prevPositionsRef = useRef(0);
    const prevSignalRef = useRef('IDLE');
    const lastRuntimeTsRef = useRef<number | null>(null);

    // 处理 WebSocket 消息
    const handleMessage = useCallback((message: any) => {
        const { type, ...rest } = message;
        // 后端直接把字段放顶层，兼容 data 包装和直接字段两种格式
        const data = message.data || rest;

        switch (type) {
            case 'welcome':
                dispatch({ type: ActionTypes.SET_WS_CONNECTED, payload: true });
                break;

            case 'tick':
                dispatch({ type: ActionTypes.UPDATE_TICK, payload: data });
                break;

            case 'status':
                dispatch({ type: ActionTypes.UPDATE_STATUS, payload: data });
                // 检测窗口完成事件
                if (data.bars_completed && data.bars_completed > prevBarsRef.current) {
                    dispatch({
                        type: ActionTypes.TRIGGER_EVENT,
                        payload: { type: 'TICK_WINDOW', data: { bars: data.bars_completed } }
                    });
                    prevBarsRef.current = data.bars_completed;
                }
                break;

            case 'position':
                dispatch({ type: ActionTypes.UPDATE_POSITION, payload: data });
                // 检测持仓变化事件
                const positions =
                    Array.isArray(data) ? data :
                    Array.isArray(data?.positions) ? data.positions :
                    data?.direction ? [data] : [];
                const posCount = positions.length;
                if (posCount !== prevPositionsRef.current) {
                    const eventType = posCount > prevPositionsRef.current ? 'ORDER_FILLED' : 'ORDER_CLOSED';
                    dispatch({
                        type: ActionTypes.TRIGGER_EVENT,
                        payload: { type: eventType, data }
                    });
                    prevPositionsRef.current = posCount;
                }
                break;

            case 'decision':
                dispatch({ type: ActionTypes.UPDATE_DECISION, payload: data });
                // 检测信号变化事件
                if (data.signal && data.signal !== prevSignalRef.current) {
                    dispatch({
                        type: ActionTypes.TRIGGER_EVENT,
                        payload: { type: 'SIGNAL_CHANGE', data: { signal: data.signal } }
                    });
                    prevSignalRef.current = data.signal;
                }
                // 推理完成事件
                dispatch({
                    type: ActionTypes.TRIGGER_EVENT,
                    payload: { type: 'INFERENCE', data }
                });
                break;

            case 'dollar_bar':
                dispatch({ type: ActionTypes.UPDATE_DOLLAR_BAR, payload: data });
                break;

            case 'inference':
                dispatch({ type: ActionTypes.UPDATE_INFERENCE, payload: data });
                break;

            case 'order':
                dispatch({ type: ActionTypes.UPDATE_ORDER, payload: data });
                // 订单发送事件
                dispatch({
                    type: ActionTypes.TRIGGER_EVENT,
                    payload: { type: 'ORDER_SENT', data }
                });
                break;

            case 'runtime_snapshot':
                dispatch({ type: ActionTypes.UPDATE_RUNTIME_SNAPSHOT, payload: data });
                if (data?.timestamp) {
                    lastRuntimeTsRef.current = data.timestamp * 1000;
                }
                break;

            case 'system':
                dispatch({ type: ActionTypes.ADD_SYSTEM_MESSAGE, payload: data });
                break;

            case 'pong':
                // 心跳响应，忽略
                break;

            default:
                console.log('[AlphaOS] Unknown message type:', type, data);
        }
    }, []);

    // WebSocket 连接
    const { connectionState, reconnect } = useWebSocket(handleMessage);

    // 更新连接状态
    useEffect(() => {
        const isConnected = connectionState === 'connected';
        dispatch({
            type: ActionTypes.SET_WS_CONNECTED,
            payload: isConnected,
        });
        // 同步 status.connected，避免 UI 显示 DISCONNECTED
        dispatch({
            type: ActionTypes.UPDATE_STATUS,
            payload: { connected: isConnected },
        });
    }, [connectionState]);

    // Fallback: poll latest runtime snapshot when WS is down or stale
    useEffect(() => {
        let isMounted = true;
        const poll = async () => {
            try {
                const response = await fetch('/api/history/runtime?limit=1');
                if (!response.ok) return;
                const rows = await response.json();
                if (!Array.isArray(rows) || rows.length === 0) return;
                const latest = rows[0];
                if (!isMounted) return;
                dispatch({ type: ActionTypes.UPDATE_RUNTIME_SNAPSHOT, payload: latest });
                if (latest?.timestamp) {
                    lastRuntimeTsRef.current = latest.timestamp * 1000;
                }
            } catch {
                // ignore
            }
        };

        const interval = setInterval(() => {
            const lastTs = lastRuntimeTsRef.current;
            const stale = lastTs ? Date.now() - lastTs > 10000 : true;
            if (connectionState !== 'connected' || stale) {
                poll();
            }
        }, 5000);

        return () => {
            isMounted = false;
            clearInterval(interval);
        };
    }, [connectionState]);

    const value: AlphaOSContextValue = {
        ...state,
        connectionState,
        reconnect,
        runtime: state.runtimeSnapshot, // Alias implementation
    };

    return (
        <AlphaOSContext.Provider value={value}>
            {children}
        </AlphaOSContext.Provider>
    );
};

export function useAlphaOS() {
    const context = useContext(AlphaOSContext);
    if (!context) {
        throw new Error('useAlphaOS must be used within an AlphaOSProvider');
    }
    return context;
}

export default AlphaOSContext;
