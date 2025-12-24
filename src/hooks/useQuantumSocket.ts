/**
 * Quantum WebSocket 连接 Hook
 * 
 * 连接 FastAPI WebSocket 网关，处理高频市场数据和量子遥测
 */

'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { useMarketStore } from '@/store/useMarketStore';
import { useQuantumStore } from '@/store/useQuantumStore';
import type { MarketTick, QuantumTelemetry, AccountState, WorkerOutputMessage, AILog, AIConfig } from '@/types/quantum';

// WebSocket 端点配置
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
const WS_ENDPOINTS = {
    MARKET_STREAM: `${WS_BASE_URL}/ws/market_stream`,
    STATE_SYNC: `${WS_BASE_URL}/ws/state_sync`,
    COMMANDS: `${WS_BASE_URL}/ws/commands`,
} as const;

interface UseQuantumSocketOptions {
    /** 是否自动连接 */
    autoConnect?: boolean;
    /** 重连延迟 (ms) */
    reconnectDelay?: number;
    /** 最大重连次数 */
    maxReconnects?: number;
}

interface QuantumSocketState {
    isConnected: boolean;
    isConnecting: boolean;
    error: string | null;
    latency: number;
    ticksPerSecond: number;
}

export function useQuantumSocket(options: UseQuantumSocketOptions = {}) {
    const {
        autoConnect = true,
        reconnectDelay = 3000,
        maxReconnects = 5,
    } = options;

    // Refs
    const marketWsRef = useRef<WebSocket | null>(null);
    const stateWsRef = useRef<WebSocket | null>(null);
    const commandWsRef = useRef<WebSocket | null>(null);
    const workerRef = useRef<Worker | null>(null);
    const reconnectCountRef = useRef(0);
    const tickCountRef = useRef(0);
    const lastTickTimeRef = useRef(Date.now());

    // State
    const [state, setState] = useState<QuantumSocketState>({
        isConnected: false,
        isConnecting: false,
        error: null,
        latency: 0,
        ticksPerSecond: 0,
    });

    // Store actions
    const setConnectionStatus = useMarketStore((s) => s.setConnectionStatus);
    const setTransientTick = useMarketStore((s) => s.setTransientTick);
    const updateTelemetry = useQuantumStore((s) => s.updateTelemetry);
    const addAILog = useQuantumStore((s) => s.addAILog);
    const updateAIConfig = useQuantumStore((s) => s.updateAIConfig);

    // 初始化 Web Worker
    useEffect(() => {
        if (typeof window === 'undefined') return;

        try {
            workerRef.current = new Worker(
                new URL('../workers/marketDataParser.ts', import.meta.url)
            );

            workerRef.current.onmessage = (event: MessageEvent<WorkerOutputMessage>) => {
                const { type, data } = event.data;

                if (type === 'TICK_PARSED' && data && !Array.isArray(data)) {
                    handleParsedTick(data);
                } else if (type === 'BATCH_PARSED' && Array.isArray(data)) {
                    data.forEach(handleParsedTick);
                }
            };
        } catch (error) {
            console.error('[QuantumSocket] Worker 初始化失败:', error);
        }

        return () => {
            workerRef.current?.terminate();
        };
    }, []);

    // 处理解析后的 Tick
    const handleParsedTick = useCallback((tick: MarketTick) => {
        tickCountRef.current++;
        setTransientTick(tick);

        // 计算 TPS (每秒更新一次)
        const now = Date.now();
        if (now - lastTickTimeRef.current >= 1000) {
            setState((prev) => ({
                ...prev,
                ticksPerSecond: tickCountRef.current,
                latency: now - tick.timestamp,
            }));
            tickCountRef.current = 0;
            lastTickTimeRef.current = now;
        }
    }, [setTransientTick]);

    // 连接 Market Stream
    const connectMarketStream = useCallback(() => {
        if (marketWsRef.current?.readyState === WebSocket.OPEN) return;

        setState((prev) => ({ ...prev, isConnecting: true }));

        const ws = new WebSocket(WS_ENDPOINTS.MARKET_STREAM);

        ws.onopen = () => {
            console.log('[QuantumSocket] Market Stream 已连接');
            reconnectCountRef.current = 0;
            setState((prev) => ({
                ...prev,
                isConnected: true,
                isConnecting: false,
                error: null,
            }));
            setConnectionStatus(true, 0);
        };

        ws.onmessage = (event) => {
            // 发送到 Worker 解析
            if (workerRef.current) {
                workerRef.current.postMessage({
                    type: 'PARSE_TICK',
                    data: event.data,
                });
            }
        };

        ws.onerror = (error) => {
            console.error('[QuantumSocket] Market Stream 错误:', error);
            setState((prev) => ({
                ...prev,
                error: 'WebSocket 连接错误',
            }));
        };

        ws.onclose = () => {
            console.log('[QuantumSocket] Market Stream 已断开');
            setState((prev) => ({
                ...prev,
                isConnected: false,
                isConnecting: false,
            }));
            setConnectionStatus(false, 0);

            // 自动重连
            if (reconnectCountRef.current < maxReconnects) {
                reconnectCountRef.current++;
                setTimeout(connectMarketStream, reconnectDelay);
            }
        };

        marketWsRef.current = ws;
    }, [maxReconnects, reconnectDelay, setConnectionStatus]);

    // 连接 State Sync
    const connectStateSync = useCallback(() => {
        if (stateWsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = new WebSocket(WS_ENDPOINTS.STATE_SYNC);

        ws.onopen = () => {
            console.log('[QuantumSocket] State Sync 已连接');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // 处理量子遥测
                if (data.type === 'TELEMETRY') {
                    updateTelemetry(data.payload as QuantumTelemetry);
                }
                // 处理 AI 日志
                else if (data.type === 'AI_LOG') {
                    addAILog(data.payload as AILog);
                }
                // 处理 AI 配置
                else if (data.type === 'AI_CONFIG') {
                    updateAIConfig(data.payload as AIConfig);
                }
                // 处理账户状态
                else if (data.type === 'ACCOUNT_STATE') {
                    // TODO: 更新账户状态 Store
                    console.log('[QuantumSocket] 账户状态:', data.payload);
                }
            } catch (error) {
                console.error('[QuantumSocket] State Sync 解析错误:', error);
            }
        };

        ws.onclose = () => {
            console.log('[QuantumSocket] State Sync 已断开');
            setTimeout(connectStateSync, reconnectDelay);
        };

        stateWsRef.current = ws;
    }, [reconnectDelay, updateTelemetry]);

    // 连接 Command Bus
    const connectCommandBus = useCallback(() => {
        if (commandWsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = new WebSocket(WS_ENDPOINTS.COMMANDS);

        ws.onopen = () => {
            console.log('[QuantumSocket] Command Bus 已连接');
        };

        ws.onclose = () => {
            console.log('[QuantumSocket] Command Bus 已断开');
            setTimeout(connectCommandBus, reconnectDelay);
        };

        commandWsRef.current = ws;
    }, [reconnectDelay]);

    // 发送命令
    const sendCommand = useCallback((command: Record<string, unknown>) => {
        if (commandWsRef.current?.readyState !== WebSocket.OPEN) {
            console.error('[QuantumSocket] Command Bus 未连接');
            return false;
        }

        commandWsRef.current.send(JSON.stringify(command));
        return true;
    }, []);

    // 紧急平仓
    const closeAllPositions = useCallback(() => {
        return sendCommand({ action: 'CLOSE_ALL' });
    }, [sendCommand]);

    // 自动连接
    useEffect(() => {
        if (autoConnect) {
            connectMarketStream();
            connectStateSync();
            connectCommandBus();
        }

        return () => {
            marketWsRef.current?.close();
            stateWsRef.current?.close();
            commandWsRef.current?.close();
        };
    }, [autoConnect, connectMarketStream, connectStateSync, connectCommandBus]);

    return {
        ...state,
        sendCommand,
        closeAllPositions,
        connect: () => {
            connectMarketStream();
            connectStateSync();
            connectCommandBus();
        },
        disconnect: () => {
            marketWsRef.current?.close();
            stateWsRef.current?.close();
            commandWsRef.current?.close();
        },
    };
}
