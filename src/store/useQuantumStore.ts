/**
 * 量子 AI 遥测状态管理
 * 
 * 用于存储和监控量子模型的健康状态
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { THRESHOLDS, type QuantumTelemetry, type SystemStatus } from '@/types/quantum';

interface QuantumState {
    // 遥测数据
    telemetry: QuantumTelemetry | null;

    // 派生状态
    systemStatus: SystemStatus;
    isBarrenPlateau: boolean;
    lastHeartbeat: number;
    isDeadManSwitch: boolean;

    // 历史数据 (用于图表)
    gradientHistory: number[];
    entropyHistory: number[];

    // Actions
    updateTelemetry: (telemetry: QuantumTelemetry) => void;
    updateHeartbeat: (timestamp: number) => void;
    checkDeadManSwitch: () => void;
    reset: () => void;
}

const MAX_HISTORY_LENGTH = 100;

export const useQuantumStore = create<QuantumState>()(
    subscribeWithSelector((set, get) => ({
        // 初始状态
        telemetry: null,
        systemStatus: 'OK',
        isBarrenPlateau: false,
        lastHeartbeat: Date.now(),
        isDeadManSwitch: false,
        gradientHistory: [],
        entropyHistory: [],

        updateTelemetry: (telemetry) => {
            const isBarrenPlateau = telemetry.gradientNorm < THRESHOLDS.BARREN_PLATEAU;

            // 计算系统状态
            let systemStatus: SystemStatus = 'OK';
            if (isBarrenPlateau) {
                systemStatus = 'CRITICAL';
            } else if (telemetry.latency > THRESHOLDS.LATENCY_CRITICAL) {
                systemStatus = 'CRITICAL';
            } else if (telemetry.latency > THRESHOLDS.LATENCY_WARNING) {
                systemStatus = 'WARNING';
            }

            set((state) => ({
                telemetry,
                isBarrenPlateau,
                systemStatus,
                lastHeartbeat: Date.now(),
                isDeadManSwitch: false,
                // 更新历史 (保持最大长度)
                gradientHistory: [
                    ...state.gradientHistory.slice(-(MAX_HISTORY_LENGTH - 1)),
                    telemetry.gradientNorm,
                ],
                entropyHistory: [
                    ...state.entropyHistory.slice(-(MAX_HISTORY_LENGTH - 1)),
                    telemetry.entropy,
                ],
            }));
        },

        updateHeartbeat: (timestamp) => {
            set({
                lastHeartbeat: timestamp,
                isDeadManSwitch: false,
            });
        },

        checkDeadManSwitch: () => {
            const { lastHeartbeat } = get();
            const now = Date.now();
            const elapsed = now - lastHeartbeat;

            if (elapsed > THRESHOLDS.HEARTBEAT_TIMEOUT) {
                set({
                    isDeadManSwitch: true,
                    systemStatus: 'CRITICAL',
                });
            }
        },

        reset: () => {
            set({
                telemetry: null,
                systemStatus: 'OK',
                isBarrenPlateau: false,
                lastHeartbeat: Date.now(),
                isDeadManSwitch: false,
                gradientHistory: [],
                entropyHistory: [],
            });
        },
    }))
);

// 死人开关检查定时器 (每秒检查一次)
if (typeof window !== 'undefined') {
    setInterval(() => {
        useQuantumStore.getState().checkDeadManSwitch();
    }, 1000);
}
