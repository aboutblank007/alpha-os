/**
 * 延迟显示组件 (瞬态 DOM 更新)
 * 
 * 使用直接 DOM 操作实现 60fps 更新，绕过 React 渲染周期
 */

'use client';

import { useEffect, useRef } from 'react';
import { useMarketStore } from '@/store/useMarketStore';
import { THRESHOLDS } from '@/types/quantum';
import { cn } from '@/lib/utils';

interface LatencyDisplayProps {
    className?: string;
    showLabel?: boolean;
}

export function LatencyDisplay({ className, showLabel = true }: LatencyDisplayProps) {
    const latencyRef = useRef<HTMLSpanElement>(null);
    const indicatorRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // 使用 Zustand subscribe 进行瞬态更新
        const unsub = useMarketStore.subscribe(
            (state) => state.latency,
            (latency) => {
                if (latencyRef.current) {
                    latencyRef.current.innerText = `${latency}ms`;
                }

                if (indicatorRef.current) {
                    // 根据延迟设置颜色
                    if (latency > THRESHOLDS.LATENCY_CRITICAL) {
                        indicatorRef.current.style.backgroundColor = '#ef4444'; // red
                        indicatorRef.current.classList.add('animate-pulse');
                    } else if (latency > THRESHOLDS.LATENCY_WARNING) {
                        indicatorRef.current.style.backgroundColor = '#f59e0b'; // amber
                        indicatorRef.current.classList.remove('animate-pulse');
                    } else {
                        indicatorRef.current.style.backgroundColor = '#22c55e'; // green
                        indicatorRef.current.classList.remove('animate-pulse');
                    }
                }
            }
        );

        return unsub;
    }, []);

    return (
        <div className={cn('flex items-center gap-2', className)}>
            <div
                ref={indicatorRef}
                className="w-2 h-2 rounded-full bg-green-500 transition-colors"
            />
            {showLabel && (
                <span className="text-xs text-text-muted uppercase tracking-wider">
                    延迟
                </span>
            )}
            <span
                ref={latencyRef}
                className="font-mono text-sm font-bold text-text-primary"
            >
                0ms
            </span>
        </div>
    );
}
