/**
 * 系统生命体征监控组件
 * 
 * 显示延迟热力带、死人开关倒计时、核心负载状态
 */

'use client';

import { useEffect, useRef } from 'react';
import { useQuantumStore } from '@/store/useQuantumStore';
import { useMarketStore } from '@/store/useMarketStore';
import { THRESHOLDS } from '@/types/quantum';
import { cn } from '@/lib/utils';
import { Activity, Cpu, AlertTriangle, Heart } from 'lucide-react';

interface SystemVitalsProps {
    className?: string;
}

export function SystemVitals({ className }: SystemVitalsProps) {
    const deadManRef = useRef<HTMLSpanElement>(null);
    const statusRef = useRef<HTMLDivElement>(null);

    // Store 状态
    const systemStatus = useQuantumStore((s) => s.systemStatus);
    const isDeadManSwitch = useQuantumStore((s) => s.isDeadManSwitch);
    const isBarrenPlateau = useQuantumStore((s) => s.isBarrenPlateau);
    const lastHeartbeat = useQuantumStore((s) => s.lastHeartbeat);
    const telemetry = useQuantumStore((s) => s.telemetry);
    const isConnected = useMarketStore((s) => s.isConnected);
    const latency = useMarketStore((s) => s.latency);

    // 死人开关倒计时
    useEffect(() => {
        const interval = setInterval(() => {
            if (deadManRef.current) {
                const elapsed = Date.now() - lastHeartbeat;
                const remaining = Math.max(0, THRESHOLDS.HEARTBEAT_TIMEOUT - elapsed);
                deadManRef.current.innerText = `${(remaining / 1000).toFixed(1)}s`;

                // 根据剩余时间变色
                if (remaining < 1000) {
                    deadManRef.current.style.color = '#ef4444';
                } else if (remaining < 2500) {
                    deadManRef.current.style.color = '#f59e0b';
                } else {
                    deadManRef.current.style.color = '#22c55e';
                }
            }
        }, 100);

        return () => clearInterval(interval);
    }, [lastHeartbeat]);

    // 延迟热力带颜色
    const getLatencyColor = () => {
        if (latency > THRESHOLDS.LATENCY_CRITICAL) return 'bg-red-500';
        if (latency > THRESHOLDS.LATENCY_WARNING) return 'bg-amber-500';
        return 'bg-green-500';
    };

    // 延迟热力带宽度
    const getLatencyWidth = () => {
        const maxLatency = 200;
        return Math.min(100, (latency / maxLatency) * 100);
    };

    return (
        <div className={cn('flex flex-col gap-3', className)}>
            {/* 系统状态指示器 */}
            <div
                ref={statusRef}
                className={cn(
                    'flex items-center gap-2 px-3 py-2 rounded-lg border transition-colors',
                    systemStatus === 'OK' && 'bg-green-500/10 border-green-500/30',
                    systemStatus === 'WARNING' && 'bg-amber-500/10 border-amber-500/30',
                    systemStatus === 'CRITICAL' && 'bg-red-500/10 border-red-500/30 animate-pulse'
                )}
            >
                <Activity
                    size={16}
                    className={cn(
                        systemStatus === 'OK' && 'text-green-500',
                        systemStatus === 'WARNING' && 'text-amber-500',
                        systemStatus === 'CRITICAL' && 'text-red-500'
                    )}
                />
                <span className="text-xs font-bold uppercase tracking-wider text-text-secondary">
                    {systemStatus === 'OK' && '系统正常'}
                    {systemStatus === 'WARNING' && '系统警告'}
                    {systemStatus === 'CRITICAL' && '系统危急'}
                </span>
            </div>

            {/* 延迟热力带 */}
            <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                    <span className="text-text-muted">延迟</span>
                    <span className={cn('font-mono font-bold', latency > THRESHOLDS.LATENCY_CRITICAL ? 'text-red-500' : 'text-text-primary')}>
                        {latency}ms
                    </span>
                </div>
                <div className="h-2 bg-bg-subtle rounded-full overflow-hidden">
                    <div
                        className={cn('h-full transition-all duration-300', getLatencyColor())}
                        style={{ width: `${getLatencyWidth()}%` }}
                    />
                </div>
            </div>

            {/* 死人开关倒计时 */}
            <div className="flex items-center justify-between px-3 py-2 bg-bg-subtle rounded-lg">
                <div className="flex items-center gap-2">
                    <Heart size={14} className={cn(isDeadManSwitch ? 'text-red-500' : 'text-green-500', 'animate-pulse')} />
                    <span className="text-xs text-text-muted">心跳倒计时</span>
                </div>
                <span ref={deadManRef} className="font-mono text-sm font-bold">
                    5.0s
                </span>
            </div>

            {/* 核心负载 */}
            {telemetry && (
                <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center gap-2 px-3 py-2 bg-bg-subtle rounded-lg">
                        <Cpu size={14} className="text-primary" />
                        <div className="flex-1">
                            <div className="text-[10px] text-text-muted uppercase">P-Core</div>
                            <div className="font-mono text-xs font-bold">{telemetry.pCoreLoad.toFixed(1)}%</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-2 bg-bg-subtle rounded-lg">
                        <Cpu size={14} className="text-text-muted" />
                        <div className="flex-1">
                            <div className="text-[10px] text-text-muted uppercase">E-Core</div>
                            <div className="font-mono text-xs font-bold">{telemetry.eCoreLoad.toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
            )}

            {/* 警报 */}
            {isBarrenPlateau && (
                <div className="flex items-center gap-2 px-3 py-2 bg-red-500/20 border border-red-500/50 rounded-lg animate-pulse">
                    <AlertTriangle size={14} className="text-red-500" />
                    <span className="text-xs font-bold text-red-400">
                        贫瘠高原: 梯度消失
                    </span>
                </div>
            )}

            {isDeadManSwitch && (
                <div className="flex items-center gap-2 px-3 py-2 bg-red-500/20 border border-red-500/50 rounded-lg animate-pulse">
                    <AlertTriangle size={14} className="text-red-500" />
                    <span className="text-xs font-bold text-red-400">
                        心跳超时: 紧急平仓已触发
                    </span>
                </div>
            )}

            {/* 连接状态 */}
            <div className="flex items-center gap-2 text-xs text-text-muted">
                <div className={cn('w-2 h-2 rounded-full', isConnected ? 'bg-green-500' : 'bg-red-500')} />
                <span>{isConnected ? '已连接' : '已断开'}</span>
            </div>
        </div>
    );
}
