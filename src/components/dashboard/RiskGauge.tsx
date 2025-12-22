/**
 * 凯利公式风险仪表组件
 * 
 * 可视化当前杠杆 vs 最优凯利杠杆
 */

'use client';

import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { AlertTriangle, TrendingUp } from 'lucide-react';

interface RiskGaugeProps {
    /** 当前杠杆 */
    currentLeverage: number;
    /** 胜率 */
    winProbability: number;
    /** 赔率 (平均盈利/平均亏损) */
    odds: number;
    /** 当前波动率 */
    volatility?: number;
    className?: string;
}

export function RiskGauge({
    currentLeverage,
    winProbability,
    odds,
    volatility,
    className,
}: RiskGaugeProps) {
    // 计算凯利比例: f* = p - (1-p)/b
    const kellyFraction = useMemo(() => {
        if (odds <= 0) return 0;
        const kelly = winProbability - (1 - winProbability) / odds;
        return Math.max(0, Math.min(1, kelly)); // 限制在 0-1
    }, [winProbability, odds]);

    // 凯利建议的最大杠杆
    const kellyLeverage = useMemo(() => {
        // 简化计算: Kelly% * 基础杠杆因子
        const baseLeverage = 10; // 基础杠杆因子
        return kellyFraction * baseLeverage;
    }, [kellyFraction]);

    // 风险状态
    const isOverLeveraged = currentLeverage > kellyLeverage;
    const leverageRatio = kellyLeverage > 0 ? currentLeverage / kellyLeverage : 0;

    // 仪表盘角度 (0-180度)
    const gaugeAngle = useMemo(() => {
        const maxRatio = 2; // 2x 凯利杠杆为最大
        const ratio = Math.min(leverageRatio, maxRatio);
        return (ratio / maxRatio) * 180;
    }, [leverageRatio]);

    // 颜色区域
    const getZoneColor = () => {
        if (leverageRatio <= 0.5) return 'text-green-500';
        if (leverageRatio <= 1.0) return 'text-amber-500';
        return 'text-red-500';
    };

    return (
        <div className={cn('flex flex-col items-center gap-4 p-4', className)}>
            {/* 半圆仪表盘 */}
            <div className="relative w-40 h-20 overflow-hidden">
                {/* 背景弧 */}
                <div
                    className="absolute inset-0 rounded-t-full"
                    style={{
                        background: 'conic-gradient(from 180deg, #22c55e 0deg, #22c55e 45deg, #f59e0b 45deg, #f59e0b 90deg, #ef4444 90deg, #ef4444 180deg)',
                        clipPath: 'polygon(0% 100%, 100% 100%, 100% 0%, 0% 0%)',
                    }}
                />

                {/* 内部遮罩 */}
                <div
                    className="absolute bg-bg-card rounded-t-full"
                    style={{
                        top: '25%',
                        left: '12.5%',
                        width: '75%',
                        height: '75%',
                    }}
                />

                {/* 指针 */}
                <div
                    className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-500"
                    style={{
                        transform: `translateX(-50%) rotate(${gaugeAngle - 90}deg)`,
                        width: '2px',
                        height: '90%',
                        background: 'linear-gradient(to top, #3b82f6, transparent)',
                    }}
                />

                {/* 中心点 */}
                <div className="absolute bottom-0 left-1/2 w-3 h-3 -translate-x-1/2 translate-y-1/2 rounded-full bg-primary shadow-lg" />
            </div>

            {/* 数值显示 */}
            <div className="flex flex-col items-center gap-1">
                <div className={cn('flex items-center gap-2', getZoneColor())}>
                    {isOverLeveraged && <AlertTriangle size={16} className="animate-pulse" />}
                    <span className="text-2xl font-bold font-mono">
                        {currentLeverage.toFixed(2)}x
                    </span>
                </div>
                <div className="text-xs text-text-muted">
                    凯利建议: <span className="font-mono font-bold text-primary">{kellyLeverage.toFixed(2)}x</span>
                </div>
            </div>

            {/* 凯利公式详情 */}
            <div className="w-full grid grid-cols-3 gap-2 text-center">
                <div className="flex flex-col">
                    <span className="text-[10px] text-text-muted uppercase">胜率</span>
                    <span className="font-mono text-xs font-bold">{(winProbability * 100).toFixed(0)}%</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-[10px] text-text-muted uppercase">赔率</span>
                    <span className="font-mono text-xs font-bold">{odds.toFixed(2)}</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-[10px] text-text-muted uppercase">凯利%</span>
                    <span className="font-mono text-xs font-bold">{(kellyFraction * 100).toFixed(1)}%</span>
                </div>
            </div>

            {/* 波动率 */}
            {volatility !== undefined && (
                <div className="flex items-center gap-2 text-xs text-text-muted">
                    <TrendingUp size={12} />
                    <span>波动率: <span className="font-mono font-bold">{(volatility * 100).toFixed(2)}%</span></span>
                </div>
            )}

            {/* 警告 */}
            {isOverLeveraged && (
                <div className="w-full px-3 py-2 bg-red-500/20 border border-red-500/50 rounded-lg animate-pulse">
                    <div className="flex items-center gap-2 text-red-400">
                        <AlertTriangle size={14} />
                        <span className="text-xs font-bold">杠杆过高! 降低仓位</span>
                    </div>
                </div>
            )}
        </div>
    );
}
