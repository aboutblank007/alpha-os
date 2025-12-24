"use client";

// [Ref: 交易系统前端功能设计.MD] 5.1 梯度范数与贫瘠高原预警
// Visualizes the Gradient Norm (L2) of the Quantum Neural Network to detect Barren Plateaus

import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useQuantumStore } from '@/store/useQuantumStore';
import { THRESHOLDS } from '@/types/quantum';
import { cn } from '@/lib/utils';

interface GradientNormChartProps {
    className?: string;
    height?: number;
}

export function GradientNormChart({ className, height = 200 }: GradientNormChartProps) {
    const gradientHistory = useQuantumStore(state => state.gradientHistory);
    const isBarrenPlateau = useQuantumStore(state => state.isBarrenPlateau);

    // Format data for Recharts
    const data = gradientHistory.map((val, idx) => ({
        step: idx,
        value: val
    }));

    // Calculate dynamic domain
    const minVal = Math.min(...gradientHistory, 0);
    const maxVal = Math.max(...gradientHistory, 0.01);

    return (
        <div className={cn("w-full relative", className)} style={{ height }}>
            {/* Status Indicator */}
            <div className="absolute top-0 right-0 z-10 flex items-center gap-2">
                <span className={cn(
                    "text-[10px] px-1.5 py-0.5 rounded font-bold uppercase",
                    isBarrenPlateau ? "bg-red-500/20 text-red-500 animate-pulse" : "bg-green-500/10 text-green-500"
                )}>
                    {isBarrenPlateau ? "BARREN PLATEAU DETECTED" : "GRADIENT HEALTHY"}
                </span>
            </div>

            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={isBarrenPlateau ? "#ef4444" : "#10b981"} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={isBarrenPlateau ? "#ef4444" : "#10b981"} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <XAxis 
                        dataKey="step" 
                        hide 
                    />
                    <YAxis 
                        hide 
                        domain={[0, 'auto']} 
                    />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', fontSize: '12px' }}
                        itemStyle={{ color: '#e4e4e7' }}
                        formatter={(value: number) => [value.toExponential(4), "Gradient L2"]}
                        labelFormatter={() => ""}
                    />
                    <ReferenceLine y={THRESHOLDS.BARREN_PLATEAU} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'right', value: 'Threshold', fill: '#ef4444', fontSize: 10 }} />
                    <Area 
                        type="monotone" 
                        dataKey="value" 
                        stroke={isBarrenPlateau ? "#ef4444" : "#10b981"} 
                        fill="url(#colorGradient)" 
                        strokeWidth={2}
                        isAnimationActive={false} // Disable animation for high frequency updates
                    />
                </AreaChart>
            </ResponsiveContainer>
            
            <div className="absolute bottom-1 left-2 text-[10px] text-text-muted font-mono">
                Current L2: {gradientHistory[gradientHistory.length - 1]?.toExponential(4) || "0.00e+0"}
            </div>
        </div>
    );
}
