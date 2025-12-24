"use client";

// [Ref: 交易系统前端功能设计.MD] 4.1 订单簿热力图与流动性墙
// Implements the limit order book heatmap visualization using HTML5 Canvas
// Optimized for M2 Pro GPU rasterization via browser compositor

import React, { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface OrderBookHeatmapProps {
    symbol: string;
    className?: string;
    height?: number;
}

export function OrderBookHeatmap({ symbol, className, height = 300 }: OrderBookHeatmapProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    
    // 模拟 LOB 数据流
    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const ctx = canvas.getContext('2d', { alpha: false }); // Optimize for speed
        if (!ctx) return;

        let animationFrameId: number;
        
        // 初始化尺寸
        const resize = () => {
            const { width } = container.getBoundingClientRect();
            // Retina display support
            const dpr = window.devicePixelRatio || 1;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
            ctx.scale(dpr, dpr);
        };
        
        resize();
        window.addEventListener('resize', resize);

        // 模拟数据生成
        const priceLevels = 100;
        const timeSteps = 200; // X axis resolution
        
        // 预生成噪声图 (Perlin-like noise simulation for liquidity walls)
        const generateHeatmapColumn = (t: number) => {
            const col = new Float32Array(priceLevels);
            for (let i = 0; i < priceLevels; i++) {
                // Create "Liquidity Walls" using sine waves
                const wall1 = Math.sin(i * 0.1 + t * 0.02) * 0.5 + 0.5;
                const wall2 = Math.sin(i * 0.05 - t * 0.01) * 0.5 + 0.5;
                // Random noise (Retail flow)
                const noise = Math.random() * 0.2;
                
                col[i] = (wall1 * wall2) + noise;
            }
            return col;
        };

        let offset = 0;
        
        const render = () => {
            offset += 1;
            const w = canvas.width / window.devicePixelRatio;
            const h = height;
            const cellW = w / timeSteps;
            const cellH = h / priceLevels;

            // Clear background
            ctx.fillStyle = '#09090b'; // bg-zinc-950
            ctx.fillRect(0, 0, w, h);

            for (let x = 0; x < timeSteps; x++) {
                const colData = generateHeatmapColumn(offset + x);
                for (let y = 0; y < priceLevels; y++) {
                    const value = colData[y];
                    
                    // Color mapping (Cold -> Hot)
                    // Blue (Low liquidity) -> Red/Orange (High liquidity wall)
                    let color = '';
                    if (value < 0.3) {
                        color = `rgba(30, 64, 175, ${value})`; // Blue
                    } else if (value < 0.6) {
                        color = `rgba(59, 130, 246, ${value})`; // Light Blue
                    } else if (value < 0.8) {
                        color = `rgba(245, 158, 11, ${value})`; // Amber
                    } else {
                        color = `rgba(239, 68, 68, ${value})`; // Red
                    }

                    ctx.fillStyle = color;
                    ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
                }
            }
            
            // Draw Price Labels (Mock)
            ctx.fillStyle = '#a1a1aa';
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.fillText((2000.50).toFixed(2), w - 5, 20);
            ctx.fillText((1995.00).toFixed(2), w - 5, h - 10);
            
            // Draw "Liquidity Break" annotation if needed
            if (offset % 500 === 0) {
                // Mock event
            }

            animationFrameId = requestAnimationFrame(render);
        };

        render();

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationFrameId);
        };
    }, [height]);

    return (
        <div ref={containerRef} className={cn("relative w-full overflow-hidden rounded-lg border border-white/5 bg-black", className)}>
            <canvas ref={canvasRef} className="block" />
            
            {/* Overlay UI */}
            <div className="absolute top-2 left-2 px-2 py-1 bg-black/50 backdrop-blur rounded text-[10px] text-text-muted font-mono border border-white/10">
                LOB Heatmap · {symbol} · Live
            </div>
            
            <div className="absolute bottom-2 left-2 flex gap-2">
                 <div className="flex items-center gap-1 text-[9px] text-text-muted bg-black/50 px-1.5 rounded">
                    <span className="w-2 h-2 rounded-full bg-blue-800"></span> Low
                 </div>
                 <div className="flex items-center gap-1 text-[9px] text-text-muted bg-black/50 px-1.5 rounded">
                    <span className="w-2 h-2 rounded-full bg-red-600"></span> High
                 </div>
            </div>
        </div>
    );
}
