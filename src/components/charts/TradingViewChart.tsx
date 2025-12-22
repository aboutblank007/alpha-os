"use client";

import React, { useEffect, useRef, memo, useState } from 'react';
import { createChart, ColorType, IChartApi, CandlestickSeries, CandlestickData, Time } from 'lightweight-charts';
import { Loader2, AlertCircle } from 'lucide-react';

interface ChartProps {
    symbol: string;
}

const ChartComponent = ({ symbol }: ChartProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<any>(null); // Keep reference to update data
    
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch Data Function
    const fetchHistory = async () => {
        try {
            setError(null);
            const response = await fetch('/api/prices', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instrument: symbol,
                    granularity: 'M1',
                    count: 500 // 500 bars
                })
            });

            if (!response.ok) {
                if (response.status === 503) {
                    throw new Error("Bridge Disconnected");
                }
                throw new Error("Failed to fetch history");
            }

            const result = await response.json();
            
            if (result.error) {
                 throw new Error(result.error);
            }

            if (result.candles && Array.isArray(result.candles)) {
                // Map API format to Lightweight Charts format
                // API: { time (seconds), open, high, low, close }
                // LWC: { time (Time), open, high, low, close }
                const data: CandlestickData[] = result.candles.map((c: any) => ({
                    time: c.time as Time,
                    open: c.open,
                    high: c.high,
                    low: c.low,
                    close: c.close
                })).sort((a: CandlestickData, b: CandlestickData) => (a.time as number) - (b.time as number)); // Ensure sorted

                // Filter out duplicates if any
                const uniqueData = data.filter((item, index, self) =>
                    index === self.findIndex((t) => t.time === item.time)
                );

                if (uniqueData.length === 0) {
                     setError("暂无数据");
                } else {
                if (seriesRef.current) {
                    seriesRef.current.setData(uniqueData);
                    }
                }
            } else {
                 setError("暂无数据");
            }
        } catch (e: any) {
            console.warn(`Chart fetch failed for ${symbol}:`, e);
            if (e.message === "Bridge Disconnected" || e.message === "MT5 Connection Failed") {
                 setError("交易桥未连接");
            } else {
                 setError("数据加载失败");
            }
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (!containerRef.current) return;
        
        setIsLoading(true);

        // Init Chart - Quantum Theme Colors
        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#94a3b8', // Slate-400
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
            },
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
            autoSize: true,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: 'rgba(255, 255, 255, 0.1)',
            },
            rightPriceScale: {
                borderColor: 'rgba(255, 255, 255, 0.1)',
            }
        });

        // Add Candle Series
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10b981', // Emerald-500
            downColor: '#ef4444', // Red-500
            borderVisible: false,
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        } as any);

        seriesRef.current = candlestickSeries;
        chartRef.current = chart;

        // Initial Fetch
        fetchHistory();

        // Polling (Every 10 seconds for fresh M1 data)
        const interval = setInterval(fetchHistory, 10000);

        const handleResize = () => {
            if (containerRef.current && chartRef.current) {
                chartRef.current.applyOptions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            clearInterval(interval);
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [symbol]);

    return (
        <div className="relative w-full h-full min-h-[400px]">
            <div ref={containerRef} className="w-full h-full" />
            
            {/* Loading / Error Overlay */}
            {(isLoading || error) && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-bg-base/50 backdrop-blur-sm z-10">
                    {isLoading && !error ? (
                        <div className="flex flex-col items-center gap-2 text-primary animate-pulse">
                            <Loader2 className="animate-spin" size={32} />
                            <span className="text-xs font-mono">加载行情数据...</span>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center gap-2 text-text-muted">
                            <AlertCircle size={32} className="opacity-50" />
                            <span className="text-sm font-bold">{error}</span>
                            <span className="text-xs opacity-50">请检查交易桥连接</span>
                        </div>
                    ) : null}
                </div>
            )}
        </div>
    );
};

export const TradingViewChart = memo(ChartComponent);
