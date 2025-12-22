/**
 * 量子 K 线图组件 (影线染色扩展)
 * 
 * 基于 lightweight-charts，当 wickRatio > 0.6 时影线染色为 Neon Cyan
 * 表示"量子噪声"高，市场处于叠加态，预测可信度低
 */

'use client';

import { useEffect, useRef, useState } from 'react';
import { createChart, CandlestickSeries, type IChartApi, type CandlestickData, type Time } from 'lightweight-charts';
import { cn } from '@/lib/utils';
import { THRESHOLDS, type CandleData } from '@/types/quantum';

interface QuantumCandleChartProps {
    /** K 线数据 */
    data: CandleData[];
    /** 交易品种 */
    symbol?: string;
    /** 容器类名 */
    className?: string;
    /** 主题 */
    theme?: 'dark' | 'light';
}

// 高熵影线颜色 (Neon Cyan)
const HIGH_ENTROPY_WICK_COLOR = '#00FFFF';
// 正常影线颜色
const NORMAL_UP_WICK_COLOR = '#26a69a';
const NORMAL_DOWN_WICK_COLOR = '#ef5350';

export function QuantumCandleChart({
    data,
    symbol = '',
    className,
    theme = 'dark',
}: QuantumCandleChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const candleSeriesRef = useRef<any>(null);
    const [isReady, setIsReady] = useState(false);

    // 初始化图表
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { color: 'transparent' },
                textColor: theme === 'dark' ? '#d1d4dc' : '#191919',
            },
            grid: {
                vertLines: { color: theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.1)' },
                horzLines: { color: theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.1)' },
            },
            crosshair: {
                mode: 1, // Magnet mode
                vertLine: {
                    labelBackgroundColor: '#3b82f6',
                },
                horzLine: {
                    labelBackgroundColor: '#3b82f6',
                },
            },
            rightPriceScale: {
                borderColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
            },
            timeScale: {
                borderColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                timeVisible: true,
                secondsVisible: false,
            },
            handleScale: {
                axisPressedMouseMove: true,
            },
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
                vertTouchDrag: true,
                horzTouchDrag: true,
            },
        });

        // 创建 K 线系列 (v5 API)
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: NORMAL_UP_WICK_COLOR,
            wickDownColor: NORMAL_DOWN_WICK_COLOR,
        } as any);

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;

        // 自适应宽度
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        handleResize();
        window.addEventListener('resize', handleResize);
        setIsReady(true);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
            chartRef.current = null;
            candleSeriesRef.current = null;
        };
    }, [theme]);

    // 更新数据
    useEffect(() => {
        if (!candleSeriesRef.current || !isReady || data.length === 0) return;

        // 转换数据格式并应用影线染色
        const chartData: CandlestickData[] = data.map((candle) => {
            const isHighEntropy = (candle.wickRatio ?? 0) > THRESHOLDS.HIGH_ENTROPY_WICK;
            const isUp = candle.close >= candle.open;

            return {
                time: candle.time as Time,
                open: candle.open,
                high: candle.high,
                low: candle.low,
                close: candle.close,
                // 高熵时影线染色为 Neon Cyan
                wickColor: isHighEntropy
                    ? HIGH_ENTROPY_WICK_COLOR
                    : isUp
                        ? NORMAL_UP_WICK_COLOR
                        : NORMAL_DOWN_WICK_COLOR,
                // 高熵时主体变灰
                color: isHighEntropy
                    ? 'rgba(128, 128, 128, 0.5)'
                    : undefined,
                borderColor: isHighEntropy
                    ? 'rgba(128, 128, 128, 0.8)'
                    : undefined,
            };
        });

        candleSeriesRef.current.setData(chartData);

        // 滚动到最新
        if (chartRef.current) {
            chartRef.current.timeScale().scrollToRealTime();
        }
    }, [data, isReady]);

    return (
        <div className={cn('relative w-full h-full', className)}>
            <div ref={chartContainerRef} className="absolute inset-0" />

            {/* 图例 */}
            <div className="absolute top-2 right-2 z-10 flex items-center gap-3 bg-bg-card/80 backdrop-blur px-3 py-1.5 rounded-lg text-xs">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: HIGH_ENTROPY_WICK_COLOR }} />
                    <span className="text-text-muted">高熵 (噪音)</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-green-500" />
                    <span className="text-text-muted">正常</span>
                </div>
            </div>

            {/* 加载状态 */}
            {!isReady && (
                <div className="absolute inset-0 flex items-center justify-center bg-bg-card/50 backdrop-blur">
                    <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                </div>
            )}
        </div>
    );
}
