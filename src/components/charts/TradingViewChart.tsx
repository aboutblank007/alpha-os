"use client";

import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';

interface TradingViewChartProps {
  symbol: string;
  height?: number;
  width?: number;
}

export function TradingViewChart({ symbol, height = 400, width }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // 创建图表
    const chart = createChart(chartContainerRef.current, {
      width: width || chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      crosshair: {
        mode: 1, // CrosshairMode.Normal
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // 添加蜡烛图系列
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#10b981',
      wickDownColor: '#ef4444',
      wickUpColor: '#10b981',
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // 获取历史数据
    fetchHistoricalData(symbol, candleSeries);

    // 响应式调整
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: width || chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [symbol, height, width]);

  // 获取历史K线数据
  const fetchHistoricalData = async (
    instrument: string,
    candleSeries: ISeriesApi<"Candlestick">
  ) => {
    try {
      const response = await fetch('/api/prices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instrument,
          granularity: 'M5', // 5分钟K线
          count: 200,
        }),
      });

      if (!response.ok) {
        throw new Error('获取历史数据失败');
      }

      const data = await response.json();

      if (data.candles && data.candles.length > 0) {
        // 转换为 TradingView 格式
        const candleData: CandlestickData[] = data.candles.map((candle: any) => ({
          time: candle.time as number,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }));

        candleSeries.setData(candleData);
      }
    } catch (error) {
      console.error('获取历史数据失败:', error);
    }
  };

  return (
    <div className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">{symbol} 价格走势</h3>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1 text-xs rounded bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
            1分钟
          </button>
          <button className="px-3 py-1 text-xs rounded bg-accent-primary text-white">
            5分钟
          </button>
          <button className="px-3 py-1 text-xs rounded bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
            1小时
          </button>
        </div>
      </div>
      <div
        ref={chartContainerRef}
        className="rounded-xl border border-white/5 bg-white/[0.02] overflow-hidden"
      />
    </div>
  );
}

