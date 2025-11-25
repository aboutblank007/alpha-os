"use client";

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, ColorType, Time, CandlestickSeries } from 'lightweight-charts';
import { CloudSeries, CloudData } from '@/components/charts/plugins/CloudSeries';

interface TradingViewChartProps {
  initialSymbol?: string;
  height?: number;
  className?: string;
}

type Period = 'M1' | 'M5' | 'M15' | 'M30' | 'H1' | 'H4' | 'D';

export function TradingViewChart({ initialSymbol = 'EUR_USD', height = 500, className }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<"Line"> | ISeriesApi<"Custom">>>(new Map());
  const cloudSeriesRef = useRef<ISeriesApi<"Custom"> | null>(null);

  const [symbol, setSymbol] = useState(initialSymbol);
  const [period, setPeriod] = useState<Period>('M5');
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<CandlestickData[]>([]);

  const [indicators, setIndicators] = useState({
    pivotTrend: true,
  });

  // Update Labels Position
  const updateLabels = useCallback(() => {
    // Your existing updateLabels logic
  }, []);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        timeVisible: true,
        rightOffset: 10,
        barSpacing: 12,
        fixLeftEdge: true,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: true,
        visible: true,
        shiftVisibleRangeOnNewBar: true,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#10b981',
      wickDownColor: '#ef4444',
      wickUpColor: '#10b981',
    });

    // Add Cloud Series with proper configuration
    const cloudSeries = chart.addCustomSeries(new CloudSeries(), {
      color: 'rgba(0, 0, 0, 0)',
      priceLineVisible: false,
      lastValueVisible: false,
    });
    cloudSeriesRef.current = cloudSeries;

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        updateLabels();
      }
    };

    window.addEventListener('resize', handleResize);

    chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      requestAnimationFrame(updateLabels);
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      cloudSeriesRef.current = null;
    };
  }, [height, updateLabels]);

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch('/api/prices', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instrument: symbol,
            granularity: period,
            count: 300,
          }),
        });

        if (!response.ok) throw new Error('Failed to fetch data');

        const data = await response.json();
        if (data.candles) {
          const formattedData = data.candles
            .map((c: { time: number; open: number; high: number; low: number; close: number }) => ({
              time: c.time as Time,
              open: c.open,
              high: c.high,
              low: c.low,
              close: c.close,
            }))
            .sort((a: { time: number }, b: { time: number }) => (a.time) - (b.time));

          setChartData(formattedData);
          if (candleSeriesRef.current) {
            candleSeriesRef.current.setData(formattedData);

            const totalBars = formattedData.length;
            if (totalBars > 0) {
              const timeScale = chartRef.current?.timeScale();
              if (timeScale) {
                timeScale.scrollToPosition(30, false);
              }
            }
          }
        }
      } catch (error) {
        console.error('Error fetching chart data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, 60000);
    return () => clearInterval(intervalId);

  }, [symbol, period]);

  // Calculate and draw indicators
  useEffect(() => {
    if (!chartRef.current || chartData.length === 0) return;

    // Clean up old indicators
    indicatorSeriesRef.current.forEach((series) => {
      if (chartRef.current) {
        try {
          chartRef.current.removeSeries(series);
        } catch (e) {
          console.warn('Error removing series:', e);
        }
      }
    });
    indicatorSeriesRef.current.clear();

    // Clear Cloud Data
    if (cloudSeriesRef.current) {
      cloudSeriesRef.current.setData([]);
    }

    if (indicators.pivotTrend) {
      // Example: Calculate EMAs (replace with your actual indicator logic)
      const ema1Data: Array<{time: Time, value: number}> = [];
      const ema2Data: Array<{time: Time, value: number}> = [];
      
      // Simple moving average calculation (replace with your EMA calculation)
      const period1 = 9;
      const period2 = 21;
      
      for (let i = period2; i < chartData.length; i++) {
        const time = chartData[i].time;
        
        // Calculate simple averages (replace with proper EMA)
        let sum1 = 0, sum2 = 0;
        for (let j = 0; j < period1; j++) {
          sum1 += chartData[i - j].close;
        }
        for (let j = 0; j < period2; j++) {
          sum2 += chartData[i - j].close;
        }
        
        const ema1 = sum1 / period1;
        const ema2 = sum2 / period2;
        
        ema1Data.push({ time, value: ema1 });
        ema2Data.push({ time, value: ema2 });
      }

      // Prepare Cloud Data
      const cloudData: CloudData[] = [];
      const ema1Map = new Map(ema1Data.map(i => [i.time, i.value]));
      const ema2Map = new Map(ema2Data.map(i => [i.time, i.value]));

      const allTimes = new Set([...ema1Data.map(d => d.time), ...ema2Data.map(d => d.time)]);
      const sortedTimes = Array.from(allTimes).sort((a, b) => (a as number) - (b as number));

      for (const t of sortedTimes) {
        const v1 = ema1Map.get(t);
        const v2 = ema2Map.get(t);
        const time = t as Time;

        if (v1 !== undefined && v2 !== undefined) {
          cloudData.push({
            time,
            ema1: v1,
            ema2: v2
          });
        }
      }

      // Set Cloud Data
      if (cloudSeriesRef.current && cloudData.length > 0) {
        cloudSeriesRef.current.setData(cloudData);
      }
    }

  }, [chartData, indicators]);

  return (
    <div className={`flex flex-col gap-4 w-full ${className}`}>
      {/* Chart Container */}
      <div className="relative w-full rounded-2xl overflow-hidden border border-white/5 bg-black/40 backdrop-blur-sm shadow-2xl">
        <div ref={chartContainerRef} className="w-full" style={{ height }} />

        {/* Floating Controls Bar */}
        <div className="absolute top-2 md:top-4 left-2 md:left-4 right-2 md:right-4 flex flex-wrap items-center justify-between gap-2 md:gap-4 pointer-events-none z-20">
          {/* Left Group: Symbol & Period */}
          <div className="flex items-center gap-1 md:gap-2 pointer-events-auto bg-black/60 backdrop-blur-md p-1 md:p-1.5 rounded-lg md:rounded-xl border border-white/10 shadow-lg">
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-auto min-w-[100px] md:min-w-[140px] h-auto py-1 md:py-1.5 text-xs md:text-sm bg-transparent border-none focus:ring-0 text-white font-medium cursor-pointer"
            >
              <option value="EUR_USD">EUR/USD</option>
              <option value="GBP_USD">GBP/USD</option>
              <option value="USD_JPY">USD/JPY</option>
              <option value="XAU_USD">XAU/USD</option>
            </select>

            <div className="w-px h-5 md:h-6 bg-white/10"></div>

            <div className="flex gap-0.5 md:gap-1">
              {([
                { label: '1M', value: 'M1' as Period },
                { label: '5M', value: 'M5' as Period },
                { label: '15M', value: 'M15' as Period },
                { label: '30M', value: 'M30' as Period },
                { label: '1H', value: 'H1' as Period },
                { label: '4H', value: 'H4' as Period },
                { label: '1D', value: 'D' as Period }
              ]).map(p => (
                <button
                  key={p.value}
                  onClick={() => setPeriod(p.value)}
                  className={`px-1.5 md:px-2 py-0.5 md:py-1 text-[10px] md:text-xs font-medium rounded transition-all ${period === p.value
                    ? 'bg-white/20 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-white/10'
                    }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Right Group: Indicators Toggle */}
          <div className="flex items-center gap-2 pointer-events-auto">
            <button
              onClick={() => setIndicators(prev => ({ ...prev, pivotTrend: !prev.pivotTrend }))}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${indicators.pivotTrend
                ? 'bg-white/20 text-white border border-white/20'
                : 'bg-black/60 text-slate-400 border border-white/10 hover:text-white hover:bg-white/10'
                }`}
            >
              EMA Cloud
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm">
            <div className="text-slate-400 text-sm">Loading...</div>
          </div>
        )}

        {/* No Data State */}
        {chartData.length === 0 && !loading && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
            No data available
          </div>
        )}
      </div>
    </div>
  );
}
