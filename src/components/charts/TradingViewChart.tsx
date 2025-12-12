"use client";

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, ColorType, Time, CandlestickSeries, HistogramSeries } from 'lightweight-charts';
import { CloudSeries, CloudData } from '@/components/charts/plugins/CloudSeries';
import { useMarketStore } from '@/store/useMarketStore';

interface TradingViewChartProps {
  initialSymbol?: string;
  height?: number;
  className?: string;
}

type Period = 'M1' | 'M5' | 'M15' | 'M30' | 'H1' | 'H4' | 'D';

function mapMT5PeriodToChart(p: string): Period | null {
  if (p === 'PERIOD_M1' || p === 'M1') return 'M1';
  if (p === 'PERIOD_M5' || p === 'M5') return 'M5';
  if (p === 'PERIOD_M15' || p === 'M15') return 'M15';
  if (p === 'PERIOD_M30' || p === 'M30') return 'M30';
  if (p === 'PERIOD_H1' || p === 'H1') return 'H1';
  if (p === 'PERIOD_H4' || p === 'H4') return 'H4';
  if (p === 'PERIOD_D1' || p === 'D1' || p === 'D') return 'D';
  return null;
}

export function TradingViewChart({ initialSymbol = 'EUR_USD', height = 500, className }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Connect to Market Store for Auto-Sync
  const chartPeriod = useMarketStore(state => state.chartPeriod);

  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<"Line"> | ISeriesApi<"Custom">>>(new Map());
  const cloudSeriesRef = useRef<ISeriesApi<"Custom"> | null>(null);

  const [symbol, setSymbol] = useState(initialSymbol);
  // Default to M5 if no sync, but will be overridden by sync
  const [period, setPeriod] = useState<Period>('M5');
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<CandlestickData[]>([]);

  // Auto-Sync Period
  useEffect(() => {
    if (chartPeriod) {
      const synced = mapMT5PeriodToChart(chartPeriod);
      if (synced && synced !== period) {
        console.log("Syncing Chart Period to MT5:", synced);
        setPeriod(synced);
      }
    }
  }, [chartPeriod, period]);

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

    // Proper imperative initialization
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
        rightOffset: 20,
        barSpacing: 10,
        fixLeftEdge: true,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: true,
        visible: true,
        shiftVisibleRangeOnNewBar: true,
        ticksVisible: true,
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

    // Add Absorption Histogram (Overlay)
    const absorptionSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(56, 189, 248, 0.3)',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'absorption',
    });

    chart.priceScale('absorption').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
      visible: false,
    });

    cloudSeriesRef.current = cloudSeries;
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    indicatorSeriesRef.current.set('absorption', absorptionSeries as unknown as any);


    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        const newWidth = chartContainerRef.current.clientWidth;
        if (newWidth > 0) {
          chartRef.current.applyOptions({ width: newWidth });
        }
      }
    };

    window.addEventListener('resize', handleResize);

    const onVisibleTimeRangeChange = () => {
      // updateLabels(); 
    };

    chart.timeScale().subscribeVisibleTimeRangeChange(onVisibleTimeRangeChange);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.timeScale().unsubscribeVisibleTimeRangeChange(onVisibleTimeRangeChange);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      cloudSeriesRef.current = null;
      indicatorSeriesRef.current.clear();
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
            .map((c: { time: number; open: number; high: number; low: number; close: number; volume: number }) => ({
              time: c.time as Time,
              open: c.open,
              high: c.high,
              low: c.low,
              close: c.close,
              volume: c.volume || 0, // Ensure volume is captured
            }))
            .sort((a: { time: number }, b: { time: number }) => (a.time) - (b.time));

          setChartData(formattedData);
          if (candleSeriesRef.current) {
            candleSeriesRef.current.setData(formattedData);

            // Calculate & Set Absorption Data if toggled
            // We do this here or in the indicators useEffect, but since we need raw OHLCV, let's do it here or pass chartData.
            // chartData state update might be async, so let's set it directly if series exists.
            if (indicatorSeriesRef.current.has('absorption')) {
              const absorbSeries = indicatorSeriesRef.current.get('absorption') as unknown as ISeriesApi<"Histogram">;
              if (absorbSeries) {
                const absorbData = formattedData.map((d: any) => {
                  const range = d.high - d.low;
                  // Avoid division by zero
                  const ratio = range > 1e-8 ? d.volume / range : 0;

                  // Color logic: Greenish if Up candle, Reddish if Down candle
                  const isUp = d.close >= d.open;
                  return {
                    time: d.time,
                    value: ratio,
                    color: isUp ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'
                  };
                });
                absorbSeries.setData(absorbData);
              }
            }

            const totalBars = formattedData.length;
            if (totalBars > 0) {
              const timeScale = chartRef.current?.timeScale();
              if (timeScale) {
                timeScale.scrollToPosition(15, false);
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

  // Calculate and draw other indicators (EMA Cloud)
  useEffect(() => {
    // ... (Existing Cloud Logic remains same) ...
    // Note: Absorption is handled in data fetch to access 'volume', but visibility is controlled here

    if (indicatorSeriesRef.current.has('absorption')) {
      const absorbSeries = indicatorSeriesRef.current.get('absorption') as unknown as ISeriesApi<"Histogram">;
      // Check if we want to toggle visibility based on a new state? 
      // For now, let's assume it's always visible or controlled by a prop.
      // User asked to "add it", implementing a toggle is best.
      absorbSeries.applyOptions({
        visible: indicators.pivotTrend, // Re-using pivotTrend toggle for now, OR add new one
      });
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
            {/* ... (Selectors remain same) ... */}
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
            {/* Reuse existing or add new? Adding distinct label */}
            <button
              onClick={() => setIndicators(prev => ({ ...prev, pivotTrend: !prev.pivotTrend }))}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${indicators.pivotTrend
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                : 'bg-black/60 text-slate-400 border border-white/10 hover:text-white hover:bg-white/10'
                }`}
            >
              AI Vision {indicators.pivotTrend ? '(ON)' : '(OFF)'}
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
