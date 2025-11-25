"use client";

import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, ColorType, Time, CandlestickSeries, LineSeries, SeriesMarker, createSeriesMarkers, LineData } from 'lightweight-charts';
import { calculatePivotTrendSignals, DEFAULT_SETTINGS, OHLC, TrendState } from '@/lib/indicators';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { Settings2 } from 'lucide-react';
import { useMarketStore } from '@/store/useMarketStore';

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
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

  const [symbol, setSymbol] = useState(initialSymbol);
  const [period, setPeriod] = useState<Period>('M5');
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<CandlestickData[]>([]);

  const [indicators, setIndicators] = useState({
    pivotTrend: false,
  });
  const [trendState, setTrendState] = useState<TrendState | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Use Market Store
  const activeSymbols = useMarketStore(state => state.activeSymbols);

  // Trade syncing is now handled by the backend (trading-bridge/src/main.py)
  // The backend automatically syncs trades to Supabase when they are reported from MT5

  // 初始化图表
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
        rightOffset: 10, // Space on the right
        barSpacing: 12, // Default spacing
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

    // 使用 addSeries(CandlestickSeries) 替代 addCandlestickSeries
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#10b981',
      wickDownColor: '#ef4444',
      wickUpColor: '#10b981',
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [height]);

  // 获取数据
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
            .sort((a: { time: number }, b: { time: number }) => (a.time) - (b.time)); // Ensure ascending order

          setChartData(formattedData);
          if (candleSeriesRef.current) {
            candleSeriesRef.current.setData(formattedData);

            // 设置默认缩放级别和偏移
            const totalBars = formattedData.length;
            if (totalBars > 0) {
              // 增加右侧空白（Right Offset）
              // Lightweight Charts 的 rightOffset 并不是直接像素值，而是 K 线数量
              // 为了把最后一根 K 线推到 2/3 处，我们需要设置较大的空白
              // 假设当前视图宽度能容纳 ~60 根 K 线，我们需要约 20 根 K 线的空白

              const timeScale = chartRef.current?.timeScale();
              if (timeScale) {
                timeScale.scrollToPosition(30, false); // 负数向左滚，正数向右留白？不，scrollToPosition 的 offset 是从右边缘算的 bar 数量
                // 正值 = 右侧留白数量
                // 20 根左右的留白通常能达到 2/3 效果（取决于缩放）
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

    // 设置轮询，每分钟更新一次
    const intervalId = setInterval(fetchData, 60000);
    return () => clearInterval(intervalId);

  }, [symbol, period]);

  // 计算并绘制指标
  useEffect(() => {
    if (!chartRef.current || chartData.length === 0) return;

    // 清理旧指标
    indicatorSeriesRef.current.forEach((series) => {
      if (chartRef.current) {
        try {
          chartRef.current.removeSeries(series);
        } catch (e) {
          // Ignore errors during cleanup (e.g. if chart is already destroyed)
          console.warn('Error removing series:', e);
        }
      }
    });
    indicatorSeriesRef.current.clear();

    // Pivot Trend Signals
    if (indicators.pivotTrend) {
      // Convert chartData to OHLC format for the library
      const ohlcData: OHLC[] = chartData.map(d => ({
        time: d.time as number, // Assuming time is number (timestamp)
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));

      // Calculate Indicator
      const result = calculatePivotTrendSignals(ohlcData, DEFAULT_SETTINGS);
      setTrendState(result.currentTrend);

      // Plot EMA1 (Short)
      const ema1Series = chartRef.current.addSeries(LineSeries, {
        color: 'rgba(255, 255, 255, 0.5)',
        lineWidth: 1,
        title: 'EMA Short',
      });
      ema1Series.setData(result.ema1 as LineData[]);
      indicatorSeriesRef.current.set('pt_ema1', ema1Series);

      // Plot EMA2 (Long)
      const ema2Series = chartRef.current.addSeries(LineSeries, {
        color: 'rgba(255, 255, 0, 0.5)',
        lineWidth: 1,
        title: 'EMA Long',
      });
      ema2Series.setData(result.ema2 as LineData[]);
      indicatorSeriesRef.current.set('pt_ema2', ema2Series);

      // Plot Center Line
      const centerSeries = chartRef.current.addSeries(LineSeries, {
        lineWidth: 2,
        title: 'Center Line',
      });
      // Use dynamic color from data
      centerSeries.setData(result.centerLine as LineData[]);
      indicatorSeriesRef.current.set('pt_center', centerSeries);

      // Add Markers to CandleSeries
      const markers: SeriesMarker<Time>[] = result.signals.map(sig => ({
        time: sig.time as Time,
        position: (sig.type === 'BUY' || sig.type === 'RECLAIM_BUY') ? 'belowBar' : 'aboveBar',
        color: (sig.type === 'BUY' || sig.type === 'RECLAIM_BUY') ? '#26ba9f' : '#ba3026',
        shape: (sig.type === 'BUY' || sig.type === 'RECLAIM_BUY') ? 'arrowUp' : 'arrowDown',
        text: sig.label,
        size: 2, // Increase size
      }));

      // Use createSeriesMarkers instead of setMarkers for v5
      if (candleSeriesRef.current) {
        createSeriesMarkers(candleSeriesRef.current, markers);
      }
    } else {
      // Clear markers
      if (candleSeriesRef.current) {
        createSeriesMarkers(candleSeriesRef.current, []);
      }
      setTrendState(null);
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
            <Select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-auto min-w-[100px] md:min-w-[140px] h-auto py-1 md:py-1.5 text-xs md:text-sm bg-transparent border-none focus:ring-0 text-white font-medium cursor-pointer"
            >
              {activeSymbols && activeSymbols.length > 0 ? (
                activeSymbols.map((s: string) => (
                  <option key={s} value={s}>{s}</option>
                ))
              ) : (
                <>
                  <option value="EUR_USD">EUR/USD</option>
                  <option value="GBP_USD">GBP/USD</option>
                  <option value="USD_JPY">USD/JPY</option>
                  <option value="XAU_USD">XAU/USD</option>
                </>
              )}
            </Select>

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

          {/* Right Group: Settings */}
          <div className="flex items-center gap-2 md:gap-3 pointer-events-auto">
            {/* Settings */}
            <div className="relative">
              <Button
                variant="secondary"
                size="sm"
                className={`h-7 w-7 md:h-9 md:w-9 p-0 rounded-lg md:rounded-xl bg-black/60 backdrop-blur-md border border-white/10 hover:bg-white/10 ${showSettings ? 'text-white' : 'text-slate-400'}`}
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings2 size={14} className="md:w-[18px] md:h-[18px]" />
              </Button>

              {/* Indicators Dropdown */}
              {showSettings && (
                <div className="absolute top-full right-0 mt-2 w-48 md:w-56 p-3 md:p-4 rounded-xl md:rounded-2xl bg-[#0f172a]/95 border border-white/10 shadow-2xl z-50 animate-in fade-in zoom-in-95 duration-200 backdrop-blur-xl">
                  <h4 className="text-[10px] md:text-xs font-bold text-slate-500 uppercase mb-2 md:mb-3">Indicators</h4>
                  <div className="space-y-2 md:space-y-3">
                    <label className="flex items-center gap-2 md:gap-3 text-xs md:text-sm text-slate-300 cursor-pointer hover:text-white transition-colors">
                      <Checkbox
                        checked={indicators.pivotTrend}
                        onChange={(e) => setIndicators(prev => ({ ...prev, pivotTrend: e.target.checked }))}
                      />
                      <span>Pivot Trend (V3)</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Watermark / Loading State */}
        {chartData.length === 0 && !loading && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
            No data available
          </div>
        )}

        {/* Trend Table Overlay */}
        {indicators.pivotTrend && trendState && (
          <div className="absolute top-4 right-4 bg-black/80 border border-white/10 rounded-lg p-2 text-xs shadow-lg z-10 backdrop-blur-md">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <div className="col-span-2 font-bold text-slate-300 border-b border-white/10 pb-1 mb-1">趋势监控</div>

              <div className="text-slate-400">当前趋势</div>
              <div className={trendState.trendUp ? 'text-[#26ba9f]' : 'text-[#ba3026]'}>
                {trendState.trendUp ? '上涨' : '下跌'}
              </div>

              <div className="text-slate-400">HTF (60)</div>
              <div className={trendState.htfTrendUp ? 'text-[#26ba9f]' : 'text-[#ba3026]'}>
                {trendState.htfTrendUp ? '上涨' : '下跌'}
              </div>

              <div className="col-span-2 border-t border-white/10 my-1"></div>

              <div className="text-slate-400">建议 TP</div>
              <div className="text-white font-mono">{trendState.tpPrice.toFixed(5)}</div>

              <div className="text-slate-400">建议 SL</div>
              <div className="text-white font-mono">{trendState.slPrice.toFixed(5)}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
