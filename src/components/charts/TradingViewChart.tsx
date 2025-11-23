"use client";

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, ColorType, LineStyle, Time, CandlestickSeries, LineSeries, SeriesMarker, SeriesMarkerPosition, SeriesMarkerShape, createSeriesMarkers } from 'lightweight-charts';
import { calculatePivotTrendSignals, DEFAULT_SETTINGS, OHLC, IndicatorResult, TrendState } from '@/lib/indicators';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { Loader2, Settings2, Wifi, WifiOff, TrendingUp, TrendingDown } from 'lucide-react';

import { supabase } from '@/lib/supabase';

interface TradingViewChartProps {
  initialSymbol?: string;
  height?: number;
  className?: string;
}

interface BridgeStatus {
  bridge_status: 'connected' | 'disconnected';
  last_mt5_update: {
    symbol?: string;
    bid?: number;
    ask?: number;
  };
  pending_commands: number;
  error?: string;
}

type Period = 'M1' | 'M5' | 'M15' | 'H1' | 'H4' | 'D';

const PERIODS: { label: string; value: Period }[] = [
  { label: '1分', value: 'M1' },
  { label: '5分', value: 'M5' },
  { label: '15分', value: 'M15' },
  { label: '1时', value: 'H1' },
  { label: '4时', value: 'H4' },
  { label: '日线', value: 'D' },
];

const SYMBOLS = [
  { label: 'EUR/USD', value: 'EUR_USD' },
  { label: 'GBP/USD', value: 'GBP_USD' },
  { label: 'USD/JPY', value: 'USD_JPY' },
  { label: 'XAU/USD (黄金)', value: 'XAU_USD' },
  { label: 'BTC/USD', value: 'BTC_USD' },
];

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
    pivotTrend: true, // Default enabled as requested
  });
  const [trendState, setTrendState] = useState<TrendState | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Bridge State
  const [bridgeStatus, setBridgeStatus] = useState<BridgeStatus | null>(null);
  const [executing, setExecuting] = useState<string | null>(null);

  // Poll Bridge Status
  const lastProcessedTicketRef = useRef<number | null>(null);
  
  useEffect(() => {
    let mounted = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/bridge/status');
        const data = await res.json();
        if (mounted) {
          setBridgeStatus(data);
          
            // Check for new trade execution and sync to Supabase
          const lastTrade = data.last_trade;
          if (lastTrade && lastTrade.ticket && lastTrade.ticket !== lastProcessedTicketRef.current) {
            console.log('New trade detected from bridge:', lastTrade);
            lastProcessedTicketRef.current = lastTrade.ticket;
            
            // Sync to Supabase
            // Convert MT5 side (BUY/SELL) to db side (buy/sell)
            const dbSide = lastTrade.type.toLowerCase();
            
            const tradeData = {
              symbol: lastTrade.symbol,
              side: dbSide,
              quantity: lastTrade.volume,
              entry_price: lastTrade.price,
              status: 'open',
              created_at: new Date().toISOString(), // Use current time as created_at
              pnl_net: 0, // Initial PnL
              pnl_gross: 0,
              commission: 0,
              swap: 0,
              // Store ticket in notes or metadata if needed
              notes: `MT5 Ticket: ${lastTrade.ticket}`
            };

            const { error } = await supabase.from('trades').insert(tradeData);
            
            if (error) {
              console.error('Failed to sync trade to Supabase:', error);
              // Fallback: Could trigger a toast here or local state update
            } else {
              console.log('Trade synced to Supabase successfully');
            }
          }
        }
      } catch (e) {
        console.error('Bridge status error:', e);
        if (mounted) setBridgeStatus(prev => prev ? { ...prev, bridge_status: 'disconnected' } : null);
      }
    };
    
    fetchStatus();
    const timer = setInterval(fetchStatus, 1000); // Poll every 1s
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  const handleTrade = async (action: 'BUY' | 'SELL') => {
    if (executing) return;
    setExecuting(action);
    try {
      const res = await fetch('/api/bridge/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          symbol: symbol.replace('_', ''), // EUR_USD -> EURUSD
          volume: 0.01 // Fixed volume for MVP
        })
      });
      
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Trade failed');
      }
      
      // Optional: Add toast notification here
      console.log('Trade queued:', data);
    } catch (e) {
      alert(`Trade Execution Failed: ${e instanceof Error ? e.message : 'Unknown error'}`);
    } finally {
      setExecuting(null);
    }
  };

  const isConnected = bridgeStatus?.bridge_status === 'connected' && bridgeStatus?.last_mt5_update?.bid;

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
          const formattedData = data.candles.map((c: any) => ({
            time: c.time as Time,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          }));

          setChartData(formattedData);
          if (candleSeriesRef.current) {
            candleSeriesRef.current.setData(formattedData);
            chartRef.current?.timeScale().fitContent();
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
      ema1Series.setData(result.ema1 as any);
      indicatorSeriesRef.current.set('pt_ema1', ema1Series);

      // Plot EMA2 (Long)
      const ema2Series = chartRef.current.addSeries(LineSeries, {
        color: 'rgba(255, 255, 0, 0.5)',
        lineWidth: 1,
        title: 'EMA Long',
      });
      ema2Series.setData(result.ema2 as any);
      indicatorSeriesRef.current.set('pt_ema2', ema2Series);

      // Plot Center Line
      const centerSeries = chartRef.current.addSeries(LineSeries, {
        lineWidth: 2,
        title: 'Center Line',
      });
      // Use dynamic color from data
      centerSeries.setData(result.centerLine as any);
      indicatorSeriesRef.current.set('pt_center', centerSeries);

      // Add Markers to CandleSeries
      const markers: SeriesMarker<Time>[] = result.signals.map(sig => ({
        time: sig.time as Time,
        position: (sig.type === 'BUY' || sig.type === 'RECLAIM_BUY') ? 'belowBar' : 'aboveBar',
        color: (sig.type === 'BUY' || sig.type === 'RECLAIM_BUY') ? '#26ba9f' : '#ba3026',
        shape: (sig.type === 'BUY' || sig.type === 'RECLAIM_BUY') ? 'arrowUp' : 'arrowDown',
        text: sig.label,
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
      {/* Controls Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-white/5 border border-white/5 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <Select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-32 h-9 text-sm bg-black/20 border-white/10"
          >
            {SYMBOLS.map(s => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </Select>

          <div className="flex bg-black/20 rounded-lg p-1 border border-white/10">
            {PERIODS.map(p => (
              <button
                key={p.value}
                onClick={() => setPeriod(p.value)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${period === p.value
                  ? 'bg-accent-primary text-white shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Trading Controls */}
          <div className="hidden md:flex items-center gap-3 mr-2 pr-4 border-r border-white/10">
            {/* Status Indicator */}
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium transition-colors ${isConnected ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
              {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
              <span>{isConnected ? 'MT5 连线' : '断开'}</span>
            </div>

            {/* Price Display */}
            {isConnected && bridgeStatus?.last_mt5_update && (
              <div className="flex items-center gap-3 text-sm font-mono">
                <div className="flex flex-col leading-none">
                  <span className="text-[10px] text-slate-500">BID</span>
                  <span className="text-red-400">{bridgeStatus.last_mt5_update.bid?.toFixed(5)}</span>
                </div>
                <div className="flex flex-col leading-none">
                  <span className="text-[10px] text-slate-500">ASK</span>
                  <span className="text-green-400">{bridgeStatus.last_mt5_update.ask?.toFixed(5)}</span>
                </div>
              </div>
            )}

            {/* Trade Buttons */}
            <div className="flex items-center gap-2">
              <Button 
                size="sm" 
                variant="outline" 
                className="bg-red-500/10 border-red-500/20 hover:bg-red-500/20 text-red-400 h-8 px-3 gap-1"
                onClick={() => handleTrade('SELL')}
                disabled={!isConnected || !!executing}
              >
                {executing === 'SELL' ? <Loader2 className="animate-spin" size={14} /> : <TrendingDown size={14} />}
                SELL
              </Button>
              <Button 
                size="sm" 
                variant="outline" 
                className="bg-green-500/10 border-green-500/20 hover:bg-green-500/20 text-green-400 h-8 px-3 gap-1"
                onClick={() => handleTrade('BUY')}
                disabled={!isConnected || !!executing}
              >
                {executing === 'BUY' ? <Loader2 className="animate-spin" size={14} /> : <TrendingUp size={14} />}
                BUY
              </Button>
            </div>
          </div>

          <div className="relative">
            <Button
              variant="secondary"
              size="sm"
              className={`gap-2 ${showSettings ? 'bg-white/10 text-white' : ''}`}
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings2 size={16} />
              指标
            </Button>

            {/* Indicators Dropdown */}
            {showSettings && (
              <div className="absolute top-full right-0 mt-2 w-48 p-3 rounded-xl bg-[#1e293b] border border-white/10 shadow-xl z-50 animate-in fade-in zoom-in-95 duration-200">
                <div className="space-y-3">
                  <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
                    <Checkbox
                      checked={indicators.pivotTrend}
                      onChange={(e) => setIndicators(prev => ({ ...prev, pivotTrend: e.target.checked }))}
                    />
                    Pivot Trend Signals (V3)
                  </label>
                </div>
              </div>
            )}
          </div>

          {loading && <Loader2 className="animate-spin text-accent-primary" size={20} />}
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative w-full rounded-xl overflow-hidden border border-white/5 bg-black/20 backdrop-blur-sm">
        <div ref={chartContainerRef} className="w-full" style={{ height }} />

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

// --- Helper Functions ---

function calculateSMA(data: CandlestickData[], period: number) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close;
    }
    result.push({
      time: data[i].time,
      value: sum / period,
    });
  }
  return result;
}

function calculateEMA(data: CandlestickData[], period: number) {
  const result = [];
  const k = 2 / (period + 1);
  let ema = data[0].close;

  // Initialize EMA with SMA of first 'period' elements roughly or just start from 0?
  // Standard EMA usually starts with SMA of first N periods. 
  // For simplicity, we start EMA from the first close price, but it stabilizes after some periods.

  for (let i = 0; i < data.length; i++) {
    ema = (data[i].close - ema) * k + ema;
    if (i >= period - 1) {
      result.push({ time: data[i].time, value: ema });
    }
  }
  return result;
}

function calculateBollingerBands(data: CandlestickData[], period: number, multiplier: number) {
  const upper = [];
  const lower = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;

    // Calculate SMA
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close;
    }
    const sma = sum / period;

    // Calculate StdDev
    let sumSqDiff = 0;
    for (let j = 0; j < period; j++) {
      sumSqDiff += Math.pow(data[i - j].close - sma, 2);
    }
    const stdDev = Math.sqrt(sumSqDiff / period);

    upper.push({ time: data[i].time, value: sma + multiplier * stdDev });
    lower.push({ time: data[i].time, value: sma - multiplier * stdDev });
  }

  return { upper, lower };
}
