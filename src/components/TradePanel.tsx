"use client";

import { useState, useEffect } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ChevronDown, Maximize2, X, Minus, Plus, Calculator, Wand2 } from 'lucide-react';
import { useMarketStore } from '@/store/useMarketStore';
import { cn } from '@/lib/utils';
import { Checkbox } from '@/components/ui/Checkbox';
import { Select } from '@/components/ui/Select';

interface TradePanelProps {
    open: boolean;
    onClose: () => void;
    symbol: string;
    initialSide?: 'BUY' | 'SELL';
    initialPrice?: number;
    initialStopLoss?: number;
    initialTakeProfit?: number;
}

export function TradePanel({ 
    open, 
    onClose, 
    symbol, 
    initialSide = 'BUY',
    initialPrice,
    initialStopLoss,
    initialTakeProfit
}: TradePanelProps) {
    // Use Market Store directly
    const isConnected = useMarketStore(state => state.isConnected);
    const priceData = useMarketStore(state => state.symbolPrices[symbol]);

    const [units, setUnits] = useState(0.01);
    const [takeProfitEnabled, setTakeProfitEnabled] = useState(!!initialTakeProfit);
    const [stopLossEnabled, setStopLossEnabled] = useState(!!initialStopLoss);
    const [takeProfitPrice, setTakeProfitPrice] = useState(initialTakeProfit?.toString() || "");
    const [stopLossPrice, setStopLossPrice] = useState(initialStopLoss?.toString() || "");
    const [activeTab, setActiveTab] = useState<'market' | 'limit' | 'stop' | 'oco'>('market');
    const [pendingPrice, setPendingPrice] = useState(initialPrice?.toString() || "");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [side, setSide] = useState<'BUY' | 'SELL'>(initialSide);
    const [validUntil, setValidUntil] = useState("手动取消前有效");
    const [error, setError] = useState<string | null>(null);
    
    // OCO State
    const [ocoBuyPrice, setOcoBuyPrice] = useState("");
    const [ocoSellPrice, setOcoSellPrice] = useState("");
    
    // Risk Calculator State
    const [showCalculator, setShowCalculator] = useState(false);
    const [riskAmount, setRiskAmount] = useState(100);
    const [riskPercent, setRiskPercent] = useState(1);

    // Determine default tab based on initialPrice
    useEffect(() => {
        if (open && initialPrice && priceData) {
            const currentPrice = initialSide === 'BUY' ? priceData.ask : priceData.bid;
            // Simple heuristic: if price is far from market, assume pending
            if (Math.abs(initialPrice - currentPrice) > 0.0005) { // Tolerance
                // Determine limit vs stop
                if (initialSide === 'BUY') {
                    setActiveTab(initialPrice < currentPrice ? 'limit' : 'stop');
                } else {
                    setActiveTab(initialPrice > currentPrice ? 'limit' : 'stop');
                }
            } else {
                setActiveTab('market');
            }
        }
    }, [open, initialPrice, priceData, initialSide]);

    // Reset form when opened
    useEffect(() => {
        if (open) {
            setSide(initialSide);
            setTakeProfitPrice(initialTakeProfit?.toString() || '');
            setStopLossPrice(initialStopLoss?.toString() || '');
            setPendingPrice(initialPrice?.toString() || '');
            setTakeProfitEnabled(!!initialTakeProfit);
            setStopLossEnabled(!!initialStopLoss);
            // activeTab logic handled above or reset to default if no initialPrice
            if (!initialPrice) setActiveTab('market');
            
            setIsSubmitting(false);
            setError(null); 
            setShowCalculator(false); // Reset calculator visibility
            setOcoBuyPrice("");
            setOcoSellPrice("");
        }
    }, [open, initialSide, symbol, initialPrice, initialStopLoss, initialTakeProfit]);

    // const price = priceData?.bid ?? 0; // Default to bid for generic price (unused)
    const bid = priceData?.bid ?? 0;
    const ask = priceData?.ask ?? 0;
    const spread = (ask - bid).toFixed(5);

    // Calculate dynamic tick value
    const tickValue = (units * 10).toFixed(2);

    // Risk Calculation Logic
    const calculateRisk = () => {
        if (!stopLossPrice || !priceData) return;
        
        const entryPrice = activeTab === 'market' 
            ? (side === 'BUY' ? ask : bid) 
            : parseFloat(pendingPrice);
            
        const sl = parseFloat(stopLossPrice);
        
        if (!entryPrice || !sl) return;
        
        const priceDiff = Math.abs(entryPrice - sl);
        
        // Heuristic for Contract Size based on symbol
        let contractSize = 100000; 
        if (symbol.includes('XAU') || symbol.includes('Gold')) contractSize = 100;
        if (symbol.includes('BTC') || symbol.includes('Bitcoin')) contractSize = 1;
        if (symbol.includes('US30') || symbol.includes('DJI')) contractSize = 1; // Indices often 1 or 10
        
        const calculatedVolume = riskAmount / (priceDiff * contractSize);
        
        // Round to 2 decimal places (standard min lot 0.01)
        const roundedVolume = Math.max(0.01, Math.floor(calculatedVolume * 100) / 100);
        
        setUnits(roundedVolume);
    };

    // Auto SL/TP Logic based on ATR (Simulated)
    const applyAutoSLTP = () => {
        // In a real app, we'd fetch ATR from backend or use indicators lib
        // Here we use a simplified heuristic: 20 pips SL, 40 pips TP
        const pips = 0.0001; // Assuming major pair
        const entryPrice = activeTab === 'market' 
            ? (side === 'BUY' ? ask : bid) 
            : parseFloat(pendingPrice);
            
        if (!entryPrice) return;

        const slDistance = 20 * pips;
        const tpDistance = 40 * pips;

        if (side === 'BUY') {
            setStopLossPrice((entryPrice - slDistance).toFixed(5));
            setTakeProfitPrice((entryPrice + tpDistance).toFixed(5));
        } else {
            setStopLossPrice((entryPrice + slDistance).toFixed(5));
            setTakeProfitPrice((entryPrice - tpDistance).toFixed(5));
        }
        
        setStopLossEnabled(true);
        setTakeProfitEnabled(true);
    };

    const handleSubmit = async () => {
        if (!isConnected) {
            setError('Bridge is not connected.');
            return;
        }
        setIsSubmitting(true);
        setError(null); 

        try {
            interface TradePayload {
                action: 'BUY' | 'SELL' | 'OCO'; // Add OCO
                symbol: string;
                volume: number;
                sl: number;
                tp: number;
                type?: string;
                price?: number;
                oco_buy_price?: number; // Add OCO params
                oco_sell_price?: number;
            }

            let payload: TradePayload;

            if (activeTab === 'oco') {
                if (!ocoBuyPrice || !ocoSellPrice) throw new Error('请输入 OCO 买入和卖出价格');
                payload = {
                    action: 'OCO',
                    symbol: symbol.replace('/', '').replace('_', ''),
                    volume: units,
                    sl: stopLossEnabled && stopLossPrice ? parseFloat(stopLossPrice) : 0,
                    tp: takeProfitEnabled && takeProfitPrice ? parseFloat(takeProfitPrice) : 0,
                    oco_buy_price: parseFloat(ocoBuyPrice),
                    oco_sell_price: parseFloat(ocoSellPrice),
                    type: 'OCO'
                };
            } else {
                payload = {
                action: side,
                symbol: symbol.replace('/', '').replace('_', ''),
                volume: units,
                sl: stopLossEnabled && stopLossPrice ? parseFloat(stopLossPrice) : 0,
                tp: takeProfitEnabled && takeProfitPrice ? parseFloat(takeProfitPrice) : 0
            };

            if (activeTab === 'limit' || activeTab === 'stop') {
                if (!pendingPrice) throw new Error('请输入挂单价格');
                payload.type = 'PENDING';
                payload.price = parseFloat(pendingPrice);
                }
            }

            const res = await fetch('/api/bridge/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || 'Trade failed');

            onClose();
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : String(err);
            setError(`下单失败: ${errorMessage}`);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <Modal
            open={open}
            onOpenChange={(o) => !o && onClose()}
            title=""
            className="w-full max-w-[420px] p-0 overflow-hidden bg-white text-slate-900 mx-auto self-center my-4 shadow-2xl"
            hideCloseButton
        >
            <div className="w-full bg-white rounded-lg overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-3 py-2 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-6 bg-green-600 rounded flex items-center justify-center">
                            <span className="text-white text-[10px]">$</span>
                        </div>
                        <span className="font-semibold text-sm text-slate-900">{symbol}</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <Button 
                            variant="ghost" 
                            size="sm" 
                            className={`h-7 w-7 p-0 hover:bg-slate-50 ${showCalculator ? 'text-blue-600 bg-blue-50' : 'text-slate-400'}`}
                            onClick={() => setShowCalculator(!showCalculator)}
                            title="风险计算器"
                        >
                            <Calculator className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 text-slate-400 hover:bg-slate-50">
                            <Maximize2 className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 text-slate-400 hover:bg-slate-50 hover:text-slate-700" onClick={onClose}>
                            <X className="h-4 w-4" />
                        </Button>
                    </div>
                </div>

                {/* Risk Calculator Overlay */}
                {showCalculator && (
                    <div className="bg-slate-50 p-3 border-b border-slate-100 animate-in slide-in-from-top-2 duration-200">
                        <div className="text-xs font-semibold text-slate-700 mb-2 flex justify-between items-center">
                            <span>风险计算器</span>
                            <span className="text-[10px] font-normal text-slate-500">基于止损距离计算手数</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 mb-2">
                            <div>
                                <label className="text-[10px] text-slate-500 mb-1 block">风险金额 ($)</label>
                                <Input 
                                    type="number" 
                                    value={riskAmount} 
                                    onChange={(e) => setRiskAmount(Number(e.target.value))}
                                    className="h-8 text-sm bg-white" 
                                />
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 mb-1 block">或者 账户 %</label>
                                <Input 
                                    type="number" 
                                    value={riskPercent} 
                                    onChange={(e) => {
                                        setRiskPercent(Number(e.target.value));
                                        // Ideally calculate amount from balance if available
                                    }}
                                    className="h-8 text-sm bg-white" 
                                />
                            </div>
                        </div>
                        <Button 
                            size="sm" 
                            className="w-full h-8 text-xs bg-blue-600 hover:bg-blue-700 text-white"
                            onClick={() => {
                                if (!stopLossPrice) {
                                    setError("请先设置止损价格以计算风险");
                                    return;
                                }
                                calculateRisk();
                                setShowCalculator(false);
                            }}
                        >
                            应用计算结果
                        </Button>
                    </div>
                )}

                {/* Price Display */}
                <div className="grid grid-cols-2 gap-2 p-3">
                    <button
                        onClick={() => { setSide("SELL"); if(activeTab === 'oco') setActiveTab('market'); }}
                        className={cn(
                            "rounded-lg p-2.5 transition-all text-left border",
                            side === "SELL" && activeTab !== 'oco'
                                ? "bg-red-500 text-white border-red-600 shadow-sm"
                                : "bg-red-50/50 text-red-600 border-transparent hover:bg-red-50"
                        )}
                    >
                        <div className="text-xs opacity-90 font-medium">卖出</div>
                        <div className="text-xl font-bold tracking-tight my-0.5">{bid.toFixed(5)}</div>
                        <div className="text-[10px] opacity-75">价差: {spread}</div>
                    </button>
                    <button
                        onClick={() => { setSide("BUY"); if(activeTab === 'oco') setActiveTab('market'); }}
                        className={cn(
                            "rounded-lg p-2.5 transition-all text-left border",
                            side === "BUY" && activeTab !== 'oco'
                                ? "bg-blue-500 text-white border-blue-600 shadow-sm"
                                : "bg-blue-50/50 text-blue-600 border-transparent hover:bg-blue-50"
                        )}
                    >
                        <div className="text-xs opacity-90 font-medium">买入</div>
                        <div className="text-xl font-bold tracking-tight my-0.5">{ask.toFixed(5)}</div>
                        <div className="text-[10px] opacity-75">价差: {spread}</div>
                    </button>
                </div>

                {/* Tabs */}
                <div className="px-3">
                    <div className="flex border-b border-slate-100 mb-3">
                        {['market', 'limit', 'stop', 'oco'].map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab as 'market' | 'limit' | 'stop' | 'oco')}
                                className={cn(
                                    "flex-1 pb-2 text-xs font-medium border-b-2 transition-colors capitalize",
                                    activeTab === tab ? "border-blue-500 text-blue-600" : "border-transparent text-slate-400 hover:text-slate-600"
                                )}
                            >
                                {{ 'market': '市价', 'limit': '限价', 'stop': '止损', 'oco': 'OCO' }[tab]}
                            </button>
                        ))}
                    </div>

                    <div className="space-y-3 pb-4">
                        {/* Pending Order Inputs */}
                        {(activeTab === 'limit' || activeTab === 'stop') && (
                            <div className="space-y-1.5">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500">价格</span>
                                    <span className="text-xs text-slate-500">Ticks</span>
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <Input
                                        value={pendingPrice}
                                        onChange={(e) => setPendingPrice(e.target.value)}
                                        className="h-9 text-sm bg-white border-slate-200 text-slate-900 px-2"
                                        placeholder={activeTab === 'limit' ? (side === 'BUY' ? `< ${ask}` : `> ${bid}`) : (side === 'BUY' ? `> ${ask}` : `< ${bid}`)}
                                    />
                                    <div className="flex items-center justify-center px-2 h-9 border border-slate-200 rounded-md bg-slate-50 text-xs text-slate-500">
                                        Auto
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* OCO Inputs */}
                        {activeTab === 'oco' && (
                            <div className="space-y-3 bg-slate-50 p-2 rounded-md border border-slate-100">
                                <div className="text-xs text-slate-500 font-medium">突破策略 (One Cancels Other)</div>
                                <div className="grid grid-cols-2 gap-2">
                                    <div>
                                        <label className="text-[10px] text-slate-500 mb-1 block text-blue-600 font-medium">Buy Stop</label>
                                        <Input
                                            value={ocoBuyPrice}
                                            onChange={(e) => setOcoBuyPrice(e.target.value)}
                                            className="h-8 text-sm bg-white"
                                            placeholder={`> ${ask}`}
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-slate-500 mb-1 block text-red-600 font-medium">Sell Stop</label>
                                        <Input
                                            value={ocoSellPrice}
                                            onChange={(e) => setOcoSellPrice(e.target.value)}
                                            className="h-8 text-sm bg-white"
                                            placeholder={`< ${bid}`}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Units Input */}
                        <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-slate-500">单位 (Lots)</span>
                                <div className="flex items-center gap-1">
                                    <span className="text-xs text-slate-500">风险，%余额</span>
                                    <ChevronDown className="h-3 w-3 text-slate-400" />
                                </div>
                            </div>
                            <div className="grid grid-cols-[1fr,auto] gap-2">
                                <div className="relative flex items-center border border-slate-200 rounded-md bg-white h-9">
                                    <Input
                                        type="number"
                                        value={units}
                                        onChange={(e) => setUnits(Math.max(0.01, Number(e.target.value)))}
                                        step="0.01"
                                        className="border-0 h-full pr-8 focus-visible:ring-1 focus-visible:ring-blue-500 text-slate-900 font-bold text-sm px-2"
                                    />
                                    <div className="absolute right-0.5 flex flex-col">
                                        <button
                                            className="h-4 w-5 flex items-center justify-center text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-sm"
                                            onClick={() => setUnits(Number((units + 0.01).toFixed(2)))}
                                        >
                                            <Plus className="h-2.5 w-2.5" />
                                        </button>
                                        <button
                                            className="h-4 w-5 flex items-center justify-center text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-sm"
                                            onClick={() => setUnits(Math.max(0.01, Number((units - 0.01).toFixed(2))))}
                                        >
                                            <Minus className="h-2.5 w-2.5" />
                                        </button>
                                    </div>
                                </div>
                                <div className="flex items-center justify-center px-2 bg-slate-50 border border-slate-200 rounded-md text-xs font-medium text-slate-600 w-20 h-9">
                                    ~25.6%
                                </div>
                            </div>
                        </div>

                        {/* Exit Section */}
                        <div>
                            <div className="flex justify-between items-center mb-2 mt-4">
                                <h3 className="text-xs font-semibold text-slate-800">退出策略</h3>
                                <Button 
                                    variant="ghost" 
                                    size="sm" 
                                    className="h-5 text-[10px] text-blue-600 hover:bg-blue-50 px-2"
                                    onClick={applyAutoSLTP}
                                    title="自动设置 2:1 盈亏比"
                                >
                                    <Wand2 size={10} className="mr-1" />
                                    Auto
                                </Button>
                            </div>

                            {/* Take Profit */}
                            <div className="space-y-1.5 mb-3">
                                <div className="flex items-center gap-2" onClick={() => setTakeProfitEnabled(!takeProfitEnabled)}>
                                    <Checkbox
                                        checked={takeProfitEnabled}
                                        onChange={(e) => setTakeProfitEnabled(e.target.checked)}
                                        className="pointer-events-none"
                                    />
                                    <span className="text-xs text-slate-600 cursor-pointer select-none">止盈 (TP)</span>
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <Input
                                        value={takeProfitPrice}
                                        onChange={(e) => setTakeProfitPrice(e.target.value)}
                                        className={cn(
                                            "h-8 text-sm border-slate-200 text-slate-900 px-2",
                                            takeProfitEnabled ? "bg-white" : "bg-slate-50"
                                        )}
                                        disabled={!takeProfitEnabled}
                                        placeholder="价格"
                                    />
                                    <Input
                                        value={takeProfitEnabled ? "47.87" : ""}
                                        readOnly
                                        className="h-8 text-sm bg-slate-50 border-slate-200 text-slate-400 px-2"
                                        disabled={!takeProfitEnabled}
                                        placeholder="奖励"
                                    />
                                </div>
                            </div>

                            {/* Stop Loss */}
                            <div className="space-y-1.5">
                                <div className="flex items-center gap-2" onClick={() => setStopLossEnabled(!stopLossEnabled)}>
                                    <Checkbox
                                        checked={stopLossEnabled}
                                        onChange={(e) => setStopLossEnabled(e.target.checked)}
                                        className="pointer-events-none"
                                    />
                                    <span className="text-xs text-slate-600 cursor-pointer select-none">止损 (SL)</span>
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <Input
                                        value={stopLossPrice}
                                        onChange={(e) => setStopLossPrice(e.target.value)}
                                        className={cn(
                                            "h-8 text-sm border-slate-200 text-slate-900 px-2",
                                            stopLossEnabled ? "bg-white" : "bg-slate-50"
                                        )}
                                        disabled={!stopLossEnabled}
                                        placeholder="价格"
                                    />
                                    <Input
                                        value={stopLossEnabled ? "165.7" : ""}
                                        readOnly
                                        className="h-8 text-sm bg-slate-50 border-slate-200 text-slate-400 px-2"
                                        disabled={!stopLossEnabled}
                                        placeholder="Ticks"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Valid Until (Only for Pending/OCO) */}
                        {(activeTab === 'limit' || activeTab === 'stop' || activeTab === 'oco') && (
                            <div className="space-y-1.5 mt-2">
                                <div className="text-xs text-slate-500">有效时间</div>
                                <Select value={validUntil} onChange={(e) => setValidUntil(e.target.value)} className="h-9 text-sm bg-white text-slate-900 border-slate-200 px-2 py-0">
                                    <option value="手动取消前有效">手动取消前有效</option>
                                    <option value="今日有效">今日有效</option>
                                    <option value="本周有效">本周有效</option>
                                </Select>
                            </div>
                        )}

                        {/* Order Info */}
                        <div className="pt-3 border-t border-slate-100 mt-2">
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-slate-500">Tick Value</span>
                                <span className="text-xs text-slate-700 font-mono">
                                    <span className="font-semibold">{tickValue}</span> USD
                                </span>
                            </div>
                        </div>

                        {/* Error Message */}
                        {error && (
                            <div className="px-3 py-2 text-xs text-red-500 bg-red-50 border border-red-100 rounded-md mb-2">
                                {error}
                            </div>
                        )}

                        {/* Order Button */}
                        <div className="pt-1">
                            <Button
                                onClick={handleSubmit}
                                disabled={!isConnected || isSubmitting}
                                className={cn(
                                    "w-full h-11 text-white rounded-lg transition-all hover:opacity-90 shadow-md",
                                    activeTab === 'oco'
                                        ? "bg-slate-800 hover:bg-slate-700"
                                        : (side === "SELL" ? "bg-red-500 hover:bg-red-600 shadow-red-500/20" : "bg-blue-500 hover:bg-blue-600 shadow-blue-500/20")
                                )}
                            >
                                <div className="flex flex-col items-center gap-0.5">
                                    <span className="text-sm font-bold leading-none">
                                        {activeTab === 'oco' ? '放置 OCO 订单' : (side === "SELL" ? "卖出" : "买入")}
                                    </span>
                                    <span className="text-[10px] opacity-90 font-normal leading-none">
                                        {units} {symbol} @ {activeTab === 'market' ? 'MKT' : (activeTab === 'oco' ? 'Breakout' : pendingPrice || '---')}
                                    </span>
                                </div>
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        </Modal>
    );
}
