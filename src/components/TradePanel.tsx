"use client";

import { useState, useEffect } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ChevronDown, Maximize2, MoreHorizontal, X, Minus, Plus, AlertTriangle } from 'lucide-react';
import { useBridgeStatus } from '@/hooks/useBridgeStatus';
import { cn } from '@/lib/utils';
import { Checkbox } from '@/components/ui/Checkbox';
import { Select } from '@/components/ui/Select';

interface TradePanelProps {
    open: boolean;
    onClose: () => void;
    symbol: string;
    initialSide?: 'BUY' | 'SELL';
}

export function TradePanel({ open, onClose, symbol, initialSide = 'BUY' }: TradePanelProps) {
    const { status, isConnected } = useBridgeStatus(1000);
    const priceData = status?.symbol_prices?.[symbol];
    
    const [units, setUnits] = useState(0.01);
    const [takeProfitEnabled, setTakeProfitEnabled] = useState(false);
    const [stopLossEnabled, setStopLossEnabled] = useState(false);
    const [takeProfitPrice, setTakeProfitPrice] = useState("");
    const [stopLossPrice, setStopLossPrice] = useState("");
    const [activeTab, setActiveTab] = useState<'market' | 'limit' | 'stop'>('market');
    const [pendingPrice, setPendingPrice] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [side, setSide] = useState<'BUY' | 'SELL'>(initialSide);
    const [validUntil, setValidUntil] = useState("手动取消前有效");
    
    // Reset form when opened
    useEffect(() => {
        if (open) {
            setSide(initialSide);
            setTakeProfitPrice('');
            setStopLossPrice('');
            setPendingPrice('');
            setActiveTab('market');
            setIsSubmitting(false);
            setTakeProfitEnabled(false);
            setStopLossEnabled(false);
        }
    }, [open, initialSide, symbol]);

    const bid = priceData?.bid || 0;
    const ask = priceData?.ask || 0;
    const spread = bid && ask ? ((ask - bid) * (symbol.includes('JPY') || symbol.includes('XAU') ? 100 : 10000)).toFixed(1) : '-';
    
    // Calculate dynamic tick value
    // Standard lot (1.0) tick value is usually $1 for 1 point move in non-JPY pairs, but varies.
    // For XAUUSD, 1 lot, 0.01 move = $1? No, 1 lot = 100 oz. 0.01 move = $1.
    // So tick value = units * 100 * 0.01?
    // Let's use a simplified approximation for now: $10 per lot per pip (standard forex)
    // For XAUUSD: 1 lot, 1 pip (0.10) = $10. 
    // So tick value = units * 10. (Very rough estimate)
    const tickValue = (units * 10).toFixed(2);

    const handleSubmit = async () => {
        if (!isConnected) return;
        setIsSubmitting(true);
        
        try {
            const payload: any = {
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
                // Note: Backend needs to distinguish Limit vs Stop based on logic or extra field.
                // Current backend logic infers based on price vs market price.
            }

            const res = await fetch('/api/bridge/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || 'Trade failed');
            
            onClose();
        } catch (e) {
            alert(`下单失败: ${e instanceof Error ? e.message : '未知错误'}`);
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
                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 text-slate-400 hover:bg-slate-50">
                            <Maximize2 className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 text-slate-400 hover:bg-slate-50">
                            <MoreHorizontal className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 text-slate-400 hover:bg-slate-50 hover:text-slate-700" onClick={onClose}>
                            <X className="h-4 w-4" />
                        </Button>
                    </div>
                </div>

                {/* Price Display */}
                <div className="grid grid-cols-2 gap-2 p-3">
                    <button
                        onClick={() => setSide("SELL")}
                        className={cn(
                            "rounded-lg p-2.5 transition-all text-left border",
                            side === "SELL"
                                ? "bg-red-500 text-white border-red-600 shadow-sm"
                                : "bg-red-50/50 text-red-600 border-transparent hover:bg-red-50"
                        )}
                    >
                        <div className="text-xs opacity-90 font-medium">卖出</div>
                        <div className="text-xl font-bold tracking-tight my-0.5">{bid.toFixed(5)}</div>
                        <div className="text-[10px] opacity-75">价差: {spread}</div>
                    </button>
                    <button
                        onClick={() => setSide("BUY")}
                        className={cn(
                            "rounded-lg p-2.5 transition-all text-left border",
                            side === "BUY"
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
                        {['market', 'limit', 'stop'].map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab as any)}
                                className={cn(
                                    "flex-1 pb-2 text-xs font-medium border-b-2 transition-colors capitalize",
                                    activeTab === tab ? "border-blue-500 text-blue-600" : "border-transparent text-slate-400 hover:text-slate-600"
                                )}
                            >
                                {{'market': '市价', 'limit': '限价', 'stop': '止损'}[tab]}
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
                            <h3 className="text-xs font-semibold mb-2 text-slate-800 mt-4">退出策略</h3>

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

                        {/* Valid Until (Only for Pending) */}
                        {(activeTab === 'limit' || activeTab === 'stop') && (
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

                        {/* Order Button */}
                        <div className="pt-1">
                            <Button
                                onClick={handleSubmit}
                                disabled={!isConnected || isSubmitting}
                                className={cn(
                                    "w-full h-11 text-white rounded-lg transition-all hover:opacity-90 shadow-md",
                                    side === "SELL" ? "bg-red-500 hover:bg-red-600 shadow-red-500/20" : "bg-blue-500 hover:bg-blue-600 shadow-blue-500/20"
                                )}
                            >
                                <div className="flex flex-col items-center gap-0.5">
                                    <span className="text-sm font-bold leading-none">{side === "SELL" ? "卖出" : "买入"}</span>
                                    <span className="text-[10px] opacity-90 font-normal leading-none">
                                        {units} {symbol} @ {activeTab === 'market' ? 'MKT' : pendingPrice || '---'}
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

