"use client";

import React, { useState, useEffect } from 'react';
import { GlassCard, CardHeader, CardTitle, CardContent } from "@/components/ui/GlassCard";
import { Search, TrendingUp, TrendingDown } from "lucide-react";
import { Input } from "@/components/ui/Input";

export function MarketWatch({ onSymbolSelect }: { onSymbolSelect?: (s: string) => void }) {
    const [filter, setFilter] = useState("");
    const [quotes, setQuotes] = useState<any[]>([]);

    useEffect(() => {
        const fetchQuotes = async () => {
            // Use bridge status for real-time quotes
            try {
                const res = await fetch('/api/bridge/status');
                if (res.ok) {
                    const data = await res.json();
                    // Data format: { symbol_prices: { EURUSD: { bid: x, ask: y ... } } }
                    if (data.symbol_prices) {
                        const list = Object.entries(data.symbol_prices).map(([sym, val]: [string, any]) => ({
                            symbol: sym,
                            bid: val.bid,
                            ask: val.ask,
                            change: 0.0, // Bridge doesn't send change% yet, default 0
                            spread: Math.round((val.ask - val.bid) * 100000)
                        }));
                        setQuotes(list);
                    }
                }
            } catch (e) {
                console.error("MarketWatch fetch error", e);
            }
        };

        fetchQuotes();
        const interval = setInterval(fetchQuotes, 2000);
        return () => clearInterval(interval);
    }, []);

    const filtered = (quotes.length > 0 ? quotes : []).filter(s => s.symbol.toLowerCase().includes(filter.toLowerCase()));

    return (
        <GlassCard className="h-full flex flex-col">
            <CardHeader className="flex flex-row items-center gap-2 p-3">
                <CardTitle className="text-xs uppercase text-text-muted font-bold">市场行情</CardTitle>
                <div className="flex-1" />
                <div className="w-24">
                    <Input
                        className="h-6 text-xs bg-bg-base border-none"
                        placeholder="搜索..."
                        value={filter}
                        onChange={e => setFilter(e.target.value)}
                    />
                </div>
            </CardHeader>

            <div className="flex-1 overflow-y-auto custom-scrollbar">
                {/* Table Header */}
                <div className="grid grid-cols-4 px-4 py-2 text-[10px] text-text-muted font-mono uppercase tracking-wider border-b border-white/5">
                    <span>品种</span>
                    <span className="text-right">买价</span>
                    <span className="text-right">卖价</span>
                    <span className="text-right">点差</span>
                </div>

                {/* Rows */}
                <div className="flex flex-col">
                    {filtered.length === 0 && (
                        <div className="p-4 text-center text-xs text-text-muted">无数据 / 连接中...</div>
                    )}
                    {filtered.map((item) => {
                        return (
                            <div
                                key={item.symbol}
                                className="grid grid-cols-4 px-4 py-2.5 hover:bg-white/5 cursor-pointer transition-colors border-b border-dashed border-white/5 last:border-0"
                                onClick={() => onSymbolSelect?.(item.symbol)}
                            >
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-bold text-text-primary">{item.symbol}</span>
                                </div>
                                <div className="text-right font-mono text-xs text-long">{item.bid}</div>
                                <div className="text-right font-mono text-xs text-short">{item.ask}</div>
                                <div className="text-right font-mono text-xs text-text-muted">
                                    {item.spread}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </GlassCard>
    );
}
