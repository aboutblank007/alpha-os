"use client";
import React from 'react';
import { useMarketStore } from '@/store/useMarketStore';
import { MarketHeader } from '@/components/market/MarketHeader';
import { SymbolRow } from '@/components/market/SymbolRow';

interface MarketWatchProps {
    // Optional prop for overrides, but we mainly use store
    isConnected?: boolean;
    onTrade?: (symbol: string, side: 'BUY' | 'SELL') => Promise<void> | void;
    onSymbolSelect?: (symbol: string) => void;
    variant?: 'default' | 'focus';
}

export function MarketWatch({ isConnected: propIsConnected, onSymbolSelect, variant = 'default' }: MarketWatchProps) {
    const bridgeConnected = useMarketStore(state => state.isConnected);
    const activeSymbols = useMarketStore(state => state.activeSymbols);
    const symbolPrices = useMarketStore(state => state.symbolPrices);

    // Use bridge connection status if prop is not provided or overrides
    // If propIsConnected is provided, we respect it (maybe for testing or manual override), 
    // otherwise default to store status.
    const isConnected = propIsConnected !== undefined ? (propIsConnected && bridgeConnected) : bridgeConnected;

    if (variant === 'focus') {
        return (
            <div className="h-full flex flex-col bg-transparent">
                {/* Simplified Header for Focus Mode */}
                <div className="px-4 py-2 border-b border-white/5 flex justify-between items-center bg-white/5 backdrop-blur-sm">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">市场监控</span>
                    <div className={`h-1.5 w-1.5 rounded-full ${isConnected ? 'bg-accent-success shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-slate-600'}`} />
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar p-0">
                     {activeSymbols.map((symbol) => {
                        const priceData = symbolPrices?.[symbol];
                        const bid = priceData?.bid ?? 0;
                        const ask = priceData?.ask ?? 0;
                        return (
                            <SymbolRow
                                key={symbol}
                                symbol={symbol}
                                bid={bid}
                                ask={ask}
                                isConnected={!!isConnected}
                                onSelect={onSymbolSelect || (() => { })}
                                compact={true} // Pass compact prop if SymbolRow supports it, otherwise it just renders normally
                            />
                        );
                    })}
                </div>
            </div>
        );
    }

    return (
        <div className="glass-panel p-0 rounded-xl h-full flex flex-col overflow-hidden bg-[#1e222d]">
            <MarketHeader isConnected={!!isConnected} symbolCount={activeSymbols.length} />

            {/* Symbol List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar bg-[#1e222d] overflow-x-hidden min-h-0">
                {activeSymbols.length === 0 ? (
                    <div className="text-center text-slate-500 py-10 text-sm">
                        {isConnected ? 'Waiting for EA data...' : 'Bridge Disconnected'}
                    </div>
                ) : (
                    activeSymbols.map((symbol) => {
                        const priceData = symbolPrices?.[symbol];
                        const bid = priceData?.bid ?? 0;
                        const ask = priceData?.ask ?? 0;

                        return (
                            <SymbolRow
                                key={symbol}
                                symbol={symbol}
                                bid={bid}
                                ask={ask}
                                isConnected={!!isConnected}
                                onSelect={onSymbolSelect || (() => { })}
                            />
                        );
                    })
                )}
            </div>
        </div>
    );
}
