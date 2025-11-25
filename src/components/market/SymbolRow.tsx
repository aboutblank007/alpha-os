import React, { memo, useState } from 'react';
import { cn } from '@/lib/utils';
import { TradePanel } from '@/components/TradePanel';

export const SYMBOL_DESCRIPTIONS: Record<string, string> = {
    'EURUSD': 'Euro vs US Dollar',
    'GBPUSD': 'Great Britain Pound vs US Dollar',
    'USDJPY': 'US Dollar vs Japanese Yen',
    'XAUUSD': 'Gold vs US Dollar',
    'BTCUSD': 'Bitcoin vs US Dollar',
    'GER40': 'DAX Index',
    'US30': 'Dow Jones Index',
    'US500': 'S&P 500 Index',
    'USTEC': 'Nasdaq 100 Index',
};

interface SymbolRowProps {
    symbol: string;
    bid?: number;
    ask?: number;
    isConnected: boolean;
    onSelect: (symbol: string) => void;
}

const renderPrice = (price: number | undefined, symbol: string) => {
    if (price === undefined) return <span className="text-lg">---</span>;

    let digits = 5;
    if (symbol.includes('JPY')) {
        digits = 3;
    } else if (symbol.includes('XAU') || symbol.includes('BTC') || symbol.includes('ETH')) {
        digits = 2;
    } else if (['GER40', 'US30', 'US500', 'USTEC'].some(s => symbol.includes(s))) {
        digits = 2;
    }

    const priceStr = price.toFixed(digits);

    const main = priceStr.slice(0, -1);
    const pipette = priceStr.slice(-1);

    return (
        <div className="flex items-baseline justify-center leading-none pointer-events-none w-full">
            <span style={{ fontSize: '16cqw' }} className="font-bold tracking-tighter">{main}</span>
            <span style={{ fontSize: '10cqw' }} className="font-bold align-top ml-[0.1em] -mt-[0.1em]">{pipette}</span>
        </div>
    );
};

export const SymbolRow = memo(function SymbolRow({
    symbol,
    bid,
    ask,
    isConnected,
    onSelect
}: SymbolRowProps) {
    const [showTradePanel, setShowTradePanel] = useState(false);
    const [tradeSide, setTradeSide] = useState<'BUY' | 'SELL'>('BUY');

    const spread = bid && ask ? ((ask - bid) * (symbol.includes('JPY') ? 100 : (symbol.includes('BTC') || symbol.includes('ETH')) ? 1 : symbol.includes('XAU') ? 100 : 10000)).toFixed(1) : '-';
    const cleanSymbol = symbol.replace('_', '').replace('/', '');
    const desc = SYMBOL_DESCRIPTIONS[cleanSymbol] || cleanSymbol;

    return (
        <>
            <div
                className="flex items-center justify-between p-2 border-b border-surface-border hover:bg-white/5 transition-colors cursor-pointer group gap-2 relative w-full"
                onClick={() => onSelect(symbol)}
                style={{ containerType: 'inline-size' }}
            >
                {/* Left: Info */}
                <div className="flex flex-col gap-[0.5cqw] min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                        <span style={{ fontSize: '4cqw' }} className="font-bold text-white group-hover:text-accent-primary transition-colors truncate leading-tight min-text-[14px] max-text-[20px]">{symbol}</span>
                    </div>
                    <span style={{ fontSize: '2.5cqw' }} className="text-slate-500 truncate block leading-tight min-text-[10px] max-text-[14px]">{desc}</span>
                    <span style={{ fontSize: '2.5cqw' }} className="text-slate-600 font-mono truncate block leading-tight min-text-[10px] max-text-[14px]">{new Date().toLocaleTimeString()}</span>
                </div>

                {/* Right: Prices */}
                <div className="flex gap-1 xl:gap-1.5 items-center relative shrink-0 w-[45%] max-w-[240px] min-w-[140px]">
                    {/* Spread Badge */}
                    <div className="absolute -top-2 left-1/2 -translate-x-1/2 bg-slate-700/80 backdrop-blur-sm text-[8px] px-1 rounded text-slate-300 z-10 border border-surface-border shadow-sm">
                        {spread}
                    </div>

                    {/* Bid Box (Sell) */}
                    <button
                        className={cn(
                            "flex flex-col items-center justify-center flex-1 aspect-[2/1] rounded transition-all relative overflow-hidden mt-1",
                            'bg-[#ff5252] hover:brightness-110 active:scale-95'
                        )}
                        style={{ containerType: 'size' }}
                        onClick={(e) => {
                            e.stopPropagation();
                            setTradeSide('SELL');
                            setShowTradePanel(true);
                        }}
                        disabled={!isConnected}
                    >
                        <>
                            <div className="text-white leading-none drop-shadow-md w-full">{renderPrice(bid, symbol)}</div>
                            <div style={{ fontSize: '8cqw' }} className="font-bold text-white/60 uppercase tracking-wider mt-[2cqw]">Sell</div>
                        </>
                    </button>

                    {/* Ask Box (Buy) */}
                    <button
                        className={cn(
                            "flex flex-col items-center justify-center flex-1 aspect-[2/1] rounded transition-all relative overflow-hidden mt-1",
                            'bg-[#00bfa5] hover:brightness-110 active:scale-95'
                        )}
                        style={{ containerType: 'size' }}
                        onClick={(e) => {
                            e.stopPropagation();
                            setTradeSide('BUY');
                            setShowTradePanel(true);
                        }}
                        disabled={!isConnected}
                    >
                        <>
                            <div className="text-white leading-none drop-shadow-md w-full">{renderPrice(ask, symbol)}</div>
                            <div style={{ fontSize: '8cqw' }} className="font-bold text-white/60 uppercase tracking-wider mt-[2cqw]">Buy</div>
                        </>
                    </button>
                </div>
            </div>

            <TradePanel
                open={showTradePanel}
                onClose={() => setShowTradePanel(false)}
                symbol={symbol}
                initialSide={tradeSide}
            />
        </>
    );
});

