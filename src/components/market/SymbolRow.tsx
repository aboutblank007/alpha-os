import React, { memo } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

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
    executingSide: 'BUY' | 'SELL' | null;
    onTrade: (symbol: string, side: 'BUY' | 'SELL', e: React.MouseEvent) => void;
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
        <div className="flex items-baseline justify-center leading-none pointer-events-none">
            <span className="text-2xl font-bold tracking-tighter">{main}</span>
            <span className="text-sm font-bold align-top -mt-1 ml-0.5">{pipette}</span>
        </div>
    );
};

export const SymbolRow = memo(function SymbolRow({
    symbol,
    bid,
    ask,
    isConnected,
    executingSide,
    onTrade,
    onSelect
}: SymbolRowProps) {
    const spread = bid && ask ? ((ask - bid) * (symbol.includes('JPY') || symbol.includes('XAU') ? 100 : 10000)).toFixed(1) : '-';
    const cleanSymbol = symbol.replace('_', '').replace('/', '');
    const desc = SYMBOL_DESCRIPTIONS[cleanSymbol] || cleanSymbol;

    // Determine color class for buttons
    // Note: In original MarketWatch, Sell was red (#ff5252), Buy was teal (#00bfa5).
    // We'll stick to those or use theme colors. Theme: danger/success.
    // Original: bg-[#ff5252] / bg-[#00bfa5]
    
    return (
        <div
            className="flex items-center justify-between p-3 border-b border-surface-border hover:bg-white/5 transition-colors cursor-pointer group gap-2"
            onClick={() => onSelect(symbol)}
        >
            {/* Left: Info */}
            <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                <div className="flex items-center gap-2">
                    <span className="text-sm xl:text-base font-bold text-white group-hover:text-accent-primary transition-colors truncate">{symbol}</span>
                </div>
                <span className="text-[9px] xl:text-[10px] text-slate-500 truncate block">{desc}</span>
                <span className="text-[9px] xl:text-[10px] text-slate-600 font-mono truncate block">{new Date().toLocaleTimeString()}</span>
            </div>

            {/* Right: Prices */}
            <div className="flex gap-1 xl:gap-1.5 items-center relative shrink-0">
                {/* Spread Badge */}
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-slate-700/80 backdrop-blur-sm text-[9px] px-1.5 rounded text-slate-300 z-10 border border-surface-border shadow-sm">
                    {spread}
                </div>

                {/* Bid Box (Sell) */}
                <button
                    className={cn(
                        "flex flex-col items-center justify-center w-[5.5rem] h-12 xl:w-28 xl:h-14 rounded-md transition-all relative overflow-hidden",
                        executingSide === 'SELL' 
                            ? 'bg-slate-700 cursor-wait' 
                            : 'bg-[#ff5252] hover:brightness-110 active:scale-95'
                    )}
                    onClick={(e) => onTrade(symbol, 'SELL', e)}
                    disabled={!isConnected || !!executingSide}
                >
                    {executingSide === 'SELL' ? (
                        <Loader2 className="animate-spin text-white" size={20} />
                    ) : (
                        <>
                            <div className="text-white leading-none drop-shadow-md scale-90 xl:scale-100 origin-center">{renderPrice(bid, symbol)}</div>
                            <div className="text-[8px] xl:text-[9px] font-bold text-white/60 uppercase tracking-wider mt-0.5">Sell</div>
                        </>
                    )}
                </button>

                {/* Ask Box (Buy) */}
                <button
                    className={cn(
                        "flex flex-col items-center justify-center w-[5.5rem] h-12 xl:w-28 xl:h-14 rounded-md transition-all relative overflow-hidden",
                        executingSide === 'BUY' 
                            ? 'bg-slate-700 cursor-wait' 
                            : 'bg-[#00bfa5] hover:brightness-110 active:scale-95'
                    )}
                    onClick={(e) => onTrade(symbol, 'BUY', e)}
                    disabled={!isConnected || !!executingSide}
                >
                    {executingSide === 'BUY' ? (
                        <Loader2 className="animate-spin text-white" size={20} />
                    ) : (
                        <>
                            <div className="text-white leading-none drop-shadow-md scale-90 xl:scale-100 origin-center">{renderPrice(ask, symbol)}</div>
                            <div className="text-[8px] xl:text-[9px] font-bold text-white/60 uppercase tracking-wider mt-0.5">Buy</div>
                        </>
                    )}
                </button>
            </div>
        </div>
    );
});

