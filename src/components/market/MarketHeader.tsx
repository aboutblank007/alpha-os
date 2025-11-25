import React from 'react';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { useTradeStore } from '@/store/useTradeStore';

interface MarketHeaderProps {
    isConnected: boolean;
    symbolCount: number;
}

export function MarketHeader({ isConnected, symbolCount }: MarketHeaderProps) {
    const account = useTradeStore(state => state.account);

    return (
        <div className="flex flex-col p-4 border-b border-surface-border bg-[#1e222d] gap-3">
            <div className="flex items-center justify-between">
                <StatusBadge
                    variant={isConnected ? "profit" : "loss"}
                    pulse={isConnected}
                    className="tracking-wide font-bold"
                >
                    {isConnected ? 'MT5 LIVE' : 'OFFLINE'}
                </StatusBadge>
                <div className="text-xs text-slate-500">
                    {symbolCount} Symbols
                </div>
            </div>

            {account && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                        <span className="text-slate-500">Balance</span>
                        <span className="text-slate-200 font-mono">{account.balance?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-500">Equity</span>
                        <span className={`font-mono font-medium ${account.equity >= account.balance ? 'text-accent-success' : 'text-accent-danger'}`}>
                            {account.equity?.toFixed(2)}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-500">Margin</span>
                        <span className="text-slate-200 font-mono">{account.margin?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-500">Free</span>
                        <span className="text-slate-200 font-mono">{account.free_margin?.toFixed(2)}</span>
                    </div>
                </div>
            )}
        </div>
    );
}
