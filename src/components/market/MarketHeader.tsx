import React from 'react';
import { StatusBadge } from '@/components/ui/StatusBadge';

interface MarketHeaderProps {
  isConnected: boolean;
  symbolCount: number;
}

export function MarketHeader({ isConnected, symbolCount }: MarketHeaderProps) {
  return (
    <div className="flex items-center justify-between p-4 border-b border-surface-border bg-[#1e222d]">
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
  );
}

