'use client';

import { useState } from 'react';
import { bridgeClient, TradeRequest } from '../lib/bridge-client';

export default function TradeControl() {
    const [symbol, setSymbol] = useState('EURUSD');
    const [volume, setVolume] = useState(0.01);
    const [status, setStatus] = useState<string>('');
    const [loading, setLoading] = useState(false);

    const handleTrade = async (action: 'BUY' | 'SELL') => {
        setLoading(true);
        setStatus('Sending command...');

        const trade: TradeRequest = {
            action,
            symbol,
            volume,
        };

        try {
            const result = await bridgeClient.executeTrade(trade);
            setStatus(`Success: ${JSON.stringify(result)}`);
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            setStatus(`Error: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4 border rounded-lg shadow-md bg-white dark:bg-gray-800">
            <h2 className="text-xl font-bold mb-4">Bridge Control</h2>

            <div className="flex gap-4 mb-4">
                <div>
                    <label className="block text-sm font-medium mb-1">Symbol</label>
                    <input
                        type="text"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value)}
                        className="p-2 border rounded w-24 text-black"
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium mb-1">Volume</label>
                    <input
                        type="number"
                        step="0.01"
                        value={volume}
                        onChange={(e) => setVolume(parseFloat(e.target.value))}
                        className="p-2 border rounded w-24 text-black"
                    />
                </div>
            </div>

            <div className="flex gap-4 mb-4">
                <button
                    onClick={() => handleTrade('BUY')}
                    disabled={loading}
                    className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                >
                    BUY
                </button>
                <button
                    onClick={() => handleTrade('SELL')}
                    disabled={loading}
                    className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
                >
                    SELL
                </button>
            </div>

            <div className="text-sm text-gray-600 dark:text-gray-400 break-all">
                {status}
            </div>
        </div>
    );
}
