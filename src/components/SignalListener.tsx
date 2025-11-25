"use client";

import { useEffect, useState, useCallback } from 'react';
import { supabase } from '@/lib/supabase';
import { Toast } from '@/components/ui/Toast';
import { TradePanel } from '@/components/TradePanel';
import { ArrowRight } from 'lucide-react';

interface Signal {
    id: string;
    created_at: string;
    symbol: string;
    action: 'BUY' | 'SELL';
    price: number;
    sl: number;
    tp: number;
    status: string;
    source: string;
    comment?: string;
}

export function SignalListener() {
    const [latestSignal, setLatestSignal] = useState<Signal | null>(null);
    const [showToast, setShowToast] = useState(false);
    const [showTradePanel, setShowTradePanel] = useState(false);

    const handleNewSignal = useCallback((signal: Signal) => {
        setLatestSignal(signal);
        setShowToast(true);
        
        // Play sound
        try {
            const AC = (window as unknown as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext }).AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
            if (AC) {
                const ctx = new AC();
                const o = ctx.createOscillator();
                const g = ctx.createGain();
                o.type = 'sine';
                o.frequency.setValueAtTime(500, ctx.currentTime);
                o.frequency.exponentialRampToValueAtTime(1000, ctx.currentTime + 0.1);
                g.gain.setValueAtTime(0.1, ctx.currentTime);
                g.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                o.connect(g);
                g.connect(ctx.destination);
                o.start();
                o.stop(ctx.currentTime + 0.5);
            }
        } catch (e) {
            console.error('Audio play failed', e);
        }
    }, []);

    useEffect(() => {
        // Subscribe to new signals
        const channel = supabase
            .channel('realtime_signals')
            .on(
                'postgres_changes',
                { event: 'INSERT', schema: 'public', table: 'signals' },
                (payload) => {
                    const newSignal = payload.new as Signal;
                    if (newSignal.status === 'new') {
                        handleNewSignal(newSignal);
                    }
                }
            )
            .subscribe();

        return () => {
            supabase.removeChannel(channel);
        };
    }, [handleNewSignal]);

    const handleAction = () => {
        setShowToast(false);
        setShowTradePanel(true);
        
        // Optionally mark signal as 'viewed' or 'processed' in DB
        if (latestSignal) {
            supabase.from('signals').update({ status: 'processed' }).eq('id', latestSignal.id).then(() => {});
        }
    };

    if (!latestSignal) return null;

    return (
        <>
            {/* Toast Notification */}
            <Toast
                open={showToast}
                onOpenChange={setShowToast}
                title={`新信号: ${latestSignal.symbol}`}
                description={
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                            <span className={`font-bold ${latestSignal.action === 'BUY' ? 'text-accent-success' : 'text-accent-danger'}`}>
                                {latestSignal.action}
                            </span>
                            <span className="text-slate-400">@ {latestSignal.price}</span>
                        </div>
                        <div className="text-xs text-slate-500 flex gap-2">
                            <span>TP: {latestSignal.tp}</span>
                            <span>SL: {latestSignal.sl}</span>
                        </div>
                    </div>
                }
                action={
                    <button 
                        onClick={handleAction}
                        className="flex items-center gap-1 bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
                    >
                        下单 <ArrowRight size={14} />
                    </button>
                }
                duration={10000} // 10 seconds
            />

            {/* Trade Panel with Pre-filled Data */}
            {showTradePanel && (
                <TradePanel
                    open={showTradePanel}
                    onClose={() => setShowTradePanel(false)}
                    symbol={latestSignal.symbol}
                    initialSide={latestSignal.action}
                    initialPrice={latestSignal.price}
                    initialStopLoss={latestSignal.sl}
                    initialTakeProfit={latestSignal.tp}
                />
            )}
        </>
    );
}
