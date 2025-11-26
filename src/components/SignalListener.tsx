"use client";

import { useEffect, useState, useCallback } from 'react';
import { supabase } from '@/lib/supabase';
import { Toast } from '@/components/ui/Toast';
import { TradePanel } from '@/components/TradePanel';
import { ArrowRight } from 'lucide-react';
import { useSignalStore, Signal } from '@/store/useSignalStore';
import { SignalHistory } from '@/components/SignalHistory';

export function SignalListener() {
    const { addSignal, setSignals } = useSignalStore();
    const [toastSignal, setToastSignal] = useState<Signal | null>(null);
    const [showToast, setShowToast] = useState(false);
    const [showTradePanel, setShowTradePanel] = useState(false);
    const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);

    const handleNewSignal = useCallback((signal: Signal) => {
        addSignal(signal);
        setToastSignal(signal);
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
    }, [addSignal]);

    // Initial fetch and subscription
    useEffect(() => {
        // Fetch recent signals
        const fetchRecent = async () => {
            const { data } = await supabase
                .from('signals')
                .select('*')
                .order('created_at', { ascending: false })
                .limit(50);
            
            if (data) {
                setSignals(data as Signal[]);
            }
        };
        
        fetchRecent();

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
    }, [handleNewSignal, setSignals]);

    const handleToastAction = () => {
        if (toastSignal) {
            setShowToast(false);
            handleSelectSignal(toastSignal);
        }
    };

    const handleSelectSignal = (signal: Signal) => {
        setSelectedSignal(signal);
        setShowTradePanel(true);
        
        // Optionally mark signal as 'viewed' or 'processed' in DB
        // We do this when opening the trade panel
        supabase.from('signals').update({ status: 'processed' }).eq('id', signal.id).then(() => {});
    };

    return (
        <>
            {/* Toast Notification */}
            {toastSignal && (
                <Toast
                    open={showToast}
                    onOpenChange={setShowToast}
                    title={`新信号: ${toastSignal.symbol}`}
                    description={
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <span className={`font-bold ${toastSignal.action === 'BUY' ? 'text-accent-success' : 'text-accent-danger'}`}>
                                    {toastSignal.action}
                                </span>
                                <span className="text-slate-400">@ {toastSignal.price}</span>
                            </div>
                            <div className="text-xs text-slate-500 flex gap-2">
                                <span>TP: {toastSignal.tp}</span>
                                <span>SL: {toastSignal.sl}</span>
                            </div>
                        </div>
                    }
                    action={
                        <button 
                            onClick={handleToastAction}
                            className="flex items-center gap-1 bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
                        >
                            下单 <ArrowRight size={14} />
                        </button>
                    }
                    duration={10000} // 10 seconds
                />
            )}

            {/* Signal History Panel */}
            <SignalHistory onSelectSignal={handleSelectSignal} />

            {/* Trade Panel */}
            {showTradePanel && selectedSignal && (
                <TradePanel
                    open={showTradePanel}
                    onClose={() => setShowTradePanel(false)}
                    symbol={selectedSignal.symbol}
                    initialSide={selectedSignal.action}
                    // initialPrice removed as requested
                    initialStopLoss={selectedSignal.sl}
                    initialTakeProfit={selectedSignal.tp}
                />
            )}
        </>
    );
}
