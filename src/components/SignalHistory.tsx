"use client";

import { useSignalStore, Signal } from '@/store/useSignalStore';
import { X, Trash2, ArrowRight, Clock, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { useEffect, useRef } from 'react';

interface SignalHistoryProps {
    onSelectSignal: (signal: Signal) => void;
}

export function SignalHistory({ onSelectSignal }: SignalHistoryProps) {
    const { signals, isHistoryOpen, setHistoryOpen, markAllAsRead, clearHistory } = useSignalStore();
    const panelRef = useRef<HTMLDivElement>(null);

    // Close on click outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (panelRef.current && !panelRef.current.contains(event.target as Node) && isHistoryOpen) {
                // Check if the click was on the toggle button (usually in header) - strictly we can just close
                // But to avoid immediate re-opening if the toggle button is clicked, we might need more logic.
                // For now, let's trust the overlay approach or just close.
                setHistoryOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [isHistoryOpen, setHistoryOpen]);

    // Mark as read when opened
    useEffect(() => {
        if (isHistoryOpen) {
            markAllAsRead();
        }
    }, [isHistoryOpen, markAllAsRead]);

    if (!isHistoryOpen) return null;

    return (
        <>
            {/* Backdrop */}
            <div 
                className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
                onClick={() => setHistoryOpen(false)}
            />
            
            {/* Slide-over Panel */}
            <div 
                ref={panelRef}
                className={cn(
                    "fixed inset-y-0 right-0 z-50 w-full max-w-sm bg-surface-glass-strong backdrop-blur-xl border-l border-white/10 shadow-2xl transform transition-transform duration-300 ease-in-out",
                    isHistoryOpen ? "translate-x-0" : "translate-x-full"
                )}
            >
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
                        <div>
                            <h2 className="text-lg font-semibold text-white">信号历史</h2>
                            <p className="text-xs text-slate-400">最近收到的交易信号</p>
                        </div>
                        <div className="flex items-center gap-2">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={clearHistory}
                                className="h-8 w-8 p-0 text-slate-400 hover:text-red-400 hover:bg-white/5"
                                title="清空历史"
                            >
                                <Trash2 size={16} />
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setHistoryOpen(false)}
                                className="h-8 w-8 p-0 text-slate-400 hover:text-white hover:bg-white/5"
                            >
                                <X size={20} />
                            </Button>
                        </div>
                    </div>

                    {/* List */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                        {signals.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-full text-slate-500">
                                <Clock size={48} className="mb-4 opacity-20" />
                                <p>暂无信号记录</p>
                            </div>
                        ) : (
                            signals.map((signal) => (
                                <div 
                                    key={signal.id}
                                    onClick={() => {
                                        onSelectSignal(signal);
                                        // Optional: close history on select?
                                        // setHistoryOpen(false); 
                                    }}
                                    className="group relative flex flex-col gap-2 p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 transition-all cursor-pointer"
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <span className={cn(
                                                "flex items-center justify-center w-6 h-6 rounded-full bg-opacity-20",
                                                signal.action === 'BUY' ? "bg-accent-success text-accent-success" : "bg-accent-danger text-accent-danger"
                                            )}>
                                                {signal.action === 'BUY' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                            </span>
                                            <span className="font-bold text-white">{signal.symbol}</span>
                                        </div>
                                        <span className="text-xs text-slate-500 font-mono">
                                            {new Date(signal.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </span>
                                    </div>
                                    
                                    <div className="flex items-center justify-between text-sm">
                                        <div className="flex flex-col">
                                            <span className="text-xs text-slate-500">价格</span>
                                            <span className="font-mono text-slate-200">{signal.price}</span>
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-xs text-slate-500">TP</span>
                                            <span className="font-mono text-accent-success">{signal.tp}</span>
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <span className="text-xs text-slate-500">SL</span>
                                            <span className="font-mono text-accent-danger">{signal.sl}</span>
                                        </div>
                                    </div>

                                    {/* Action Hover */}
                                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-transparent to-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl flex items-center justify-end pr-4 pointer-events-none">
                                        <ArrowRight size={20} className="text-white/80 transform translate-x-2 group-hover:translate-x-0 transition-transform" />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}

