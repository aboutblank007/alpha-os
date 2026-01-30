
import React from 'react';
import { Target } from 'lucide-react';
import { useAlphaOS } from '../../context/AlphaOSContext';

const SignalFeed: React.FC = () => {
    // Note: In a real implementation, we would keep a history of signals in the context store
    // For now, we visualize the current decision
    const { decision, runtime } = useAlphaOS();
    const confidence = Number.isFinite(decision?.meta_confidence)
        ? decision.meta_confidence
        : Number.isFinite(decision?.confidence)
            ? decision.confidence
            : 0;

    return (
        <div className="glass-panel p-4 rounded-lg flex flex-col gap-3 min-h-[150px]">
            <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2">
                <Target size={16} className="text-secondary" />
                PRIMARY SIGNAL / META FILTER
            </h3>

            <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                {decision?.signal && decision.signal !== 'HOLD' ? (
                    <div className="bg-slate-800/50 p-2 rounded border-l-2 border-primary flex justify-between items-center animate-pulse">
                        <div>
                            <div className="text-xs font-bold text-primary">Primary: {decision.signal}</div>
                            <div className="text-[10px] text-slate-400 font-mono">{decision.timestamp ? new Date(decision.timestamp * 1000).toLocaleTimeString() : '--:--:--'}</div>
                        </div>
                        <div className="text-right">
                            <div className="text-xs text-slate-300">meta_conf: {(confidence * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                ) : (
                    <div className="h-full flex items-center justify-center text-xs text-slate-600 italic">
                        {runtime?.market_phase
                            ? `Phase: ${runtime.market_phase} · Temp ${runtime.temperature?.toFixed(2)} · Ent ${runtime.entropy?.toFixed(2)}`
                            : 'No active signals...'}
                    </div>
                )}

                {/* Placeholder for history list */}
            </div>
        </div>
    );
};

export default SignalFeed;
