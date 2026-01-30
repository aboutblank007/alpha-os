
import React from 'react';
import { Shield, Zap, Database } from 'lucide-react';
import { useAlphaOS } from '../../context/AlphaOSContext';

const HealthMonitor: React.FC = () => {
    const { runtime, connectionState } = useAlphaOS();

    return (
        <div className="glass-panel p-4 rounded-lg flex flex-col gap-3">
            <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2">
                <Shield size={16} className="text-emerald-400" />
                RUNTIME HEALTH
            </h3>

            <div className="space-y-3">
                {/* Connection Status */}
                <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">WSRuntime</span>
                    <div className="flex items-center gap-2">
                        <span className={`status-dot ${connectionState === 'connected' ? 'active' : 'error'}`} />
                        <span className="text-slate-300 font-mono text-xs uppercase">{connectionState}</span>
                    </div>
                </div>

                {/* Guardian Status */}
                <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">ModelGuardian</span>
                    <span className={`font-mono text-xs px-2 py-0.5 rounded ${runtime?.guardian_halt
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        }`}>
                        {runtime?.guardian_halt ? 'HALTED' : 'ACTIVE'}
                    </span>
                </div>

                {/* Ticks */}
                <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500 flex items-center gap-1">
                        <Zap size={12} /> ticks_total
                    </span>
                    <span className="font-mono text-slate-200">
                        {runtime?.ticks_total?.toLocaleString() ?? 0}
                    </span>
                </div>

                {/* Warmup Progress */}
                <div className="space-y-1">
                    <div className="flex justify-between items-center text-xs text-slate-500">
                        <span className="flex items-center gap-1"><Database size={12} /> Confidence Gate Warmup</span>
                        <span>{((runtime?.warmup_progress || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                        <div
                            className="bg-primary h-full transition-all duration-500"
                            style={{ width: `${(runtime?.warmup_progress || 0) * 100}%` }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HealthMonitor;
