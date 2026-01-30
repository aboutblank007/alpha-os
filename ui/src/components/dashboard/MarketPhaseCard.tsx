
import React from 'react';
import { Activity, Thermometer, Wind } from 'lucide-react';
import { useAlphaOS } from '../../context/AlphaOSContext';

const MarketPhaseCard: React.FC = () => {
    const { runtime } = useAlphaOS();

    // Default values if runtime is not yet available
    const temperature = runtime?.temperature || 0;
    const entropy = runtime?.entropy || 0;
    const phase = runtime?.market_phase || 'UNKNOWN';
    const tempWidth = temperature > 0 ? Math.max(temperature * 100, 1) : 0;
    const entropyWidth = entropy > 0 ? Math.max(entropy * 100, 1) : 0;

    // Determine color based on phase
    const getPhaseColor = (p: string) => {
        switch (p) {
            case 'LAMINAR': return 'text-emerald-400';
            case 'TURBULENT': return 'text-amber-400';
            case 'TRANSITION':
            case 'PHASE_TRANSITION':
                return 'text-primary';
            case 'FROZEN': return 'text-blue-400';
            default: return 'text-slate-400';
        }
    };

    return (
        <div className="glass-panel p-4 rounded-lg flex flex-col gap-4">
            <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2">
                    <Activity size={16} className="text-primary" />
                    MARKET PHASE (ts_phase)
                </h3>
                <span className={`text-xs font-bold px-2 py-1 rounded bg-slate-800/50 border border-slate-700/50 ${getPhaseColor(phase)}`}>
                    {phase}
                </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
                {/* Temperature */}
                <div className="bg-slate-900/30 p-2 rounded border border-slate-800">
                    <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
                        <Thermometer size={12} />
                        market_temperature
                    </div>
                    <div className="text-xl font-mono text-slate-200">
                        {temperature.toFixed(5)}
                    </div>
                    <div className="w-full bg-slate-800 h-1 mt-2 rounded-full overflow-hidden">
                        <div
                            className="bg-gradient-to-r from-blue-500 to-red-500 h-full transition-all duration-500"
                            style={{ width: `${Math.min(tempWidth, 100)}%` }}
                        />
                    </div>
                </div>

                {/* Entropy */}
                <div className="bg-slate-900/30 p-2 rounded border border-slate-800">
                    <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
                        <Wind size={12} />
                        market_entropy
                    </div>
                    <div className="text-xl font-mono text-slate-200">
                        {entropy.toFixed(5)}
                    </div>
                    <div className="w-full bg-slate-800 h-1 mt-2 rounded-full overflow-hidden">
                        <div
                            className="bg-purple-500 h-full transition-all duration-500"
                            style={{ width: `${Math.min(entropyWidth, 100)}%` }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MarketPhaseCard;
