import React from 'react';
import FlowChart from '../components/FlowChart';
import PositionPanel from '../components/PositionPanel';
import MarketPhaseCard from '../components/dashboard/MarketPhaseCard';
import HealthMonitor from '../components/dashboard/HealthMonitor';
import SignalFeed from '../components/dashboard/SignalFeed';
import { useAlphaOS } from '../context/AlphaOSContext';
import useRuntimeMetrics from '../hooks/useRuntimeMetrics';

const LivePage: React.FC = () => {
    const { runtime } = useAlphaOS();
    const { tickRate, ageSeconds } = useRuntimeMetrics(runtime);

    return (
        <div className="flex-1 flex overflow-hidden">
            {/* Main Flow Chart Area */}
            <div className="flex-1 relative border-r border-border-highlight/30">
                <FlowChart />

                {/* Floating Overlay for Critical Stats or Ticker? Optional */}
            </div>

            {/* Right Control Panel */}
            <div className="w-96 flex flex-col gap-4 p-4 overflow-y-auto bg-bg-panel/30 backdrop-blur-sm">
                <MarketPhaseCard />

                <div className="grid grid-cols-2 gap-2">
                    <HealthMonitor />
                    {/* Placeholder for Mini Chart or specific Stat */}
                    <div className="glass-panel p-4 rounded-lg flex flex-col gap-2">
                        <div className="flex items-center justify-between text-xs text-slate-500">
                            <span>Tick Rate</span>
                            <span className="font-mono text-slate-300">
                                {tickRate > 0 ? `${tickRate.toFixed(1)}/s` : '--'}
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-500">
                            <span>Open Positions</span>
                            <span className="font-mono text-slate-300">{runtime?.open_positions ?? 0}</span>
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-500">
                            <span>Last Update</span>
                            <span className="font-mono text-slate-300">
                                {ageSeconds !== null ? `${ageSeconds.toFixed(1)}s` : '--'}
                            </span>
                        </div>
                        <div className="mt-1 h-1 w-full rounded-full bg-slate-800 overflow-hidden">
                            <div
                                className="h-full bg-primary transition-all duration-500"
                                style={{ width: `${Math.min((runtime?.warmup_progress || 0) * 100, 100)}%` }}
                            />
                        </div>
                        <div className="text-[10px] text-slate-500 flex justify-between">
                            <span>Warmup</span>
                            <span>{((runtime?.warmup_progress || 0) * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                </div>

                <SignalFeed />

                <div className="flex-1 min-h-[200px] flex flex-col">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Open Positions (ExitPolicyV21)</h3>
                    <div className="flex-1 glass-panel rounded-lg overflow-hidden">
                        <PositionPanel />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LivePage;
