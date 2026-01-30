import React from 'react';
import { Activity, TrendingUp, TrendingDown, Minus, Clock } from 'lucide-react';
import { useAlphaOS } from '../context/AlphaOSContext';

const SignalPanel = () => {
    const { decision, signalHistory } = useAlphaOS();

    // 当前信号
    const currentSignal = getSignalInfo(decision);

    return (
        <div className="glass-panel rounded-lg p-4">
            <h3 className="text-sm font-bold text-dim uppercase tracking-wider mb-3 flex items-center gap-2">
                <Activity size={14} />
                Signal Monitor
            </h3>

            {/* 当前信号卡片 */}
            <div className="bg-panel rounded-lg p-4 mb-4 border border-gray-800">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-dim text-xs uppercase">Current Signal</span>
                    <SignalBadge signal={currentSignal} />
                </div>

                {/* Meta-Labeling 概率条：信号正确/错误的概率 */}
                <div className="space-y-2 mb-4">
                    <ProbabilityBar 
                        label="Correct" 
                        value={decision.factors?.win_prob || 0}
                        color="success"
                    />
                    <ProbabilityBar 
                        label="Wrong" 
                        value={decision.factors?.loss_prob || 0}
                        color="danger"
                    />
                </div>

                {/* 市场热力学 */}
                <div className="space-y-3 pt-3 border-t border-gray-700">
                    <div className="flex justify-between items-center text-xs">
                        <span className="text-dim uppercase tracking-tighter">Market Phase</span>
                        <span className={`font-bold font-mono ${getPhaseColor(decision.market_phase || decision.ts_phase)}`}>
                            {decision.market_phase || decision.ts_phase || 'UNKNOWN'}
                        </span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2">
                        <div className="bg-dark/50 rounded p-2 border border-glass">
                            <div className="text-[10px] text-dim uppercase">Temp (T)</div>
                            <div className="text-sm font-bold text-warning font-mono">
                                {(decision.temperature || 0).toFixed(5)}
                            </div>
                        </div>
                        <div className="bg-dark/50 rounded p-2 border border-glass">
                            <div className="text-[10px] text-dim uppercase">Entropy (S)</div>
                            <div className="text-sm font-bold text-secondary font-mono">
                                {(decision.entropy || 0).toFixed(5)}
                            </div>
                        </div>
                    </div>
                </div>

                {/* 预测类别 */}
                <div className="mt-3 pt-3 border-t border-gray-700 text-xs">
                    <div className="flex justify-between text-dim">
                        <span>Confidence</span>
                        <span className="font-mono">{((decision.confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-dim mt-1">
                        <span>Score</span>
                        <span className="font-mono">{(decision.score || 0).toFixed(4)}</span>
                    </div>
                </div>
            </div>

            {/* 信号历史 */}
            <div>
                <h4 className="text-xs text-dim uppercase tracking-wider mb-2 flex items-center gap-1">
                    <Clock size={12} />
                    Recent Signals
                </h4>
                <div className="space-y-1 max-h-48 overflow-y-auto">
                    {signalHistory.length === 0 ? (
                        <div className="text-dim text-xs text-center py-4">
                            No signals yet
                        </div>
                    ) : (
                        signalHistory.slice(0, 10).map((sig, index) => (
                            <SignalHistoryItem key={index} signal={sig} />
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

// 概率条组件
const ProbabilityBar = ({ label, value, color }) => {
    const percent = Math.max(0, Math.min(100, value * 100));
    const colorClass = {
        success: 'bg-success',
        danger: 'bg-danger',
        warning: 'bg-warning',
    }[color] || 'bg-primary';

    return (
        <div className="flex items-center gap-2">
            <span className="text-xs text-dim w-12">{label}</span>
            <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div 
                    className={`h-full ${colorClass} transition-all duration-300`}
                    style={{ width: `${percent}%` }}
                />
            </div>
            <span className="text-xs font-mono w-12 text-right">
                {percent.toFixed(1)}%
            </span>
        </div>
    );
};

// 信号徽章
const SignalBadge = ({ signal }) => {
    const { label, color, Icon } = signal;
    
    return (
        <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-bold ${color}`}>
            <Icon size={12} />
            {label}
        </div>
    );
};

// 信号历史项
const SignalHistoryItem = ({ signal }) => {
    const info = getSignalInfo(signal);
    const time = signal.timestamp ? new Date(signal.timestamp).toLocaleTimeString() : '--:--:--';
    const winProb = ((signal.factors?.win_prob || signal.entry_prob || 0) * 100).toFixed(1);

    return (
        <div className="flex items-center justify-between text-xs py-1 px-2 bg-panel rounded">
            <span className="text-dim font-mono">{time}</span>
            <span className={info.color}>{info.label}</span>
            <span className="font-mono text-dim">{winProb}%</span>
        </div>
    );
};

// 获取信号信息
function getSignalInfo(decision) {
    const cls = decision?.factors?.predicted_class;
    
    if (cls === 2) {
        return { label: 'WIN', color: 'text-success', Icon: TrendingUp };
    }
    if (cls === 0) {
        return { label: 'LOSS', color: 'text-danger', Icon: TrendingDown };
    }
    return { label: 'NEUTRAL', color: 'text-warning', Icon: Minus };
}

// 获取类别名称
function getClassName(cls) {
    if (cls === 0) return 'Loss (0)';
    if (cls === 1) return 'Neutral (1)';
    if (cls === 2) return 'Win (2)';
    return 'Unknown';
}

// 获取相位颜色
function getPhaseColor(phase) {
    const colors = {
        'LAMINAR': 'text-blue-400',
        'TURBULENT': 'text-purple-400',
        'PHASE_TRANSITION': 'text-orange-400',
        'FROZEN': 'text-gray-400',
        'UNKNOWN': 'text-dim',
    };
    return colors[phase] || colors.UNKNOWN;
}

export default SignalPanel;
