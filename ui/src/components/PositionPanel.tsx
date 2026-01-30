import React from 'react';
import { TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';
import { useAlphaOS } from '../context/AlphaOSContext';

const PositionPanel = () => {
    const { positions, tick, runtime } = useAlphaOS();

    if (positions.length === 0) {
        const openCount = runtime?.open_positions ?? 0;
        return (
            <div className="glass-panel rounded-lg p-4">
                <h3 className="text-sm font-bold text-dim uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Target size={14} />
                    Open Positions (ExitPolicyV21)
                </h3>
                <div className="flex items-center justify-center py-8 text-dim">
                    <AlertCircle size={16} className="mr-2" />
                    {openCount > 0 ? `open_positions: ${openCount}` : 'No Open Positions'}
                </div>
                {openCount > 0 && (
                    <div className="text-xs text-slate-500 text-center">
                        Waiting for MT5 position stream (ZMQ / MT5 EA)...
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="glass-panel rounded-lg p-4">
            <h3 className="text-sm font-bold text-dim uppercase tracking-wider mb-3 flex items-center gap-2">
                <Target size={14} />
                Open Positions (ExitPolicyV21) ({positions.length})
            </h3>
            
            <div className="space-y-3">
                {positions.map((pos, index) => (
                    <PositionCard key={index} position={pos} currentPrice={tick.bid} />
                ))}
            </div>

            {/* 汇总 */}
            <div className="mt-4 pt-3 border-t border-gray-700">
                <div className="flex justify-between text-sm">
                    <span className="text-dim">Total P&L</span>
                    <TotalPnL positions={positions} />
                </div>
            </div>
        </div>
    );
};

const PositionCard = ({ position, currentPrice }) => {
    const isLong = position.direction === 'LONG';
    const pnl = position.unrealized_pnl || 0;
    const pnlColor = pnl >= 0 ? 'text-success' : 'text-danger';
    
    // 计算盈亏百分比（基于入场价）
    const pnlPercent = position.entry_price > 0 
        ? ((currentPrice - position.entry_price) / position.entry_price * 100 * (isLong ? 1 : -1))
        : 0;

    return (
        <div className="bg-panel rounded p-3 border border-gray-800">
            {/* 顶部：方向 + P&L */}
            <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                    {isLong ? (
                        <TrendingUp size={16} className="text-success" />
                    ) : (
                        <TrendingDown size={16} className="text-danger" />
                    )}
                    <span className={`font-bold ${isLong ? 'text-success' : 'text-danger'}`}>
                        {position.direction}
                    </span>
                    <span className="text-dim text-sm">
                        {position.volume?.toFixed(2)} lots
                    </span>
                </div>
                <span className={`font-mono font-bold ${pnlColor}`}>
                    {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                </span>
            </div>

            {/* 价格信息 */}
            <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                    <span className="text-dim">Entry</span>
                    <div className="font-mono">{position.entry_price?.toFixed(2) || '----'}</div>
                </div>
                <div>
                    <span className="text-dim">Current</span>
                    <div className="font-mono">{currentPrice?.toFixed(2) || '----'}</div>
                </div>
            </div>

            {/* SL/TP */}
            {(position.stop_loss || position.take_profit) && (
                <div className="grid grid-cols-2 gap-2 text-xs mt-2 pt-2 border-t border-gray-700">
                    <div>
                        <span className="text-danger">SL</span>
                        <div className="font-mono text-dim">
                            {position.stop_loss?.toFixed(2) || '----'}
                        </div>
                    </div>
                    <div>
                        <span className="text-success">TP</span>
                        <div className="font-mono text-dim">
                            {position.take_profit?.toFixed(2) || '----'}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

const TotalPnL = ({ positions }) => {
    const total = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);
    const color = total >= 0 ? 'text-success' : 'text-danger';
    
    return (
        <span className={`font-mono font-bold ${color}`}>
            {total >= 0 ? '+' : ''}{total.toFixed(2)}
        </span>
    );
};

export default PositionPanel;
