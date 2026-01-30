import React from 'react';
import { TrendingUp, TrendingDown, Minus, Target, Activity } from 'lucide-react';
import { useAlphaOS } from '../context/AlphaOSContext';

const Footer = () => {
    const { positions, decision, tick } = useAlphaOS();

    // 计算总 P&L
    const totalUnrealizedPnL = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);
    const totalRealizedPnL = positions.reduce((sum, p) => sum + (p.realized_pnl || 0), 0);
    const totalVolume = positions.reduce((sum, p) => sum + (p.volume || 0), 0);

    // 获取主要持仓方向
    const mainDirection = positions.length > 0 ? positions[0].direction : null;

    // P&L 颜色
    const pnlColor = (pnl) => {
        if (pnl > 0) return 'text-success';
        if (pnl < 0) return 'text-danger';
        return 'text-dim';
    };

    // P&L 图标
    const PnLIcon = ({ pnl }) => {
        if (pnl > 0) return <TrendingUp size={14} className="ml-1" />;
        if (pnl < 0) return <TrendingDown size={14} className="ml-1" />;
        return <Minus size={14} className="ml-1" />;
    };

    // 格式化概率
    const formatProb = (prob) => {
        if (typeof prob !== 'number') return '0.00';
        return (prob * 100).toFixed(1);
    };

    // 体制显示
    const regimeProb = decision.factors?.regime_prob || 0;
    const winProb = decision.factors?.win_prob || decision.entry_prob || 0;
    const lossProb = decision.factors?.loss_prob || decision.exit_prob || 0;

    // 信号状态
    const getSignalDisplay = () => {
        const cls = decision.factors?.predicted_class;
        if (cls === 2) return { label: 'WIN', color: 'text-success' };
        if (cls === 0) return { label: 'LOSS', color: 'text-danger' };
        return { label: 'NEUTRAL', color: 'text-warning' };
    };

    const signalDisplay = getSignalDisplay();

    return (
        <div className="footer-height glass-panel flex items-center justify-between px-6 z-10 relative mt-auto border-t">
            {/* 左侧：持仓信息 */}
            <div className="flex items-center gap-6 text-sm">
                {/* 持仓状态 */}
                <div className="flex items-center gap-2">
                    <span className="text-dim">POSITION</span>
                    {positions.length > 0 ? (
                        <span className={`font-mono font-bold ${mainDirection === 'LONG' ? 'text-success' : 'text-danger'}`}>
                            {mainDirection} × {positions.length}
                        </span>
                    ) : (
                        <span className="font-mono text-dim">FLAT</span>
                    )}
                </div>

                {/* 手数 */}
                {totalVolume > 0 && (
                    <div className="flex items-center gap-2">
                        <span className="text-dim">VOL</span>
                        <span className="font-mono text-main">{totalVolume.toFixed(2)}</span>
                    </div>
                )}

                {/* 未实现 P&L */}
                <div className="flex items-center gap-2">
                    <span className="text-dim">UNREALIZED</span>
                    <span className={`font-mono flex items-center ${pnlColor(totalUnrealizedPnL)}`}>
                        {totalUnrealizedPnL >= 0 ? '+' : ''}{totalUnrealizedPnL.toFixed(2)}
                        <PnLIcon pnl={totalUnrealizedPnL} />
                    </span>
                </div>

                {/* 已实现 P&L */}
                <div className="flex items-center gap-2">
                    <span className="text-dim">REALIZED</span>
                    <span className={`font-mono flex items-center ${pnlColor(totalRealizedPnL)}`}>
                        {totalRealizedPnL >= 0 ? '+' : ''}{totalRealizedPnL.toFixed(2)}
                    </span>
                </div>
            </div>

            {/* 右侧：策略信息 */}
            <div className="flex items-center gap-4 text-xs font-mono text-dim">
                {/* 当前信号 */}
                <div className="flex items-center gap-2">
                    <Target size={12} />
                    <span className={signalDisplay.color}>{signalDisplay.label}</span>
                </div>

                <span>|</span>

                {/* Win/Loss 概率 */}
                <div className="flex items-center gap-2">
                    <span className="text-success">W:{formatProb(winProb)}%</span>
                    <span className="text-danger">L:{formatProb(lossProb)}%</span>
                </div>

                <span>|</span>

                {/* 模式 */}
                <span className="uppercase">
                    MODE: {decision.factors?.mode || 'TICK'}
                </span>
            </div>
        </div>
    );
};

export default Footer;
