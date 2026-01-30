import React, { useState, useEffect } from 'react';
import { Activity, Wifi, WifiOff, Clock, Zap, BarChart2 } from 'lucide-react';
import { useAlphaOS } from '../context/AlphaOSContext';
import { HeaderNav } from './HeaderNav';

const Header: React.FC = () => {
    const [time, setTime] = useState(new Date());
    const { tick, status, wsConnected, connectionState, decision, runtime } = useAlphaOS();

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    // 格式化运行时间
    const formatUptime = (seconds: number) => {
        if (!seconds) return '0s';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    };

    // 价格变化颜色
    const priceColor = tick.bid > 0 ? 
        (decision.factors?.trend_direction > 0 ? 'text-success' : 
         decision.factors?.trend_direction < 0 ? 'text-danger' : 'text-main') 
        : 'text-dim';

    const symbol = tick.bid > 0 ? tick.symbol : (runtime?.symbol || tick.symbol);

    // 连接状态
    const getConnectionStatus = () => {
        if (connectionState === 'connected' && status.connected) {
            return { label: 'SYSTEM ONLINE', className: 'text-success', dot: 'status-active' };
        }
        if (connectionState === 'connecting') {
            return { label: 'CONNECTING...', className: 'text-warning', dot: 'status-warning' };
        }
        return { label: 'DISCONNECTED', className: 'text-danger', dot: 'status-error' };
    };

    const connStatus = getConnectionStatus();

    return (
        <div className="header-height glass-panel flex items-center justify-between px-6 z-10 relative">
            {/* 左侧：Logo + 状态 */}
            <div className="flex items-center gap-4">
                <h1 className="text-xl font-bold tracking-widest neon-title flex items-center gap-2">
                    <Activity size={20} />
                    ALPHA<span className="text-white">OS</span>
                    <span className="text-[10px] bg-secondary/30 text-secondary px-1.5 py-0.5 rounded border border-secondary/50 font-black ml-1">v4 META</span>
                    <span className="text-[8px] text-dim ml-2 opacity-50">Build: {new Date().toLocaleTimeString()}</span>
                </h1>
                <div className="h-6 w-[1px] bg-panel"></div>
                <div className={`flex items-center gap-2 text-sm ${connStatus.className}`}>
                    <span className={`status-dot ${connStatus.dot}`}></span>
                    {connStatus.label}
                </div>
            </div>

            {/* 中间：价格 + 指标 */}
            <div className="flex items-center gap-8">
                {/* 品种价格 */}
                <div className="flex items-center gap-3">
                    <span className="text-dim text-xs uppercase tracking-wider">{symbol}</span>
                    <span className={`text-lg font-mono font-bold ${priceColor}`}>
                        {tick.bid > 0 ? tick.bid.toFixed(2) : '----'}
                    </span>
                    {tick.spread > 0 && (
                        <span className="text-xs text-dim">
                            ({tick.spread.toFixed(1)})
                        </span>
                    )}
                </div>

                {/* 分隔线 */}
                <div className="h-6 w-[1px] bg-panel"></div>

                {/* 系统指标 */}
                <div className="flex items-center gap-4 text-dim text-sm">
                    {/* ZMQ 延迟 */}
                    <div className="flex items-center gap-1" title="ZMQ Latency">
                        <Zap size={14} className={status.zmq_latency_ms < 100 ? 'text-success' : status.zmq_latency_ms < 500 ? 'text-warning' : 'text-danger'} />
                        <span className="font-mono">{status.zmq_latency_ms.toFixed(0)}ms</span>
                    </div>

                    {/* Tick 窗口 */}
                    <div className="flex items-center gap-1" title="Tick Windows">
                        <BarChart2 size={14} />
                        <span className="font-mono">{status.bars_completed}</span>
                    </div>

                    {/* 运行时间 */}
                    <div className="flex items-center gap-1" title="Uptime">
                        <Clock size={14} />
                        <span className="font-mono">{formatUptime(status.uptime_seconds)}</span>
                    </div>

                    {/* WebSocket 状态 */}
                    <div className="flex items-center gap-1" title="WebSocket">
                        {wsConnected ? (
                            <Wifi size={14} className="text-success" />
                        ) : (
                            <WifiOff size={14} className="text-danger" />
                        )}
                    </div>
                </div>
            </div>

            {/* 右侧：时间 + 视图切换 */}
            <div className="flex items-center gap-6">
                {/* 视图切换 (Refactored to use HeaderNav) */}
                <HeaderNav />

                <div className="text-sm font-mono text-dim">
                    {time.toLocaleTimeString()}
                </div>
            </div>
        </div>
    );
};

export default Header;
