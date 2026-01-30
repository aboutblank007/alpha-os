import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
    Monitor, Server, Database, Activity, Zap, Shield, Cpu, Layers,
    Radio, Brain, Target, BarChart3, Gauge, TrendingUp, Wifi
} from 'lucide-react';

// 动画状态映射
const ANIMATION_CLASSES = {
    breathe: 'node-breathe',
    flash: 'node-flash',
    inferring: 'node-inferring',
    burst: 'node-burst',
    success: 'node-success-pulse',
};

// 节点包装器 - 增强动画效果
type NodeWrapperProps = {
    children: React.ReactNode;
    className?: string;
    glow?: string;
    selected?: boolean;
    active?: boolean;
    animationState?: keyof typeof ANIMATION_CLASSES | null;
    onAnimationEnd?: () => void;
};

const NodeWrapper = ({
    children,
    className = '',
    glow = 'primary',
    selected,
    active = true,
    animationState = null,  // 'breathe' | 'flash' | 'inferring' | 'burst' | 'success'
    onAnimationEnd = null,
}: NodeWrapperProps) => {
    const [currentAnimation, setCurrentAnimation] = useState(animationState);

    // 处理一次性动画
    useEffect(() => {
        if (animationState && ['flash', 'burst', 'success'].includes(animationState)) {
            setCurrentAnimation(animationState);
            const timer = setTimeout(() => {
                setCurrentAnimation(null);
                onAnimationEnd?.();
            }, 800);
            return () => clearTimeout(timer);
        } else {
            setCurrentAnimation(animationState);
        }
    }, [animationState, onAnimationEnd]);

    const animationClass = currentAnimation ? ANIMATION_CLASSES[currentAnimation] : '';

    return (
        <div className={`
            p-3 rounded-lg node-wrapper backdrop-blur-md 
            border transition-all duration-300
            ${active ? 'border-glass' : 'border-gray-800 opacity-50'}
            ${selected ? `shadow-[0_0_20px_var(--${glow})] border-${glow}` : 'shadow-lg'}
            ${animationClass}
            ${className}
        `}>
            {children}
        </div>
    );
};

// 通用连接点
const HandleSet = ({ hideLeft = false, hideRight = false, hideTop = false, hideBottom = false }) => (
    <>
        {!hideTop && <Handle type="target" position={Position.Top} className="!bg-dim !w-2 !h-2 !border-0" />}
        {!hideBottom && <Handle type="source" position={Position.Bottom} className="!bg-primary !w-2 !h-2 !border-0" />}
        {!hideLeft && <Handle type="target" position={Position.Left} className="!bg-dim !w-2 !h-2 !border-0" />}
        {!hideRight && <Handle type="source" position={Position.Right} className="!bg-primary !w-2 !h-2 !border-0" />}
    </>
);

// 归一化节点 (Dynamic Z-Score)
export const NormalizerNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper selected={selected} glow="primary">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Gauge size={16} className="text-primary" />
                <div className="text-xs font-bold text-main">Z-Score Normalizer</div>
            </div>
            <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                    <span className="text-dim">Method</span>
                    <span className="text-primary">EWMA</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-dim">Range</span>
                    <span className="text-primary">[-5, +5]</span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// 热力学节点 (Temperature & Entropy)
export const ThermodynamicsNode = memo(({ data, selected }: NodeProps<any>) => {
    const temp = (data.temperature || 0) * 100;
    const entropy = (data.entropy || 0) * 100;

    return (
        <NodeWrapper selected={selected} glow="warning" className="!min-w-[150px]">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Zap size={16} className="text-warning" />
                <div className="text-xs font-bold text-main">Thermodynamics</div>
            </div>
            <div className="space-y-3">
                <div className="space-y-1">
                    <div className="flex justify-between text-[9px] font-mono">
                        <span className="text-dim uppercase">Temperature (T)</span>
                        <span className="text-warning">{temp.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-blue-500 via-yellow-500 to-red-500 transition-all duration-500"
                            style={{ width: `${temp}%` }}
                        />
                    </div>
                </div>
                <div className="space-y-1">
                    <div className="flex justify-between text-[9px] font-mono">
                        <span className="text-dim uppercase">Entropy (S)</span>
                        <span className="text-warning">{entropy.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-green-500 via-gray-500 to-purple-500 transition-all duration-500"
                            style={{ width: `${entropy}%` }}
                        />
                    </div>
                </div>
            </div>
        </NodeWrapper>
    );
});

// CfC Encoder 节点 (LNN)
export const CfCEncoderNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper
            selected={selected}
            glow="secondary"
            animationState={data.active ? 'inferring' : null}
            className="!min-w-[140px] node-liquid"
        >
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Brain size={18} className="text-secondary" />
                <div className="text-xs font-bold text-main">CfC Encoder (LNN)</div>
            </div>
            <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                    <span className="text-dim">Hidden Dim</span>
                    <span className="text-secondary">64</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-dim">ODE Dynamics</span>
                    <span className="text-success text-[8px] uppercase">Active</span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// XGBoost Head 节点
export const XGBoostHeadNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper selected={selected} glow="warning">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <TrendingUp size={16} className="text-warning" />
                <div className="text-xs font-bold text-main">XGBoost Decision</div>
            </div>
            <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                    <span className="text-dim">Estimators</span>
                    <span className="text-warning">250</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-dim">Depth</span>
                    <span className="text-warning">7</span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// Model Guardian 节点
export const ModelGuardianNode = memo(({ data, selected }: NodeProps<any>) => {
    const isHalted = data.halted === true;
    return (
        <NodeWrapper
            selected={selected}
            glow={isHalted ? 'danger' : 'success'}
            animationState={isHalted ? 'flash' : 'breathe'}
        >
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Shield size={16} className={isHalted ? 'text-danger' : 'text-success'} />
                <div className="text-xs font-bold text-main">Model Guardian</div>
            </div>
            <div className={`text-[10px] font-bold text-center p-1 rounded ${isHalted ? 'bg-danger/20 text-danger' : 'bg-success/20 text-success'}`}>
                {isHalted ? 'SYSTEM HALTED' : 'NOMINAL'}
            </div>
        </NodeWrapper>
    );
});

// T-S Phase Classifier 节点
export const TSPhaseClassifierNode = memo(({ data, selected }: NodeProps<any>) => {
    const phase = data.phase || 'UNKNOWN';
    const PHASE_CONFIGS = {
        'LAMINAR': { color: 'text-blue-400', label: 'Laminar', desc: 'Trend Stable' },
        'TURBULENT': { color: 'text-purple-400', label: 'Turbulent', desc: 'Mean-Rev' },
        'PHASE_TRANSITION': { color: 'text-orange-400', label: 'Transition', desc: 'Breakout!' },
        'FROZEN': { color: 'text-gray-400', label: 'Frozen', desc: 'No Liquidity' },
        'UNKNOWN': { color: 'text-dim', label: 'Unknown', desc: 'Waiting...' },
    } as const;
    const phaseConfig = (PHASE_CONFIGS as any)[phase] || PHASE_CONFIGS.UNKNOWN;

    return (
        <NodeWrapper selected={selected} glow="primary">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Target size={16} className="text-primary" />
                <div className="text-xs font-bold text-main">Execution Phase</div>
            </div>
            <div className="text-center p-1">
                <div className={`text-sm font-black font-mono tracking-tighter ${phaseConfig.color} uppercase`}>
                    {phaseConfig.label}
                </div>
                <div className="text-[8px] text-dim italic">{phaseConfig.desc}</div>
            </div>
        </NodeWrapper>
    );
});

// MT5 终端节点
export const MT5Node = memo(({ data, selected }: NodeProps<any>) => {
    const isOnline = data.online !== false;
    const [priceFlash, setPriceFlash] = useState(false);
    const [lastPrice, setLastPrice] = useState(data.price);

    // 价格变化时闪烁
    useEffect(() => {
        if (data.price && data.price !== lastPrice) {
            setPriceFlash(true);
            setLastPrice(data.price);
            const timer = setTimeout(() => setPriceFlash(false), 300);
            return () => clearTimeout(timer);
        }
    }, [data.price, lastPrice]);

    return (
        <NodeWrapper
            selected={selected}
            glow="secondary"
            active={isOnline}
            animationState={isOnline ? 'breathe' : null}
        >
            <HandleSet hideTop />
            <div className="flex items-center gap-2 mb-2">
                <Monitor size={18} className="text-secondary" />
                <div className="text-sm font-bold text-main">{data.label}</div>
                {isOnline && (
                    <div className="w-2 h-2 rounded-full bg-success connection-indicator active ml-auto" />
                )}
            </div>
            <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                    <span className="text-dim">Symbol</span>
                    <span className="text-primary">{data.symbol || 'XAUUSD'}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-dim">Price</span>
                    <span className={`${data.priceUp ? 'text-success' : 'text-danger'} ${priceFlash ? 'number-pop' : ''}`}>
                        {data.price || '----'}
                    </span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// ZMQ 通信节点
export const ZMQNode = memo(({ data, selected }: NodeProps<any>) => {
    const isConnected = data.connected !== false;
    return (
        <NodeWrapper selected={selected} glow="primary" active={isConnected}>
            <HandleSet />
            <div className="flex items-center gap-2 mb-2 border-b border-glass pb-2">
                <Radio size={14} className={isConnected ? 'text-primary' : 'text-dim'} />
                <div className="text-xs font-bold uppercase tracking-wider text-main">{data.label}</div>
            </div>
            <div className="text-[10px] font-mono text-dim space-y-1">
                {data.details && data.details.map((d, i) => (
                    <div key={i} className="flex justify-between">
                        <span>{d.label}</span>
                        <span className={d.active ? 'text-success' : 'text-primary'}>{d.value}</span>
                    </div>
                ))}
            </div>
        </NodeWrapper>
    );
});

// Tick 处理节点
export const TickNode = memo(({ data, selected }: NodeProps<any>) => {
    const progress = data.progress || 0;
    const [windowFlash, setWindowFlash] = useState(false);
    const [lastWindow, setLastWindow] = useState(data.value);

    // 窗口完成时闪烁
    useEffect(() => {
        if (data.value && data.value !== lastWindow) {
            setWindowFlash(true);
            setLastWindow(data.value);
            const timer = setTimeout(() => setWindowFlash(false), 500);
            return () => clearTimeout(timer);
        }
    }, [data.value, lastWindow]);

    return (
        <NodeWrapper
            selected={selected}
            glow="primary"
            className="!min-w-[160px]"
            animationState={windowFlash ? 'flash' : null}
        >
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <BarChart3 size={14} className={`text-primary ${progress > 80 ? 'indicator-blink' : ''}`} />
                <span className="text-xs font-bold">{data.label}</span>
            </div>
            {/* 进度条 */}
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden mb-2 relative">
                <div
                    className="h-full bg-gradient-to-r from-primary to-success transition-all duration-300"
                    style={{ width: `${progress}%` }}
                />
                {/* 进度条发光效果 */}
                {progress > 0 && (
                    <div
                        className="absolute top-0 h-full w-4 bg-white opacity-30 blur-sm"
                        style={{ left: `${Math.max(0, progress - 5)}%` }}
                    />
                )}
            </div>
            <div className="flex justify-between text-[10px] font-mono">
                <span className="text-dim">Window</span>
                <span className={`text-success ${windowFlash ? 'number-pop' : ''}`}>{data.value || 0}</span>
            </div>
            <div className="flex justify-between text-[10px] font-mono">
                <span className="text-dim">Ticks</span>
                <span className="text-primary">{data.ticks || 0}</span>
            </div>
        </NodeWrapper>
    );
});

// 特征引擎节点
export const FeatureNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper selected={selected} glow="warning">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Layers size={16} className="text-warning" />
                <div className="text-xs font-bold text-main">{data.label}</div>
            </div>
            <div className="grid grid-cols-2 gap-1 text-[9px] font-mono">
                <div className="bg-dark rounded px-2 py-1 text-center">
                    <div className="text-dim">Dims</div>
                    <div className="text-warning">{data.dims || 26}</div>
                </div>
                <div className="bg-dark rounded px-2 py-1 text-center">
                    <div className="text-dim">Seq</div>
                    <div className="text-warning">{data.seq || 20}</div>
                </div>
            </div>
        </NodeWrapper>
    );
});

// 模型推理节点
export const ModelNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper
            selected={selected}
            glow="secondary"
            animationState={data.inferring ? 'inferring' : null}
        >
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Brain size={18} className={`text-secondary ${data.inferring ? 'indicator-blink' : ''}`} />
                <div className="text-sm font-bold text-main">{data.label}</div>
            </div>
            <div className="space-y-1">
                <div className="flex items-center gap-2 text-[10px]">
                    <div className={`w-2 h-2 rounded-full transition-colors duration-300 ${data.inferring ? 'bg-success connection-indicator active' : 'bg-dim'}`}></div>
                    <span className="text-dim">LNN</span>
                    <span className="text-success ml-auto">64-dim</span>
                </div>
                <div className="flex items-center gap-2 text-[10px]">
                    <div className={`w-2 h-2 rounded-full transition-colors duration-300 ${data.inferring ? 'bg-warning connection-indicator active' : 'bg-dim'}`}></div>
                    <span className="text-dim">XGBoost</span>
                    <span className="text-warning ml-auto">3-class</span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// 概率输出节点
export const ProbNode = memo(({ data, selected }: NodeProps<any>) => {
    const winProb = parseFloat(data.win) || 0;
    const lossProb = parseFloat(data.loss) || 0;
    const neutralProb = Math.max(0, 100 - winProb - lossProb);

    return (
        <NodeWrapper selected={selected} glow="success" className="!min-w-[140px]">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Gauge size={14} className="text-success" />
                <span className="text-xs font-bold">{data.label}</span>
            </div>
            {/* 堆叠概率条 */}
            <div className="h-3 bg-gray-800 rounded-full overflow-hidden flex mb-2">
                <div className="bg-success transition-all" style={{ width: `${winProb}%` }} />
                <div className="bg-warning transition-all" style={{ width: `${neutralProb}%` }} />
                <div className="bg-danger transition-all" style={{ width: `${lossProb}%` }} />
            </div>
            <div className="grid grid-cols-3 gap-1 text-[9px] font-mono text-center">
                <div>
                    <div className="text-success">{winProb.toFixed(0)}%</div>
                    <div className="text-dim">Win</div>
                </div>
                <div>
                    <div className="text-warning">{neutralProb.toFixed(0)}%</div>
                    <div className="text-dim">Neu</div>
                </div>
                <div>
                    <div className="text-danger">{lossProb.toFixed(0)}%</div>
                    <div className="text-dim">Loss</div>
                </div>
            </div>
        </NodeWrapper>
    );
});

// 信号过滤节点
export const SignalNode = memo(({ data, selected }: NodeProps<any>) => {
    const signal = data.signal || 'IDLE';
    const signalConfig = {
        'WIN': { color: 'text-success', animation: 'signal-win', glow: 'success' },
        'LOSS': { color: 'text-danger', animation: 'signal-loss', glow: 'danger' },
        'NEUTRAL': { color: 'text-warning', animation: '', glow: 'warning' },
        'IDLE': { color: 'text-dim', animation: '', glow: 'primary' },
    }[signal] || { color: 'text-dim', animation: '', glow: 'primary' };

    return (
        <NodeWrapper
            selected={selected}
            glow={signalConfig.glow}
            animationState={signal === 'WIN' || signal === 'LOSS' ? 'flash' : null}
        >
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Target size={16} className={signal !== 'IDLE' ? 'indicator-blink text-primary' : 'text-dim'} />
                <div className="text-xs font-bold text-main">{data.label}</div>
            </div>
            <div className="bg-dark rounded p-2 text-center">
                <div className={`text-lg font-bold font-mono ${signalConfig.color} ${signalConfig.animation}`}>
                    {signal}
                </div>
                <div className="text-[9px] text-dim mt-1">
                    Threshold: {data.threshold || '50%'}
                </div>
            </div>
        </NodeWrapper>
    );
});

// 风控节点
export const RiskNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper selected={selected} glow="danger">
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Shield size={16} className="text-danger" />
                <div className="text-xs font-bold text-main">{data.label}</div>
            </div>
            <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                    <span className="text-dim">Max Pos</span>
                    <span className="text-primary">{data.maxPos || 3}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-dim">SL/TP</span>
                    <span className="text-warning">{data.sltp || 'Dynamic'}</span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// 订单执行节点
export const ExecutionNode = memo(({ data, selected }: NodeProps<any>) => {
    const [lastOrders, setLastOrders] = useState(data.orders);
    const [orderChanged, setOrderChanged] = useState(false);

    // 订单变化时爆发效果
    useEffect(() => {
        if (data.orders !== lastOrders) {
            setOrderChanged(true);
            setLastOrders(data.orders);
            const timer = setTimeout(() => setOrderChanged(false), 600);
            return () => clearTimeout(timer);
        }
    }, [data.orders, lastOrders]);

    return (
        <NodeWrapper
            selected={selected}
            glow="danger"
            animationState={orderChanged ? 'burst' : (data.orders > 0 ? 'breathe' : null)}
        >
            <HandleSet />
            <div className="flex items-center gap-2 mb-2">
                <Zap size={16} className={`text-danger ${data.orders > 0 ? 'indicator-blink' : ''}`} />
                <div className="text-xs font-bold text-main">{data.label}</div>
            </div>
            <div className="bg-dark rounded p-2">
                <div className="flex justify-between text-[10px]">
                    <span className="text-dim">Positions</span>
                    <span className={`text-primary font-mono ${orderChanged ? 'number-pop' : ''}`}>{data.orders || 0}</span>
                </div>
                <div className="flex justify-between text-[10px] mt-1">
                    <span className="text-dim">Status</span>
                    <span className={data.orders > 0 ? 'text-warning' : 'text-success'}>
                        {data.orders > 0 ? 'ACTIVE' : 'READY'}
                    </span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// 数据库节点
export const DatabaseNode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper selected={selected} glow="warning">
            <HandleSet />
            <div className="flex flex-col items-center gap-2 text-center p-1">
                <Database size={20} className="text-warning" />
                <div className="text-xs font-bold">{data.label}</div>
                <div className="text-[9px] font-mono bg-dark px-2 py-1 rounded w-full">
                    {data.tables || 4} tables
                </div>
            </div>
        </NodeWrapper>
    );
});

// WebSocket 节点
export const WSNode = memo(({ data, selected }: NodeProps<any>) => {
    const isConnected = data.connected !== false;
    return (
        <NodeWrapper selected={selected} glow="success" active={isConnected}>
            <HandleSet hideRight hideBottom />
            <div className="flex items-center gap-2 mb-2">
                <Wifi size={14} className={isConnected ? 'text-success' : 'text-dim'} />
                <div className="text-xs font-bold text-main">{data.label}</div>
            </div>
            <div className="text-[10px] font-mono">
                <div className="flex justify-between">
                    <span className="text-dim">Port</span>
                    <span className="text-success">{data.port || ':8765'}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-dim">Status</span>
                    <span className={isConnected ? 'text-success' : 'text-dim'}>{isConnected ? 'OK' : 'N/A'}</span>
                </div>
            </div>
        </NodeWrapper>
    );
});

// UI 节点
export const UINode = memo(({ data, selected }: NodeProps<any>) => {
    return (
        <NodeWrapper selected={selected} glow="success">
            <HandleSet hideRight hideBottom />
            <div className="flex items-center gap-2 mb-2">
                <Monitor size={16} className="text-success" />
                <div className="text-xs font-bold text-main">{data.label}</div>
            </div>
            <div className="bg-dark rounded p-2 text-center">
                <TrendingUp size={20} className="text-success mx-auto mb-1" />
                <div className="text-[9px] text-dim">Live Dashboard</div>
            </div>
        </NodeWrapper>
    );
});

// Alpha191 主模型节点
export const Alpha191Node = memo(({ data, selected }: NodeProps<any>) => (
    <NodeWrapper selected={selected} glow="secondary">
        <HandleSet />
        <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={14} className="text-secondary" />
            <span className="text-xs font-bold">{data.label}</span>
        </div>
        <div className="text-[10px] font-mono">
            <div className="flex justify-between">
                <span className="text-dim">Direction</span>
                <span className={`font-bold ${data.signal === 'LONG' ? 'text-success' : data.signal === 'SHORT' ? 'text-danger' : 'text-warning'}`}>
                    {data.signal || 'NEUTRAL'}
                </span>
            </div>
        </div>
    </NodeWrapper>
));

// 保留旧节点类型的兼容
export const ServiceNode = ZMQNode;
export const ProcessNode = memo(({ data, selected }: NodeProps<any>) => (
    <NodeWrapper selected={selected} glow="primary">
        <HandleSet />
        <div className="flex items-center gap-2 mb-2">
            <Cpu size={18} className="text-primary" />
            <div className="text-sm font-bold text-main">{data.label}</div>
        </div>
        <div className="flex justify-between items-center bg-dark rounded p-2 border border-glass">
            <span className="text-[10px] text-dim">STATUS</span>
            <span className="text-[10px] text-success animate-pulse">RUNNING</span>
        </div>
    </NodeWrapper>
));
export const MetricNode = ProbNode;
