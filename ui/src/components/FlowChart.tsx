import React, { useEffect, useMemo, useRef, useCallback } from 'react';
import ReactFlow, {
    Background,
    useNodesState,
    useEdgesState,
    Node,
    Edge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { 
    MT5Node, ZMQNode, TickNode, FeatureNode, ModelNode, 
    ProbNode, SignalNode, RiskNode, ExecutionNode, 
    DatabaseNode, WSNode, UINode, NormalizerNode,
    ThermodynamicsNode, CfCEncoderNode, XGBoostHeadNode,
    ModelGuardianNode, TSPhaseClassifierNode, Alpha191Node
} from './CustomNodes';
import { FlowingEdge, PulseEdge, EdgeFilters } from './AnimatedEdge';
import { useAlphaOS } from '../context/AlphaOSContext';

// 注册所有节点类型
const nodeTypes = {
    mt5: MT5Node,
    zmq: ZMQNode,
    tick: TickNode,
    feature: FeatureNode,
    model: ModelNode,
    prob: ProbNode,
    signal: SignalNode,
    risk: RiskNode,
    execution: ExecutionNode,
    database: DatabaseNode,
    ws: WSNode,
    ui: UINode,
    normalizer: NormalizerNode,
    thermo: ThermodynamicsNode,
    cfc: CfCEncoderNode,
    xgb: XGBoostHeadNode,
    guardian: ModelGuardianNode,
    phase: TSPhaseClassifierNode,
    alpha191: Alpha191Node,
};

// 注册自定义边类型
const edgeTypes = {
    flowing: FlowingEdge,
    pulse: PulseEdge,
};

// 布局常量 (v4 架构化布局)
const COL = {
    DATA: 50,      // 数据源
    ZMQ: 300,      // 通信层
    PROC: 550,     // 预处理层
    BRAIN: 850,    // 核心脑
    DECIDE: 1150,  // 决策层
    EXEC: 1450,    // 执行层
};

const ROW = {
    R1: 50,
    R2: 200,
    R3: 350,
    R4: 500,
    R5: 650,
};

// 初始节点配置
type FlowNodeData = Record<string, any>;
type FlowEdgeData = Record<string, any>;
type FlowNode = Node<FlowNodeData>;
type FlowEdge = Edge<FlowEdgeData>;

const createInitialNodes = (): FlowNode[] => [
    // ========== DATA LAYER (Column 1-2) ==========
    {
        id: 'mt5',
        type: 'mt5',
        position: { x: COL.DATA, y: ROW.R2 },
        data: { label: 'MT5 EA (external)', symbol: 'XAUUSD', price: '----', online: false },
    },
    {
        id: 'zmq-pub',
        type: 'zmq',
        position: { x: COL.ZMQ, y: ROW.R1 },
        data: { label: 'ZMQ PUB (mt5 tick)', connected: false, details: [{ label: 'Port', value: ':5555' }, { label: 'Mode', value: 'PUB' }] },
    },
    {
        id: 'zmq-sub',
        type: 'zmq',
        position: { x: COL.ZMQ, y: ROW.R2 },
        data: { label: 'alphaos.execution.zmq_client.ZMQClient', connected: false, details: [{ label: 'Latency', value: '0ms' }] },
    },
    {
        id: 'zmq-router',
        type: 'zmq',
        position: { x: COL.ZMQ, y: ROW.R3 },
        data: { label: 'MT5 EA Router (external)', connected: false, details: [{ label: 'Port', value: ':5556' }, { label: 'Mode', value: 'ROUTER' }] },
    },

    // ========== PRE-PROCESS LAYER (Column 3) ==========
    {
        id: 'tick-processor',
        type: 'tick',
        position: { x: COL.PROC, y: ROW.R1 },
        data: { label: 'alphaos.v4.sampling.UnifiedSampler', value: 0, ticks: 0, progress: 0 },
    },
    {
        id: 'normalizer',
        type: 'normalizer',
        position: { x: COL.PROC, y: ROW.R2 },
        data: { label: 'alphaos.v4.features.FeaturePipelineV4 (zscore)' },
    },
    {
        id: 'thermo',
        type: 'thermo',
        position: { x: COL.PROC, y: ROW.R3 },
        data: { label: 'alphaos.v4.features.ThermodynamicsConfig', temperature: 0, entropy: 0 },
    },

    // ========== BRAIN LAYER (Column 4) ==========
    {
        id: 'cfc-encoder',
        type: 'cfc',
        position: { x: COL.BRAIN, y: ROW.R2 },
        data: { label: 'alphaos.v4.models.CfCEncoder', active: false },
    },
    {
        id: 'alpha191',
        type: 'alpha191',
        position: { x: COL.BRAIN, y: ROW.R1 },
        data: { label: 'alphaos.v4.primary.PrimaryEngineV4', signal: 'NEUTRAL' },
    },
    {
        id: 'xgb-head',
        type: 'xgb',
        position: { x: COL.BRAIN, y: ROW.R3 },
        data: { label: 'alphaos.v4.inference.XGBoost (xgb.XGBClassifier)', win: 0, loss: 0 },
    },

    // ========== DECISION LAYER (Column 5) ==========
    {
        id: 'signal-filter',
        type: 'signal',
        position: { x: COL.DECIDE, y: ROW.R2 },
        data: { label: 'alphaos.v4.inference.ConfidenceGate', signal: 'IDLE', threshold: 'rolling' },
        },
    {
        id: 'model-guardian',
        type: 'guardian',
        position: { x: COL.DECIDE, y: ROW.R1 },
        data: { label: 'alphaos.monitoring.ModelGuardian', halted: false },
    },
    {
        id: 'phase-classifier',
        type: 'phase',
        position: { x: COL.DECIDE, y: ROW.R3 },
        data: { label: 'alphaos.v4.features.ts_phase', phase: 'UNKNOWN' },
    },

    // ========== EXECUTION LAYER (Column 6) ==========
    {
        id: 'risk-manager',
        type: 'risk',
        position: { x: COL.EXEC, y: ROW.R1 },
        data: { label: 'alphaos.monitoring.RiskManager', maxPos: 0.01, sltp: 'Dynamic' },
    },
    {
        id: 'execution',
        type: 'execution',
        position: { x: COL.EXEC, y: ROW.R2 },
        data: { label: 'alphaos.v4.cli Execution', orders: 0, status: 'READY' },
    },
    {
        id: 'zmq-dealer',
        type: 'zmq',
        position: { x: COL.EXEC, y: ROW.R3 },
        data: { label: 'ZMQ DEALER (order)', connected: false, details: [{ label: 'Mode', value: 'DEALER' }] },
    },

    // ========== INFRASTRUCTURE ==========
    {
        id: 'timescaledb',
        type: 'database',
        position: { x: COL.PROC, y: ROW.R4 },
        data: { label: 'TimescaleDB runtime_state', tables: 4 },
    },
    {
        id: 'websocket',
        type: 'ws',
        position: { x: COL.DECIDE, y: ROW.R4 },
        data: { label: 'alphaos.monitoring.WSRuntimeServer', port: ':8765', clients: 0, connected: false },
    },
];

// 边的配置 (v3.3 拓扑连接)
const createInitialEdges = (): FlowEdge[] => [
    // --- Connectivity Layer ---
    { id: 'e-mt5-pub', source: 'mt5', target: 'zmq-pub', type: 'flowing', data: { particleColor: 'tick', flowSpeed: 'fast', isActive: true }, style: { stroke: '#00f3ff', opacity: 0.6 } },
    { id: 'e-pub-sub', source: 'zmq-pub', target: 'zmq-sub', type: 'flowing', data: { particleColor: 'tick', flowSpeed: 'fast', isActive: true }, style: { stroke: '#00f3ff', opacity: 0.6 } },
    
    // --- Processing Layer ---
    { id: 'e-sub-tick', source: 'zmq-sub', target: 'tick-processor', type: 'flowing', data: { particleColor: 'tick', flowSpeed: 'normal', isActive: true }, style: { stroke: '#00f3ff', opacity: 0.5 } },
    { id: 'e-tick-norm', source: 'tick-processor', target: 'normalizer', type: 'flowing', data: { particleColor: 'feature', flowSpeed: 'normal', isActive: true }, style: { stroke: '#ffbe00', opacity: 0.5 } },
    { id: 'e-norm-thermo', source: 'normalizer', target: 'thermo', type: 'flowing', data: { particleColor: 'feature', flowSpeed: 'normal', isActive: true }, style: { stroke: '#ffbe00', opacity: 0.5 } },

    // --- Intelligence Layer ---
    { id: 'e-thermo-cfc', source: 'thermo', target: 'cfc-encoder', type: 'flowing', data: { particleColor: 'model', flowSpeed: 'normal', isActive: true }, style: { stroke: '#7000ff', opacity: 0.5 } },
    { id: 'e-cfc-alpha', source: 'cfc-encoder', target: 'alpha191', type: 'flowing', data: { particleColor: 'model', flowSpeed: 'normal', isActive: true }, style: { stroke: '#7000ff', opacity: 0.5 } },
    { id: 'e-cfc-xgb', source: 'cfc-encoder', target: 'xgb-head', type: 'flowing', data: { particleColor: 'model', flowSpeed: 'normal', isActive: true }, style: { stroke: '#7000ff', opacity: 0.5 } },

    // --- Decision Layer ---
    { id: 'e-alpha-signal', source: 'alpha191', target: 'signal-filter', type: 'flowing', data: { particleColor: 'websocket', flowSpeed: 'normal', isActive: true }, style: { stroke: '#00ff9d', opacity: 0.5 } },
    { id: 'e-xgb-signal', source: 'xgb-head', target: 'signal-filter', type: 'flowing', data: { particleColor: 'websocket', flowSpeed: 'normal', isActive: true }, style: { stroke: '#00ff9d', opacity: 0.5 } },
    { id: 'e-xgb-guardian', source: 'xgb-head', target: 'model-guardian', type: 'flowing', data: { particleColor: 'websocket', flowSpeed: 'normal', isActive: true }, style: { stroke: '#00ff9d', opacity: 0.5 } },
    { id: 'e-signal-phase', source: 'signal-filter', target: 'phase-classifier', type: 'flowing', data: { particleColor: 'websocket', flowSpeed: 'normal', isActive: true }, style: { stroke: '#00ff9d', opacity: 0.5 } },
    
    // --- Execution Layer ---
    { id: 'e-phase-risk', source: 'phase-classifier', target: 'risk-manager', type: 'flowing', data: { particleColor: 'order', flowSpeed: 'normal', isActive: false }, style: { stroke: '#ff0055', opacity: 0.5 } },
    { id: 'e-risk-exec', source: 'risk-manager', target: 'execution', type: 'flowing', data: { particleColor: 'order', flowSpeed: 'normal', isActive: false }, style: { stroke: '#ff0055', opacity: 0.5 } },
    { id: 'e-exec-dealer', source: 'execution', target: 'zmq-dealer', type: 'flowing', data: { particleColor: 'order', flowSpeed: 'normal', isActive: false }, style: { stroke: '#ff0055', opacity: 0.5 } },
    { id: 'e-dealer-router', source: 'zmq-dealer', target: 'zmq-router', type: 'flowing', data: { particleColor: 'order', flowSpeed: 'normal', isActive: false }, style: { stroke: '#ff0055', opacity: 0.5 } },
    { id: 'e-router-mt5', source: 'zmq-router', target: 'mt5', type: 'flowing', data: { particleColor: 'order', flowSpeed: 'normal', isActive: false }, style: { stroke: '#ff0055', opacity: 0.5 } },

    // --- Persistence & Feedback ---
    { id: 'e-tick-db', source: 'tick-processor', target: 'timescaledb', type: 'smoothstep', style: { stroke: '#ffbe00', strokeWidth: 1, strokeDasharray: '6,4', opacity: 0.3 } },
    { id: 'e-guardian-ws', source: 'model-guardian', target: 'websocket', type: 'flowing', data: { particleColor: 'websocket', flowSpeed: 'normal', isActive: true }, style: { stroke: '#00ff9d', opacity: 0.4 } },
];

const FlowChart = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState<FlowNodeData>(createInitialNodes());
    const [edges, setEdges, onEdgesChange] = useEdgesState<FlowEdgeData>(createInitialEdges());
    
    const { status, decision, wsConnected, tick, positions, lastEvent } = useAlphaOS();
    
    // 跟踪上一次事件用于触发边的动画
    const lastEventRef = useRef(null);

    // 根据实时数据更新节点
    useEffect(() => {
        setNodes((nds) => nds.map((node) => {
            switch (node.id) {
                case 'mt5':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            price: tick.bid > 0 ? tick.bid.toFixed(2) : '----',
                            priceUp: decision.factors?.trend_direction > 0,
                            online: status.connected,
                        }
                    };
                    
                case 'zmq-pub':
                case 'zmq-router':
                    return {
                        ...node,
                        data: { ...node.data, connected: status.connected }
                    };
                    
                case 'zmq-sub':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            connected: status.connected,
                            details: [
                                { label: 'Status', value: status.connected ? 'OK' : 'N/A', active: status.connected },
                                { label: 'Latency', value: `${status.zmq_latency_ms?.toFixed(0) || 0}ms` },
                            ]
                        }
                    };
                    
                case 'zmq-dealer':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            connected: status.connected,
                            details: [
                                { label: 'Mode', value: 'DEALER' },
                                { label: 'Positions', value: positions.length.toString() },
                            ]
                        }
                    };
                    
                case 'tick-processor':
                    const ticksInWindow = status.ticks_received % 500;
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            value: status.bars_completed,
                            ticks: status.ticks_received || 0,
                            progress: (ticksInWindow / 500) * 100,
                        }
                    };
                    
                case 'thermo':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            temperature: status.temperature || decision.temperature || 0,
                            entropy: status.entropy || decision.entropy || 0,
                        }
                    };
                    
                case 'cfc-encoder':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            active: status.connected && !status.model_halted,
                        }
                    };
                    
                case 'alpha191':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            signal: decision.signal, // 使用主模型的方向信号
                        }
                    };
                    
                case 'xgb-head':
                    const winProb = (decision.factors?.win_prob || 0) * 100;
                    const lossProb = (decision.factors?.loss_prob || 0) * 100;
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            win: winProb,
                            loss: lossProb,
                        }
                    };
                    
                case 'signal-filter':
                    const cls = decision.factors?.predicted_class;
                    const signal = cls === 2 ? 'WIN' : cls === 0 ? 'LOSS' : 'NEUTRAL';
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            signal: signal,
                        }
                    };
                    
                case 'model-guardian':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            halted: status.model_halted,
                        }
                    };
                    
                case 'phase-classifier':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            phase: decision.market_phase || decision.ts_phase || 'UNKNOWN',
                        }
                    };
                    
                case 'execution':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            orders: positions.length,
                            status: positions.length > 0 ? 'ACTIVE' : 'READY',
                        }
                    };
                    
                case 'websocket':
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            connected: wsConnected,
                            clients: wsConnected ? 1 : 0,
                        }
                    };
                    
                default:
                    return node;
            }
        }));
    }, [status, decision, wsConnected, tick, positions, setNodes]);

    // 更新边的动画状态（基于连接状态和持仓）
    useEffect(() => {
        setEdges((eds) => eds.map((edge) => {
            // 识别边的类型
            const isDataFlow = ['e-mt5-pub', 'e-pub-sub', 'e-sub-tick'].includes(edge.id);
            const isFeatureFlow = ['e-tick-norm', 'e-norm-thermo'].includes(edge.id);
            const isModelFlow = ['e-thermo-cfc', 'e-cfc-alpha', 'e-cfc-xgb'].includes(edge.id);
            const isSignalFlow = ['e-alpha-signal', 'e-xgb-signal', 'e-xgb-guardian'].includes(edge.id);
            const isOrderFlow = ['e-signal-phase', 'e-phase-risk', 'e-risk-exec', 'e-exec-dealer', 'e-dealer-router', 'e-router-mt5'].includes(edge.id);
            const isWSFlow = ['e-xgb-ws'].includes(edge.id);
            
            // 判断是否激活
            let isActive = false;
            if (isDataFlow) isActive = status.connected;
            else if (isFeatureFlow) isActive = status.connected;
            else if (isModelFlow) isActive = status.connected && !status.model_halted;
            else if (isSignalFlow) isActive = decision.factors?.predicted_class !== undefined;
            else if (isOrderFlow) isActive = positions.length > 0;
            else if (isWSFlow) isActive = wsConnected;
            
            // 更新 data.isActive
            if (edge.type === 'flowing' && edge.data) {
                return {
                    ...edge,
                    data: { ...edge.data, isActive },
                };
            }
            return edge;
        }));
    }, [status.connected, status.model_halted, wsConnected, positions.length, decision.factors?.predicted_class, setEdges]);
    
    // 事件触发时的边高亮动画
    useEffect(() => {
        if (lastEvent && lastEvent.timestamp !== lastEventRef.current) {
            lastEventRef.current = lastEvent.timestamp;
            
            // 根据事件类型高亮相关边
            const highlightEdges = [];
            switch (lastEvent.type) {
                case 'TICK_WINDOW':
                    highlightEdges.push('e-tick-norm', 'e-norm-thermo');
                    break;
                case 'INFERENCE':
                    highlightEdges.push('e-thermo-cfc', 'e-cfc-alpha', 'e-cfc-xgb', 'e-alpha-signal', 'e-xgb-signal');
                    break;
                case 'ORDER_SENT':
                    highlightEdges.push('e-phase-risk', 'e-risk-exec', 'e-exec-dealer');
                    break;
                case 'ORDER_FILLED':
                case 'ORDER_CLOSED':
                    highlightEdges.push('e-dealer-router', 'e-router-mt5');
                    break;
                case 'SIGNAL_CHANGE':
                    highlightEdges.push('e-xgb-signal', 'e-signal-phase');
                    break;
                default:
                    break;
            }
            
            if (highlightEdges.length > 0) {
                // 临时加速粒子表示事件
                setEdges((eds) => eds.map((edge) => {
                    if (highlightEdges.includes(edge.id) && edge.type === 'flowing' && edge.data) {
                        return {
                            ...edge,
                            data: { ...edge.data, flowSpeed: 'fast', particleCount: 'high' },
                        };
                    }
                    return edge;
                }));
                
                // 1秒后恢复正常速度
                setTimeout(() => {
                    setEdges((eds) => eds.map((edge) => {
                        if (highlightEdges.includes(edge.id) && edge.type === 'flowing' && edge.data) {
                            return {
                                ...edge,
                                data: { ...edge.data, flowSpeed: 'normal', particleCount: 'medium' },
                            };
                        }
                        return edge;
                    }));
                }, 1000);
            }
        }
    }, [lastEvent, setEdges]);

    return (
        <div className="w-full h-full bg-dark relative">
            {/* SVG 滤镜定义 */}
            <EdgeFilters />
            
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                fitView
                fitViewOptions={{ padding: 0.15 }}
                className="alpha-flow"
                minZoom={0.3}
                maxZoom={1.5}
                defaultViewport={{ x: 0, y: 0, zoom: 0.65 }}
                proOptions={{ hideAttribution: true }}
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                panOnDrag={true}
                zoomOnScroll={true}
            >
                <Background color="#1a1a3a" gap={40} size={1} />
            </ReactFlow>
        </div>
    );
};

export default FlowChart;
