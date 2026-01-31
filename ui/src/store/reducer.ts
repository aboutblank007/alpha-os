import { AlphaOSState, Action, ActionTypes } from './types';

export const initialState: AlphaOSState = {
    wsConnected: false,
    tick: {
        symbol: 'XAUUSD',
        bid: 0,
        ask: 0,
        spread: 0,
        timestamp: undefined,
    },
    status: {
        connected: false,
        zmq_latency_ms: 0,
        bars_completed: 0,
        uptime_seconds: 0,
        ticks_received: 0,
        tick_intensity: 0,
        temperature: 0,
        entropy: 0,
        warmup_complete: false,
        model_halted: false,
    },
    positions: [],
    decision: {
        signal: 'IDLE',
        score: 0,
        entry_prob: 0,
        exit_prob: 0,
        confidence: 0,
        meta_confidence: 0,
        market_phase: 'UNKNOWN',
        ts_phase: 'UNKNOWN',
        temperature: 0,
        entropy: 0,
        factors: {
            regime_prob: 0,
            win_prob: 0,
            loss_prob: 0,
            predicted_class: 1,
            mode: 'tick',
        },
    },
    dollarBar: {
        progress: 0,
        complete: false,
        bar_id: 0,
    },
    inference: {
        stage: 'idle',
    },
    lastOrder: null,
    systemMessages: [],
    signalHistory: [],
    runtimeSnapshot: {
        timestamp: 0,
        symbol: 'XAUUSD',
        warmup_progress: 0,
        ticks_total: 0,
        open_positions: 0,
        guardian_halt: false,
        exit_v21_enabled: true,
        db_snapshot_count: 0,
        market_phase: 'UNKNOWN',
        temperature: 0,
        entropy: 0,
    },
    lastEvent: {
        type: null,
        timestamp: null,
        data: null,
    },
    _prev: {
        bars_completed: 0,
        predicted_class: 1,
        positions_count: 0,
    },
};

export function alphaOSReducer(state: AlphaOSState, action: Action): AlphaOSState {
    switch (action.type) {
        case ActionTypes.SET_WS_CONNECTED:
            return { ...state, wsConnected: action.payload };
            
        case ActionTypes.UPDATE_TICK:
            return {
                ...state,
                tick: { ...state.tick, ...action.payload },
                status: {
                    ...state.status,
                    ticks_received: state.status.ticks_received + 1,
                },
            };
            
        case ActionTypes.UPDATE_STATUS:
            return {
                ...state,
                status: { ...state.status, ...action.payload },
            };
            
        case ActionTypes.UPDATE_POSITION:
            // 后端发送的是单个持仓的汇总信息，直接替换
            // direction 为 null 表示无持仓
            {
                const payload = action.payload;
                let nextPositions = [];
                if (Array.isArray(payload)) {
                    nextPositions = payload;
                } else if (payload && typeof payload === 'object' && 'positions' in payload) {
                    const positions = (payload as { positions?: unknown }).positions;
                    if (Array.isArray(positions)) {
                        nextPositions = positions;
                    }
                } else if (payload && typeof payload === 'object' && 'direction' in payload) {
                    const direction = (payload as { direction?: unknown }).direction;
                    if (direction) {
                        nextPositions = [payload];
                    }
                }
                return {
                    ...state,
                    positions: nextPositions.map((pos, index) => ({
                        ...pos,
                        id: pos.id ?? pos.ticket ?? `pos-${index}`,
                    })),
                };
            }
            
        case ActionTypes.UPDATE_DECISION:
            const newDecision = { ...state.decision, ...action.payload };
            // 添加到历史记录
            const newHistory = [
                { ...newDecision, timestamp: Date.now() } as any, // Temporary loose typing for timestamp
                ...state.signalHistory.slice(0, 19),
            ];
            return {
                ...state,
                decision: newDecision,
                signalHistory: newHistory,
            };
            
        case ActionTypes.UPDATE_DOLLAR_BAR:
            return {
                ...state,
                dollarBar: { ...state.dollarBar, ...action.payload },
            };
            
        case ActionTypes.UPDATE_INFERENCE:
            return {
                ...state,
                inference: { ...state.inference, ...action.payload },
            };
            
        case ActionTypes.UPDATE_ORDER:
            return { ...state, lastOrder: action.payload };

        case ActionTypes.UPDATE_RUNTIME_SNAPSHOT:
            return {
                ...state,
                runtimeSnapshot: { ...state.runtimeSnapshot, ...action.payload },
                decision: {
                    ...state.decision,
                    market_phase: action.payload.market_phase ?? state.decision.market_phase,
                    temperature: action.payload.temperature ?? state.decision.temperature,
                    entropy: action.payload.entropy ?? state.decision.entropy,
                },
                // 同时也更新部分 status 以保持兼容
                status: {
                    ...state.status,
                    warmup_complete: (action.payload.warmup_progress || 0) >= 1.0,
                    model_halted: !!action.payload.guardian_halt,
                    ticks_received: action.payload.ticks_total || state.status.ticks_received,
                    temperature: action.payload.temperature ?? state.status.temperature,
                    entropy: action.payload.entropy ?? state.status.entropy,
                }
            };
            
        case ActionTypes.ADD_SYSTEM_MESSAGE:
            return {
                ...state,
                systemMessages: [
                    { ...action.payload, timestamp: Date.now() },
                    ...state.systemMessages.slice(0, 49),
                ],
            };
        
        case ActionTypes.TRIGGER_EVENT:
            return {
                ...state,
                lastEvent: {
                    type: action.payload.type,
                    timestamp: Date.now(),
                    data: action.payload.data || null,
                },
            };
            
        default:
            return state;
    }
}
