import { TickData, SystemStatus, Position, Decision, DollarBar, InferenceState, RuntimeSnapshot, SystemMessage } from '../shared/ws/types';

export interface AlphaOSState {
    wsConnected: boolean;
    connectionState?: string; // from useWebSocket hook
    tick: TickData;
    status: SystemStatus;
    positions: Position[];
    decision: Decision;
    dollarBar: DollarBar;
    inference: InferenceState;
    lastOrder: any | null;
    systemMessages: SystemMessage[];
    signalHistory: Decision[];
    runtimeSnapshot: RuntimeSnapshot;
    lastEvent: {
        type: string | null;
        timestamp: number | null;
        data: any | null;
    };
    _prev: {
        bars_completed: number;
        predicted_class: number;
        positions_count: number;
    };
}

export const ActionTypes = {
    SET_WS_CONNECTED: 'SET_WS_CONNECTED',
    UPDATE_TICK: 'UPDATE_TICK',
    UPDATE_STATUS: 'UPDATE_STATUS',
    UPDATE_POSITION: 'UPDATE_POSITION',
    UPDATE_DECISION: 'UPDATE_DECISION',
    UPDATE_DOLLAR_BAR: 'UPDATE_DOLLAR_BAR',
    TRIGGER_EVENT: 'TRIGGER_EVENT',
    UPDATE_INFERENCE: 'UPDATE_INFERENCE',
    UPDATE_ORDER: 'UPDATE_ORDER',
    UPDATE_RUNTIME_SNAPSHOT: 'UPDATE_RUNTIME_SNAPSHOT',
    ADD_SYSTEM_MESSAGE: 'ADD_SYSTEM_MESSAGE',
} as const;

export type Action =
    | { type: typeof ActionTypes.SET_WS_CONNECTED; payload: boolean }
    | { type: typeof ActionTypes.UPDATE_TICK; payload: Partial<TickData> }
    | { type: typeof ActionTypes.UPDATE_STATUS; payload: Partial<SystemStatus> }
    | { type: typeof ActionTypes.UPDATE_POSITION; payload: Position }
    | { type: typeof ActionTypes.UPDATE_DECISION; payload: Partial<Decision> }
    | { type: typeof ActionTypes.UPDATE_DOLLAR_BAR; payload: Partial<DollarBar> }
    | { type: typeof ActionTypes.TRIGGER_EVENT; payload: { type: string; data?: any } }
    | { type: typeof ActionTypes.UPDATE_INFERENCE; payload: Partial<InferenceState> }
    | { type: typeof ActionTypes.UPDATE_ORDER; payload: any }
    | { type: typeof ActionTypes.UPDATE_RUNTIME_SNAPSHOT; payload: Partial<RuntimeSnapshot> }
    | { type: typeof ActionTypes.ADD_SYSTEM_MESSAGE; payload: SystemMessage };
