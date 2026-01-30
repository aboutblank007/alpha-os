// WebSocket Message Types

export type MessageType =
    | 'welcome'
    | 'tick'
    | 'status'
    | 'position'
    | 'decision'
    | 'dollar_bar'
    | 'inference'
    | 'order'
    | 'runtime_snapshot'
    | 'system'
    | 'pong';

export interface WSMessage<T = any> {
    type: MessageType;
    data?: T;
    [key: string]: any; // Compatibility for top-level fields
}

export interface TickData {
    symbol: string;
    bid: number;
    ask: number;
    spread: number;
    timestamp?: number;
}

export interface SystemStatus {
    connected: boolean;
    zmq_latency_ms: number;
    bars_completed: number;
    uptime_seconds: number;
    ticks_received: number;
    tick_intensity: number;
    temperature: number;
    entropy: number;
    warmup_complete: boolean;
    model_halted: boolean;
}

export interface Position {
    id?: string;
    direction: 'LONG' | 'SHORT' | null;
    volume: number;
    entry_price: number;
    stop_loss?: number;
    take_profit?: number;
    unrealized_pnl: number;
    realized_pnl?: number;
    magic?: number;
    ticket?: number;
}

export interface Decision {
    signal: string;
    score: number;
    entry_prob: number;
    exit_prob: number;
    confidence: number;
    meta_confidence: number;
    market_phase: string;
    ts_phase: string;
    temperature: number;
    entropy: number;
    timestamp?: number;
    factors: {
        regime_prob: number;
        win_prob: number;
        loss_prob: number;
        predicted_class: number;
        mode: string;
        trend_direction?: number;
        [key: string]: any;
    };
}

export interface DollarBar {
    progress: number;
    complete: boolean;
    bar_id: number;
}

export interface InferenceState {
    stage: string;
    [key: string]: any;
}

export interface RuntimeSnapshot {
    timestamp: number;
    warmup_progress: number;
    ticks_total: number;
    open_positions: number;
    guardian_halt: boolean;
    exit_v21_enabled: boolean;
    db_snapshot_count: number;
    // Added fields
    market_phase: string;
    temperature: number;
    entropy: number;
    symbol: string;
}

export interface SystemMessage {
    message: string;
    level: 'info' | 'warning' | 'error';
    timestamp?: number;
}
