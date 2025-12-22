/**
 * 量子 HFT 系统类型定义
 * 与 Q-Link 2.0 协议对齐
 */

// ================== 市场数据类型 ==================

/**
 * 高频 Tick 数据
 * CSV 格式: TICK,timestamp,symbol,bid,ask,volume,wick_ratio,vol_density,vol_shock
 */
export interface MarketTick {
  type: 'TICK';
  timestamp: number;
  symbol: string;
  bid: number;
  ask: number;
  volume: number;
  /** 影线比率 (量子噪声代理) */
  wickRatio: number;
  /** 流动性密度 */
  volumeDensity: number;
  /** 波动率冲击 */
  volumeShock: number;
  /** 可选扩展字段 */
  spread?: number;
  tickRate?: number;
  bidAskImbalance?: number;
}

/**
 * K 线数据
 */
export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  /** 影线比率 (用于染色) */
  wickRatio?: number;
}

// ================== 量子 AI 遥测 ==================

/**
 * 量子模型遥测数据
 */
export interface QuantumTelemetry {
  /** 梯度范数 (监控 < 1e-4 → 贫瘠高原) */
  gradientNorm: number;
  /** 纠缠熵 */
  entropy: number;
  /** 预测置信度 [0.0 - 1.0] */
  confidence: number;
  /** M2 Pro P-Core 负载 (%) */
  pCoreLoad: number;
  /** M2 Pro E-Core 负载 (%) */
  eCoreLoad: number;
  /** Tick-to-Trade 延迟 (ms) */
  latency: number;
  /** 时间戳 */
  timestamp: number;
}

/**
 * Alpha 信号
 */
export interface AlphaSignal {
  timestamp: number;
  symbol: string;
  prediction: number;
  direction: 'BUY' | 'SELL';
  confidence: number;
  tickData: MarketTick;
  /** 量子态熵 */
  qStateEntropy?: number;
}

/**
 * 风控决策
 */
export interface RiskDecision {
  timestamp: number;
  symbol: string;
  action: 'BET' | 'PASS';
  /** Meta-Labeling 概率 */
  metaProb: number;
  /** 最终仓位 (手数) */
  positionSize: number;
  /** 凯利比例 */
  kellyFraction: number;
  /** 波动率缩放系数 */
  volScalar: number;
  /** 隐含滑点成本 */
  lvarCost: number;
}

// ================== 系统状态 ==================

/**
 * 系统健康状态
 */
export type SystemStatus = 'OK' | 'WARNING' | 'CRITICAL';

/**
 * 心跳消息
 */
export interface Heartbeat {
  source: 'ALPHA' | 'RISK' | 'MT5';
  timestamp: number;
  status: string;
}

/**
 * 账户状态
 */
export interface AccountState {
  balance: number;
  equity: number;
  marginUsed: number;
  marginFree: number;
  positions: Position[];
  timestamp: number;
}

/**
 * 持仓信息
 */
export interface Position {
  ticket: number;
  symbol: string;
  side: 'BUY' | 'SELL';
  lots: number;
  openPrice: number;
  currentPrice: number;
  profit: number;
  sl?: number;
  tp?: number;
  magic?: number;
  comment?: string;
}

// ================== Worker 消息类型 ==================

/**
 * Worker 输入消息
 */
export interface WorkerInputMessage {
  type: 'PARSE_TICK' | 'PARSE_BATCH';
  data: string | string[];
}

/**
 * Worker 输出消息
 */
export interface WorkerOutputMessage {
  type: 'TICK_PARSED' | 'BATCH_PARSED' | 'ERROR';
  data?: MarketTick | MarketTick[];
  error?: string;
}

// ================== WebSocket 消息 ==================

/**
 * WebSocket 消息封装
 */
export interface WSMessage<T = unknown> {
  type: string;
  payload: T;
  timestamp: number;
}

/**
 * 命令总线消息
 */
export interface CommandMessage {
  action: 'CLOSE_ALL' | 'UPDATE_PARAM' | 'PAUSE' | 'RESUME';
  key?: string;
  value?: number | string | boolean;
}

// ================== 常量 ==================

export const PORTS = {
  MARKET_STREAM: 6557,
  COMMAND_BUS: 6558,
  STATE_SYNC: 6559,
  ALPHA_TO_RISK: 6560,
} as const;

export const THRESHOLDS = {
  /** 延迟阈值 (ms) */
  LATENCY_WARNING: 50,
  LATENCY_CRITICAL: 100,
  /** 心跳超时 (ms) */
  HEARTBEAT_TIMEOUT: 5000,
  /** 梯度消失阈值 */
  BARREN_PLATEAU: 1e-5,
  /** Meta-Labeling 阈值 */
  META_THRESHOLD: 0.6,
  /** 高熵影线比率 */
  HIGH_ENTROPY_WICK: 0.6,
} as const;
