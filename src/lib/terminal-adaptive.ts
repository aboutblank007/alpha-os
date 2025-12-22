import type { MT5Account, MT5Position } from "@/store/useTradeStore";
import type { Signal } from "@/store/useSignalStore";
import type { Density, UserRole } from "@/store/useSettingsStore";

export type TerminalMode = "trading" | "monitoring" | "risk_handling";
export type RiskState = "normal" | "elevated" | "critical";

export interface DashboardStatsLite {
  totalPnL?: number;
  winRate?: number;
  profitFactor?: number;
  maxDrawdown?: number; // percent
  executionEfficiency?: number;
}

export interface RiskAssessment {
  state: RiskState;
  score: number; // 0-100
  reasons: string[];
  metrics: {
    marginLevelPct: number | null;
    freeMarginPct: number | null;
    drawdownPct: number | null;
  };
}

export interface SignalAssessment {
  level: "none" | "weak" | "strong";
  confidence: number; // 0-1
  symbol?: string;
  action?: "BUY" | "SELL";
  ageSec?: number;
}

export interface DeviceProfile {
  width: number;
  isMobile: boolean;
  columns: 1 | 2 | 3 | 4 | 6;
}

export interface TerminalContext {
  device: DeviceProfile;
  role: UserRole;
  density: Density;
  mode: TerminalMode;
  risk: RiskAssessment;
  signal: SignalAssessment;
  isConnected: boolean;
  latencyMs: number | null;
}

export interface ModuleDef {
  key: string;
  label: string;
  group: "risk" | "execution" | "ai" | "market" | "analysis" | "utility";
  canFocus: boolean;
  hideInCompact?: boolean;
  hideInRiskCritical?: boolean;
  basePriority: Record<TerminalMode, number>; // 0-100
}

export const DASHBOARD_MODULES: Record<string, ModuleDef> = {
  chart: {
    key: "chart",
    label: "行情图表",
    group: "market",
    canFocus: true,
    basePriority: { trading: 70, monitoring: 65, risk_handling: 45 },
  },
  orders: {
    key: "orders",
    label: "持仓与执行",
    group: "execution",
    canFocus: true,
    basePriority: { trading: 95, monitoring: 55, risk_handling: 90 },
  },
  marketWatch: {
    key: "marketWatch",
    label: "Market Watch",
    group: "market",
    canFocus: false,
    hideInRiskCritical: true,
    basePriority: { trading: 55, monitoring: 85, risk_handling: 10 },
  },
  market: {
    key: "market",
    label: "权益曲线",
    group: "analysis",
    hideInCompact: true,
    hideInRiskCritical: false,
    canFocus: false,
    basePriority: { trading: 35, monitoring: 55, risk_handling: 55 },
  },
  ai_monitor: {
    key: "ai_monitor",
    label: "AI 决策监控",
    group: "ai",
    canFocus: true,
    basePriority: { trading: 80, monitoring: 70, risk_handling: 60 },
  },
  alerts: {
    key: "alerts",
    label: "风险预警",
    group: "risk",
    canFocus: true,
    basePriority: { trading: 65, monitoring: 55, risk_handling: 100 },
  },
  recent: {
    key: "recent",
    label: "Recent Trades",
    group: "analysis",
    hideInCompact: true,
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 35, monitoring: 45, risk_handling: 5 },
  },
  analytics: {
    key: "analytics",
    label: "Analytics",
    group: "analysis",
    hideInCompact: true,
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 10, monitoring: 35, risk_handling: 0 },
  },
  symbols: {
    key: "symbols",
    label: "Symbol Performance",
    group: "analysis",
    hideInCompact: true,
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 15, monitoring: 35, risk_handling: 0 },
  },
  sentiment: {
    key: "sentiment",
    label: "Sentiment",
    group: "analysis",
    hideInCompact: true,
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 5, monitoring: 20, risk_handling: 0 },
  },
  sessions: {
    key: "sessions",
    label: "Market Sessions",
    group: "utility",
    hideInCompact: true,
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 5, monitoring: 25, risk_handling: 0 },
  },
  ai_controls: {
    key: "ai_controls",
    label: "AI 控制",
    group: "ai",
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 15, monitoring: 10, risk_handling: 0 },
  },
  ai_performance: {
    key: "ai_performance",
    label: "AI 性能",
    group: "ai",
    hideInCompact: true,
    hideInRiskCritical: true,
    canFocus: false,
    basePriority: { trading: 10, monitoring: 15, risk_handling: 0 },
  },
};

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

export function computeDeviceProfile(width: number): DeviceProfile {
  const isMobile = width < 768;
  const columns: DeviceProfile["columns"] =
    // Dashboard 更接近“数据分析页”的大卡片布局：先保证单卡片可读宽度，再决定列数
    width >= 2560 ? 6 :
    width >= 1920 ? 4 :
    width >= 1536 ? 3 :
    width >= 1024 ? 3 :
    width >= 768 ? 2 : 1;
  return { width, isMobile, columns };
}

export function computeSignalState(signals: Signal[]): SignalAssessment {
  if (!signals || signals.length === 0) return { level: "none", confidence: 0 };
  const latest = signals[0];
  const ageSec = Math.max(0, Math.floor((Date.now() - new Date(latest.created_at).getTime()) / 1000));
  const aiMatch = latest.comment?.match(/AI:\s*(\d+\.?\d*)/);
  let confidence = aiMatch ? parseFloat(aiMatch[1]) : 0;
  if (confidence > 1) confidence = confidence / 100;
  confidence = clamp(confidence, 0, 1);

  const freshStrong = ageSec <= 300 && confidence >= 0.75;
  const freshWeak = ageSec <= 900 && confidence >= 0.55;
  return {
    level: freshStrong ? "strong" : freshWeak ? "weak" : "none",
    confidence,
    symbol: latest.symbol,
    action: latest.action,
    ageSec,
  };
}

export function computeRiskState(params: {
  account: MT5Account | null;
  positions: MT5Position[];
  stats: DashboardStatsLite;
  riskLevel: "low" | "medium" | "high";
  isConnected: boolean;
}): RiskAssessment {
  const { account, stats, riskLevel, isConnected } = params;

  const marginLevelPct =
    account && account.margin > 0 ? (account.equity / account.margin) * 100 : null;
  const freeMarginPct =
    account && account.equity > 0 ? (account.free_margin / account.equity) * 100 : null;
  const drawdownPct = typeof stats.maxDrawdown === "number" ? stats.maxDrawdown : null;
  const totalPnL = typeof stats.totalPnL === "number" ? stats.totalPnL : null;

  let score = 0;
  const reasons: string[] = [];

  if (marginLevelPct !== null) {
    if (marginLevelPct < 200) { score += 60; reasons.push(`保证金水平过低（${marginLevelPct.toFixed(0)}%）`); }
    else if (marginLevelPct < 300) { score += 40; reasons.push(`保证金水平偏低（${marginLevelPct.toFixed(0)}%）`); }
    else if (marginLevelPct < 500) { score += 20; reasons.push(`保证金水平下降（${marginLevelPct.toFixed(0)}%）`); }
  }

  if (freeMarginPct !== null) {
    if (freeMarginPct < 20) { score += 30; reasons.push(`可用保证金紧张（${freeMarginPct.toFixed(0)}%）`); }
    else if (freeMarginPct < 30) { score += 20; reasons.push(`可用保证金偏低（${freeMarginPct.toFixed(0)}%）`); }
  }

  if (drawdownPct !== null) {
    if (drawdownPct > 20) { score += 40; reasons.push(`最大回撤偏高（${drawdownPct.toFixed(1)}%）`); }
    else if (drawdownPct > 10) { score += 20; reasons.push(`回撤抬升（${drawdownPct.toFixed(1)}%）`); }
  }

  if (totalPnL !== null) {
    if (totalPnL < -1500) { score += 40; reasons.push(`当期亏损显著（$${totalPnL.toFixed(0)}）`); }
    else if (totalPnL < -500) { score += 20; reasons.push(`当期亏损超过阈值（$${totalPnL.toFixed(0)}）`); }
  }

  if (!isConnected) {
    score += 15;
    reasons.push("桥接断连（存在执行/同步风险）");
  }

  const sensitivity = riskLevel === "low" ? 1.15 : riskLevel === "high" ? 0.9 : 1.0;
  score = clamp(Math.round(score * sensitivity), 0, 100);

  const state: RiskState =
    (marginLevelPct !== null && marginLevelPct < 200) || score >= 70 ? "critical" :
    score >= 35 ? "elevated" : "normal";

  return {
    state,
    score,
    reasons: reasons.slice(0, 3),
    metrics: {
      marginLevelPct,
      freeMarginPct,
      drawdownPct,
    },
  };
}

export interface FocusSuggestion {
  shouldEnter: boolean;
  type: "risk" | "trade";
  reason: string;
  primaryModules: string[]; // 1-2 keys
}

export interface PriorityResult {
  effectiveLayout: string[];
  visibility: Record<string, boolean>;
  focusSuggestion: FocusSuggestion | null;
}

export function computePriority(params: {
  baseLayout: string[];
  pinned: Set<string>;
  ctx: TerminalContext;
}): PriorityResult {
  const { baseLayout, pinned, ctx } = params;

  const baseIndex = new Map<string, number>();
  baseLayout.forEach((k, i) => baseIndex.set(k, i));

  const visibility: Record<string, boolean> = {};

  const scoreOf = (key: string) => {
    const def = DASHBOARD_MODULES[key];
    if (!def) return 0;

    let s = def.basePriority[ctx.mode] ?? 0;

    // 风险态强制抬权
    if (ctx.risk.state === "critical") {
      if (def.group === "risk") s += 40;
      if (key === "orders") s += 25;
      if (key === "chart") s += 10;
      if (def.hideInRiskCritical) s -= 60;
    } else if (ctx.risk.state === "elevated") {
      if (def.group === "risk") s += 15;
      if (key === "orders") s += 10;
    }

    // 强信号抬权（偏 Trading）
    if (ctx.signal.level === "strong") {
      if (key === "ai_monitor") s += 25;
      if (key === "orders") s += 15;
      if (key === "chart") s += 10;
    } else if (ctx.signal.level === "weak") {
      if (key === "ai_monitor") s += 10;
    }

    // 角色/密度降权
    if (ctx.role === "novice") {
      if (def.group === "analysis") s -= 20;
      if (key === "analytics") s -= 30;
    }
    if (ctx.density === "compact" && def.hideInCompact) s -= 25;

    return clamp(s, 0, 120);
  };

  // 初始可见性（按密度和设备）
  const minScore =
    ctx.device.isMobile ? 999 : // mobile 走专用布局
    ctx.density === "compact" ? 18 :
    ctx.density === "balanced" ? 10 : 0;

  for (const key of baseLayout) {
    const def = DASHBOARD_MODULES[key];
    const s = scoreOf(key);
    const hiddenByRisk = ctx.risk.state === "critical" && !!def?.hideInRiskCritical;
    visibility[key] = !hiddenByRisk && s >= minScore;
  }

  // 排序：Pinned 优先（保持原序），其余按 baseIndex (用户自定义顺序)
  // 移除基于 score 的自动重排，防止拖拽后回弹
  const pinnedKeys = baseLayout.filter(k => pinned.has(k));
  const rest = baseLayout
    .filter(k => !pinned.has(k))
    .sort((a, b) => {
      // 完全尊重 baseIndex 的顺序
      return (baseIndex.get(a) ?? 0) - (baseIndex.get(b) ?? 0);
    });

  const effectiveLayout = [...pinnedKeys, ...rest].filter(k => visibility[k] !== false);

  // Focus 建议
  let focusSuggestion: FocusSuggestion | null = null;
  if (ctx.device.isMobile) {
    focusSuggestion = null;
  } else if (ctx.risk.state === "critical") {
    focusSuggestion = {
      shouldEnter: true,
      type: "risk",
      reason: ctx.risk.reasons[0] ? `高风险：${ctx.risk.reasons[0]}` : "高风险状态",
      primaryModules: ["alerts", "orders"].filter(k => baseLayout.includes(k)),
    };
  } else if (ctx.mode === "trading" && ctx.signal.level === "strong") {
    focusSuggestion = {
      shouldEnter: true,
      type: "trade",
      reason: `强信号（${Math.round(ctx.signal.confidence * 100)}%）`,
      primaryModules: ["orders", "chart"].filter(k => baseLayout.includes(k)),
    };
  }

  return { effectiveLayout, visibility, focusSuggestion };
}


