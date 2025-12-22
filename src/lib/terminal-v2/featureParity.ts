export type TerminalWidgetKey =
  | "chart"
  | "orders"
  | "recent"
  | "alerts"
  | "ai_monitor"
  | "marketWatch"
  | "sessions"
  | "market"
  | "symbols"
  | "analytics"
  | "ai_controls"
  | "ai_performance"
  | "insights"
  | "sentiment";

export type TerminalZoneKey = "main" | "side" | "bottom";

/**
 * Terminal V2 的“现有功能不丢”契约（Feature Parity Contract）
 * - 任何重做都必须保证这些 widget 与交互能力仍可用
 * - 允许 UI 重绘/布局重构，但功能入口/数据链路不可缺失
 */
export const TERMINAL_V2_FEATURES = {
  widgets: [
    {
      key: "chart",
      label: "行情图表",
      mustHave: true,
      dataDeps: ["prices (chart lib)", "selectedSymbol (local)"],
      defaultZone: "main",
    },
    {
      key: "orders",
      label: "MT5 持仓与执行",
      mustHave: true,
      dataDeps: ["bridge status", "mt5 positions/account (store)"],
      defaultZone: "main",
      notes: ["orders_auto: 少仓→主区，重仓/列多→通栏"],
    },
    {
      key: "recent",
      label: "近期交易",
      mustHave: true,
      dataDeps: ["supabase trades (realtime)"],
      defaultZone: "bottom",
      notes: ["默认通栏 100%，可折叠"],
    },
    {
      key: "alerts",
      label: "风险预警",
      mustHave: true,
      dataDeps: ["risk assessment (derived)"],
      defaultZone: "side",
      notes: ["风险 critical 时置顶并增强视觉权重"],
    },
    {
      key: "ai_monitor",
      label: "AI 决策监控",
      mustHave: true,
      dataDeps: ["ai stats/settings", "signals"],
      defaultZone: "side",
    },
    {
      key: "marketWatch",
      label: "MT5 Live / Market Watch",
      mustHave: true,
      dataDeps: ["market prices", "symbols list"],
      defaultZone: "side",
      notes: ["固定窄列，不允许占半屏"],
    },
    {
      key: "sessions",
      label: "市场时段",
      mustHave: false,
      dataDeps: ["clock/timezone (local)"],
      defaultZone: "side",
    },
    {
      key: "market",
      label: "权益曲线",
      mustHave: false,
      dataDeps: ["trades (closed)", "account balance"],
      defaultZone: "main",
    },
    {
      key: "symbols",
      label: "Symbol Performance",
      mustHave: false,
      dataDeps: ["trades"],
      defaultZone: "side",
    },
    {
      key: "analytics",
      label: "Analytics Panel",
      mustHave: false,
      dataDeps: ["trades"],
      defaultZone: "main",
    },
    {
      key: "ai_controls",
      label: "AI 控制",
      mustHave: false,
      dataDeps: ["ai settings"],
      defaultZone: "side",
    },
    {
      key: "ai_performance",
      label: "AI 性能",
      mustHave: false,
      dataDeps: ["ai stats"],
      defaultZone: "side",
    },
    {
      key: "insights",
      label: "Trading Insights",
      mustHave: false,
      dataDeps: ["trades"],
      defaultZone: "main",
    },
    {
      key: "sentiment",
      label: "Sentiment",
      mustHave: false,
      dataDeps: ["trades"],
      defaultZone: "main",
    },
  ] as const satisfies Array<{
    key: TerminalWidgetKey;
    label: string;
    mustHave: boolean;
    dataDeps: string[];
    defaultZone: TerminalZoneKey;
    notes?: string[];
  }>,
  interactions: [
    "默认主状态 Trading（本地设置）",
    "Focus Mode（快捷键/风险强制/建议进入）",
    "区内拖拽排序 + 布局记忆（本地）",
    "跨区移动通过菜单动作（移动到主区/侧栏/底部）",
  ] as const,
} as const;

export const TERMINAL_V2_DEFAULT_WIDGETS_TRADING: TerminalWidgetKey[] = [
  "chart",
  "orders",
  "alerts",
  "ai_monitor",
  "marketWatch",
  "recent",
];


