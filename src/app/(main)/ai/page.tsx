"use client";

import React, { useState, useEffect, useCallback } from "react";
import { GlassCard, CardHeader, CardTitle, CardContent } from "@/components/ui/GlassCard";
import { Button } from "@/components/ui/Button";
import { 
    Brain, Settings, Activity, Shield, Zap, RefreshCw, 
    TrendingUp, TrendingDown, Minus, AlertTriangle, 
    ToggleLeft, ToggleRight, ChevronDown, ChevronUp,
    Clock, Target, Layers, Database, Bot, Cpu
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface AISettings {
    id: number;
    risk_off: boolean;
    min_confidence: number;
    max_vol_mult: number;
    mode: string;
    updated_at: string;
}

interface AutomationRule {
    id: string;
    symbol: string;
    is_enabled: boolean;
    fixed_lot_size: number;
    max_spread_points: number;
    ai_mode: string;
    ai_confidence_threshold: number;
    use_kelly_sizing: boolean;
    kelly_fraction: number;
    min_lot_size: number;  // 最小手数
    max_lot_size: number;  // 最大手数
    max_daily_loss: number;
    max_vol_mult: number;
    risk_off: boolean;
    updated_at: string;
}

interface AILog {
    id: string;
    symbol: string;
    action: string;
    price: number;
    timestamp: string;
    resultProfit: number | null;
    aiScore: number;
    regime: string;
    metaProb: number;
    dqnAction: number;
    quantumPolicy: number[];
}

// 状态标签组件
function StatusBadge({ status, size = "md" }: { status: 'active' | 'inactive' | 'warning'; size?: 'sm' | 'md' }) {
    const styles = {
        active: "bg-success/10 text-success border-success/20",
        inactive: "bg-text-muted/10 text-text-muted border-text-muted/20",
        warning: "bg-danger/10 text-danger border-danger/20",
    };
    const labels = { active: "运行中", inactive: "已停止", warning: "暂停中" };
    const sizeStyles = size === 'sm' ? "text-[10px] px-1.5 py-0.5" : "text-xs px-2 py-1";
    
    return (
        <span className={cn("rounded-full border font-medium", styles[status], sizeStyles)}>
            {labels[status]}
        </span>
    );
}

// 开关组件
function Toggle({ 
    enabled, 
    onChange, 
    disabled = false 
}: { 
    enabled: boolean; 
    onChange: (val: boolean) => void;
    disabled?: boolean;
}) {
    return (
        <button 
            onClick={() => !disabled && onChange(!enabled)}
            disabled={disabled}
            className={cn(
                "relative w-10 h-5 rounded-full transition-colors",
                enabled ? "bg-success" : "bg-bg-subtle",
                disabled && "opacity-50 cursor-not-allowed"
            )}
        >
            <div className={cn(
                "absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform",
                enabled ? "translate-x-5" : "translate-x-0.5"
            )} />
        </button>
    );
}

// AI 配置面板
function AIConfigPanel({ 
    settings, 
    onUpdate 
}: { 
    settings: AISettings | null;
    onUpdate: (data: Partial<AISettings>) => void;
}) {
    if (!settings) return <div className="text-text-muted text-sm">加载中...</div>;
    
    const modes = ['conservative', 'balanced', 'aggressive'];
    
    return (
        <div className="space-y-4">
            {/* 全局风险开关 */}
            <div className="flex items-center justify-between p-3 bg-bg-subtle/50 rounded-lg">
                <div className="flex items-center gap-3">
                    <Shield size={18} className={settings.risk_off ? "text-danger" : "text-success"} />
                    <div>
                        <div className="font-medium text-sm">全局风险开关</div>
                        <div className="text-[10px] text-text-muted">关闭后 AI 将停止所有交易</div>
                    </div>
                </div>
                <Toggle 
                    enabled={!settings.risk_off} 
                    onChange={(val) => onUpdate({ risk_off: !val })} 
                />
            </div>
            
            {/* 运行模式 */}
            <div className="space-y-2">
                <label className="text-xs font-medium text-text-muted">运行模式</label>
                <div className="grid grid-cols-3 gap-2">
                    {modes.map(mode => (
                        <button
                            key={mode}
                            onClick={() => onUpdate({ mode })}
                            className={cn(
                                "py-2 px-3 rounded-lg text-xs font-medium transition-all",
                                settings.mode === mode 
                                    ? "bg-primary text-white" 
                                    : "bg-bg-subtle text-text-secondary hover:bg-bg-subtle/80"
                            )}
                        >
                            {mode === 'conservative' ? '保守' : mode === 'balanced' ? '均衡' : '激进'}
                        </button>
                    ))}
                </div>
            </div>
            
            {/* 最小置信度 */}
            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <label className="text-xs font-medium text-text-muted">最小置信度</label>
                    <span className="text-xs font-mono text-primary">{(settings.min_confidence * 100).toFixed(0)}%</span>
                </div>
                <input 
                    type="range" 
                    min="0" max="1" step="0.05"
                    value={settings.min_confidence}
                    onChange={(e) => onUpdate({ min_confidence: parseFloat(e.target.value) })}
                    className="w-full h-1 bg-bg-subtle rounded-lg appearance-none cursor-pointer accent-primary"
                />
            </div>
            
            {/* 最大波动倍数 */}
            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <label className="text-xs font-medium text-text-muted">最大波动倍数</label>
                    <span className="text-xs font-mono text-primary">{settings.max_vol_mult}x</span>
                </div>
                <input 
                    type="range" 
                    min="1" max="3" step="0.1"
                    value={settings.max_vol_mult}
                    onChange={(e) => onUpdate({ max_vol_mult: parseFloat(e.target.value) })}
                    className="w-full h-1 bg-bg-subtle rounded-lg appearance-none cursor-pointer accent-primary"
                />
            </div>
            
            <div className="text-[10px] text-text-muted pt-2 border-t border-white/5">
                最后更新: {new Date(settings.updated_at).toLocaleString('zh-CN')}
            </div>
        </div>
    );
}

// 自动化规则卡片
function RuleCard({ 
    rule, 
    onUpdate 
}: { 
    rule: AutomationRule;
    onUpdate: (id: string, data: Partial<AutomationRule>) => void;
}) {
    const [expanded, setExpanded] = useState(false);
    
    // 智能开关逻辑：开启时同时关闭 risk_off，关闭时只关闭 is_enabled
    const handleMainToggle = (val: boolean) => {
        if (val) {
            // 开启：同时设置 is_enabled=true 和 risk_off=false
            onUpdate(rule.id, { is_enabled: true, risk_off: false });
        } else {
            // 关闭：只关闭 is_enabled
            onUpdate(rule.id, { is_enabled: false });
        }
    };
    
    // 实际运行状态
    const isActuallyEnabled = rule.is_enabled && !rule.risk_off;
    
    return (
        <GlassCard className={cn(
            "p-3 transition-all",
            rule.risk_off && "border-danger/30 bg-danger/5"
        )}>
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <span className="text-sm font-bold text-primary">{rule.symbol.substring(0, 2)}</span>
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <span className="font-bold text-sm">{rule.symbol}</span>
                            <StatusBadge 
                                status={rule.risk_off ? 'warning' : rule.is_enabled ? 'active' : 'inactive'} 
                                size="sm" 
                            />
                            {rule.risk_off && (
                                <span className="text-[9px] text-danger bg-danger/10 px-1.5 py-0.5 rounded">
                                    风控暂停
                                </span>
                            )}
                        </div>
                        <div className="text-[10px] text-text-muted">
                            {rule.ai_mode} · {rule.fixed_lot_size} lots · {(rule.ai_confidence_threshold * 100).toFixed(0)}% 阈值
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <Toggle 
                        enabled={isActuallyEnabled} 
                        onChange={handleMainToggle} 
                    />
                    <button 
                        onClick={() => setExpanded(!expanded)}
                        className="p-1.5 hover:bg-bg-subtle rounded-lg transition-colors"
                    >
                        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                </div>
            </div>
            
            {expanded && (
                <div className="mt-3 pt-3 border-t border-white/5 space-y-4 text-xs">
                    {/* 凯利资金管理区块 */}
                    <div className={cn(
                        "p-3 rounded-lg border",
                        rule.use_kelly_sizing ? "bg-primary/5 border-primary/20" : "bg-bg-subtle/50 border-white/5"
                    )}>
                        <div className="flex items-center justify-between mb-3">
                            <span className={cn("font-medium", rule.use_kelly_sizing && "text-primary")}>
                                凯利动态资金管理
                            </span>
                            <Toggle 
                                enabled={rule.use_kelly_sizing} 
                                onChange={(val) => onUpdate(rule.id, { use_kelly_sizing: val })} 
                            />
                        </div>
                        
                        {rule.use_kelly_sizing && (
                            <div className="space-y-3">
                                {/* Kelly 分数滑块 */}
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <span className="text-text-muted">Kelly 比例</span>
                                        <span className="font-mono text-primary">{(rule.kelly_fraction * 100).toFixed(0)}%</span>
                                    </div>
                                    <input 
                                        type="range" 
                                        min="0.1" max="1" step="0.05"
                                        value={rule.kelly_fraction}
                                        onChange={(e) => onUpdate(rule.id, { kelly_fraction: parseFloat(e.target.value) })}
                                        className="w-full h-1 bg-bg-base rounded-lg appearance-none cursor-pointer accent-primary"
                                    />
                                    <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
                                        <span>保守 10%</span>
                                        <span>激进 100%</span>
                                    </div>
                                </div>
                                
                                {/* 手数范围 */}
                                <div className="grid grid-cols-2 gap-2">
                                    <div>
                                        <label className="text-[10px] text-text-muted block mb-1">最小手数</label>
                                        <input 
                                            type="number" 
                                            step="0.01" 
                                            min="0.01"
                                            value={rule.min_lot_size || 0.01}
                                            onChange={(e) => onUpdate(rule.id, { min_lot_size: parseFloat(e.target.value) })}
                                            className="w-full bg-bg-base border border-white/10 rounded px-2 py-1 text-xs font-mono"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted block mb-1">最大手数</label>
                                        <input 
                                            type="number" 
                                            step="0.01" 
                                            min="0.01"
                                            value={rule.max_lot_size}
                                            onChange={(e) => onUpdate(rule.id, { max_lot_size: parseFloat(e.target.value) })}
                                            className="w-full bg-bg-base border border-white/10 rounded px-2 py-1 text-xs font-mono"
                                        />
                                    </div>
                                </div>
                                
                                <div className="text-[10px] text-text-muted bg-bg-base/50 p-2 rounded">
                                    💡 系统将根据 AI 置信度和账户余额动态计算手数，范围限制在 {rule.min_lot_size || 0.01} - {rule.max_lot_size} 手之间
                                </div>
                            </div>
                        )}
                        
                        {!rule.use_kelly_sizing && (
                            <div className="flex items-center justify-between">
                                <span className="text-text-muted">固定手数</span>
                                <input 
                                    type="number" 
                                    step="0.01" 
                                    min="0.01"
                                    value={rule.fixed_lot_size}
                                    onChange={(e) => onUpdate(rule.id, { fixed_lot_size: parseFloat(e.target.value) })}
                                    className="w-20 bg-bg-base border border-white/10 rounded px-2 py-1 text-xs font-mono text-right"
                                />
                            </div>
                        )}
                    </div>
                    
                    {/* 其他设置 */}
                    <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="text-text-muted">最大点差</span>
                                <span className="font-mono">{rule.max_spread_points}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-muted">波动倍数</span>
                                <span className="font-mono">{rule.max_vol_mult}x</span>
                            </div>
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="text-text-muted">置信度阈值</span>
                                <span className="font-mono">{(rule.ai_confidence_threshold * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className={cn("text-text-muted", rule.risk_off && "text-danger")}>
                                    风险暂停
                                </span>
                                <Toggle 
                                    enabled={rule.risk_off} 
                                    onChange={(val) => onUpdate(rule.id, { risk_off: val })} 
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </GlassCard>
    );
}

// AI 日志列表
function AILogList({ logs }: { logs: AILog[] }) {
    if (logs.length === 0) {
        return <div className="text-center text-text-muted py-8">暂无日志</div>;
    }
    
    return (
        <div className="space-y-1">
            {logs.map((log) => (
                <div 
                    key={log.id} 
                    className="flex items-center gap-3 p-2 hover:bg-bg-subtle/50 rounded-lg transition-colors text-xs"
                >
                    <div className={cn(
                        "w-6 h-6 rounded flex items-center justify-center shrink-0",
                        log.action === 'BUY' ? "bg-success/10 text-success" :
                        log.action === 'SELL' ? "bg-danger/10 text-danger" :
                        "bg-text-muted/10 text-text-muted"
                    )}>
                        {log.action === 'BUY' ? <TrendingUp size={12} /> : 
                         log.action === 'SELL' ? <TrendingDown size={12} /> : 
                         <Minus size={12} />}
                    </div>
                    
                    <div className="flex-1 min-w-0 grid grid-cols-6 gap-2 items-center">
                        <span className="font-medium truncate">{log.symbol}</span>
                        <span className={cn(
                            "font-bold",
                            log.action === 'BUY' ? "text-success" :
                            log.action === 'SELL' ? "text-danger" : "text-text-muted"
                        )}>
                            {log.action}
                        </span>
                        <span className="font-mono text-text-secondary">${log.price?.toFixed(2)}</span>
                        <span className="text-text-muted">{log.regime || '-'}</span>
                        <span className="font-mono text-primary">{(log.aiScore * 100).toFixed(0)}%</span>
                        <span className="text-text-muted text-[10px]">
                            {new Date(log.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                        </span>
                    </div>
                </div>
            ))}
        </div>
    );
}

// 模型状态面板
function ModelStatusPanel() {
    const models = [
        { name: 'QuantumNet', version: 'V1', status: 'active', accuracy: '53.6%', icon: Brain },
        { name: 'DQN Agent', version: 'V2', status: 'active', accuracy: '48.2%', icon: Bot },
        { name: 'CatBoost Meta', version: 'V1', status: 'active', accuracy: '64.1%', icon: Layers },
        { name: 'ARIMA-GARCH', version: 'V1', status: 'active', accuracy: '-', icon: Activity },
    ];
    
    return (
        <div className="space-y-2">
            {models.map((model) => (
                <div key={model.name} className="flex items-center gap-3 p-2 bg-bg-subtle/50 rounded-lg">
                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <model.icon size={14} className="text-primary" />
                    </div>
                    <div className="flex-1">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">{model.name}</span>
                            <StatusBadge status={model.status as 'active'} size="sm" />
                        </div>
                        <div className="flex items-center justify-between text-[10px] text-text-muted">
                            <span>{model.version}</span>
                            <span>准确率: {model.accuracy}</span>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}

export default function AIManagementPage() {
    const [settings, setSettings] = useState<AISettings | null>(null);
    const [rules, setRules] = useState<AutomationRule[]>([]);
    const [logs, setLogs] = useState<AILog[]>([]);
    const [logStats, setLogStats] = useState({ total: 0, buyCount: 0, sellCount: 0, waitCount: 0 });
    const [isLoading, setIsLoading] = useState(true);
    const [logFilter, setLogFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'WAIT'>('ALL');
    const [autoRefresh, setAutoRefresh] = useState(true);
    
    // 获取配置
    const fetchConfig = useCallback(async () => {
        try {
            const res = await fetch('/api/ai/config');
            if (res.ok) {
                const data = await res.json();
                setSettings(data.settings);
                setRules(data.rules);
            }
        } catch (e) {
            console.error(e);
        }
    }, []);
    
    // 获取日志
    const fetchLogs = useCallback(async () => {
        try {
            const res = await fetch(`/api/ai/logs?limit=50&action=${logFilter}`);
            if (res.ok) {
                const data = await res.json();
                setLogs(data.logs);
                setLogStats(data.stats);
            }
        } catch (e) {
            console.error(e);
        }
    }, [logFilter]);
    
    // 更新设置
    const updateSettings = async (data: Partial<AISettings>) => {
        try {
            const res = await fetch('/api/ai/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'settings', id: settings?.id, data }),
            });
            if (res.ok) {
                setSettings(prev => prev ? { ...prev, ...data, updated_at: new Date().toISOString() } : null);
            }
        } catch (e) {
            console.error(e);
        }
    };
    
    // 更新规则
    const updateRule = async (id: string, data: Partial<AutomationRule>) => {
        try {
            const res = await fetch('/api/ai/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'rule', id, data }),
            });
            if (res.ok) {
                setRules(prev => prev.map(r => r.id === id ? { ...r, ...data } : r));
            }
        } catch (e) {
            console.error(e);
        }
    };
    
    useEffect(() => {
        setIsLoading(true);
        Promise.all([fetchConfig(), fetchLogs()]).finally(() => setIsLoading(false));
    }, [fetchConfig, fetchLogs]);
    
    // 自动刷新日志
    useEffect(() => {
        if (!autoRefresh) return;
        const interval = setInterval(fetchLogs, 5000);
        return () => clearInterval(interval);
    }, [autoRefresh, fetchLogs]);
    
    if (isLoading) {
        return (
            <div className="h-full flex items-center justify-center">
                <Brain className="animate-pulse text-primary" size={40} />
            </div>
        );
    }
    
    return (
        <div className="h-full flex flex-col gap-4 overflow-auto pb-4">
            {/* 页面标题 */}
            <div className="flex items-center justify-between shrink-0">
                <div>
                    <h1 className="text-xl font-bold tracking-tight flex items-center gap-2">
                        <Brain size={24} className="text-primary" />
                        AI 管理中心
                    </h1>
                    <p className="text-xs text-text-muted mt-0.5">配置 AI 策略、监控模型状态、查看推理日志</p>
                </div>
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-2 text-xs text-text-muted">
                        <span>自动刷新</span>
                        <Toggle enabled={autoRefresh} onChange={setAutoRefresh} />
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => { fetchConfig(); fetchLogs(); }}>
                        <RefreshCw size={14} className="mr-1" /> 刷新
                    </Button>
                </div>
            </div>
            
            {/* 状态概览 */}
            <div className="grid grid-cols-4 gap-3 shrink-0">
                <GlassCard className="p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-success/10 flex items-center justify-center">
                        <Cpu size={20} className="text-success" />
                    </div>
                    <div>
                        <div className="text-lg font-bold text-success">在线</div>
                        <div className="text-[10px] text-text-muted">AI Engine</div>
                    </div>
                </GlassCard>
                <GlassCard className="p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                        <Database size={20} className="text-primary" />
                    </div>
                    <div>
                        <div className="text-lg font-bold">{logStats.total}</div>
                        <div className="text-[10px] text-text-muted">今日信号</div>
                    </div>
                </GlassCard>
                <GlassCard className="p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-success/10 flex items-center justify-center">
                        <TrendingUp size={20} className="text-success" />
                    </div>
                    <div>
                        <div className="text-lg font-bold text-success">{logStats.buyCount}</div>
                        <div className="text-[10px] text-text-muted">买入信号</div>
                    </div>
                </GlassCard>
                <GlassCard className="p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-danger/10 flex items-center justify-center">
                        <TrendingDown size={20} className="text-danger" />
                    </div>
                    <div>
                        <div className="text-lg font-bold text-danger">{logStats.sellCount}</div>
                        <div className="text-[10px] text-text-muted">卖出信号</div>
                    </div>
                </GlassCard>
            </div>
            
            {/* 主要内容区域 */}
            <div className="grid grid-cols-12 gap-4 flex-1 min-h-0">
                {/* 左侧：配置面板 */}
                <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
                    {/* 全局配置 */}
                    <GlassCard className="p-4">
                        <CardHeader className="px-0 pt-0 pb-3">
                            <CardTitle className="flex items-center gap-2 text-sm">
                                <Settings size={16} className="text-primary" />
                                全局配置
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-0">
                            <AIConfigPanel settings={settings} onUpdate={updateSettings} />
                        </CardContent>
                    </GlassCard>
                    
                    {/* 模型状态 */}
                    <GlassCard className="p-4 flex-1">
                        <CardHeader className="px-0 pt-0 pb-3">
                            <CardTitle className="flex items-center gap-2 text-sm">
                                <Layers size={16} className="text-primary" />
                                模型状态
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-0">
                            <ModelStatusPanel />
                        </CardContent>
                    </GlassCard>
                </div>
                
                {/* 中间：AI 日志 */}
                <GlassCard className="col-span-12 lg:col-span-6 p-4 flex flex-col overflow-hidden">
                    <CardHeader className="px-0 pt-0 pb-3 shrink-0">
                        <div className="flex items-center justify-between">
                            <CardTitle className="flex items-center gap-2 text-sm">
                                <Activity size={16} className="text-primary" />
                                推理日志
                            </CardTitle>
                            <div className="flex gap-1">
                                {(['ALL', 'BUY', 'SELL', 'WAIT'] as const).map(filter => (
                                    <button
                                        key={filter}
                                        onClick={() => setLogFilter(filter)}
                                        className={cn(
                                            "px-2 py-1 text-[10px] rounded transition-colors",
                                            logFilter === filter 
                                                ? "bg-primary text-white" 
                                                : "bg-bg-subtle text-text-muted hover:text-text-primary"
                                        )}
                                    >
                                        {filter === 'ALL' ? '全部' : filter}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </CardHeader>
                    
                    {/* 表头 */}
                    <div className="grid grid-cols-6 gap-2 px-2 py-1.5 text-[10px] font-medium text-text-muted border-b border-white/5 shrink-0">
                        <span>品种</span>
                        <span>信号</span>
                        <span>价格</span>
                        <span>Regime</span>
                        <span>置信度</span>
                        <span>时间</span>
                    </div>
                    
                    <CardContent className="p-0 flex-1 overflow-auto">
                        <AILogList logs={logs} />
                    </CardContent>
                </GlassCard>
                
                {/* 右侧：自动化规则 */}
                <div className="col-span-12 lg:col-span-3 flex flex-col gap-3 overflow-auto">
                    <div className="flex items-center justify-between shrink-0">
                        <h3 className="text-sm font-bold flex items-center gap-2">
                            <Target size={16} className="text-primary" />
                            自动化规则
                        </h3>
                    </div>
                    {rules.length === 0 ? (
                        <div className="text-center text-text-muted py-8">暂无规则</div>
                    ) : (
                        <div className="space-y-2">
                            {rules.map(rule => (
                                <RuleCard key={rule.id} rule={rule} onUpdate={updateRule} />
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

