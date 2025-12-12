"use client";

import { useState, useEffect } from 'react';
import { Card } from '@/components/Card';
import { supabase } from '@/lib/supabase';
import { Plus, Trash2, Save, AlertTriangle, Bot, CheckCircle, XCircle, Settings2, Activity } from 'lucide-react';
import { Toast } from '@/components/ui/Toast';

interface AutomationRule {
    id?: string;
    symbol: string;
    is_enabled: boolean;
    fixed_lot_size: number;
    max_spread_points: number;
    ai_mode?: 'legacy' | 'indicator_ai' | 'pure_ai' | 'dom_ai';
    ai_confidence_threshold?: number;
    strategy_id?: string;
    // Kelly Criterion fields
    use_kelly_sizing?: boolean;
    kelly_fraction?: number;
    kelly_lookback_trades?: number;
    max_lot_size?: number;
    max_daily_loss?: number;
}

export function AutomationRules() {
    const [rules, setRules] = useState<AutomationRule[]>([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [showToast, setShowToast] = useState(false);
    const [activeRuleId, setActiveRuleId] = useState<string | null>(null);

    useEffect(() => {
        fetchRules();
    }, []);

    const fetchRules = async () => {
        const { data } = await supabase
            .from('automation_rules')
            .select('*')
            .order('symbol');

        if (data && data.length > 0) {
            const typedData = data.map(r => ({
                ...r,
                fixed_lot_size: Number(r.fixed_lot_size),
                max_spread_points: Number(r.max_spread_points),
                ai_mode: r.ai_mode || 'pure_ai',
                ai_confidence_threshold: r.ai_confidence_threshold !== null ? Number(r.ai_confidence_threshold) : 0.75,
                // Kelly fields
                use_kelly_sizing: r.use_kelly_sizing || false,
                kelly_fraction: r.kelly_fraction !== null ? Number(r.kelly_fraction) : 0.25,
                kelly_lookback_trades: r.kelly_lookback_trades || 50,
                max_lot_size: r.max_lot_size !== null ? Number(r.max_lot_size) : 1.0,
                max_daily_loss: r.max_daily_loss !== null ? Number(r.max_daily_loss) : 0
            }));
            setRules(typedData);
        } else {
            setRules([{
                symbol: 'GLOBAL',
                is_enabled: false,
                fixed_lot_size: 0.01,
                max_spread_points: 50,
                ai_mode: 'pure_ai',
                ai_confidence_threshold: 0.75
            }]);
        }
        setLoading(false);
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            for (const rule of rules) {
                if (!rule.symbol) continue;

                const payload = {
                    symbol: rule.symbol,
                    is_enabled: rule.is_enabled,
                    fixed_lot_size: rule.fixed_lot_size,
                    max_spread_points: rule.max_spread_points,
                    ai_mode: rule.ai_mode,
                    ai_confidence_threshold: rule.ai_confidence_threshold,
                    // Kelly fields
                    use_kelly_sizing: rule.use_kelly_sizing || false,
                    kelly_fraction: rule.kelly_fraction || 0.25,
                    kelly_lookback_trades: rule.kelly_lookback_trades || 50,
                    max_lot_size: rule.max_lot_size || 1.0,
                    max_daily_loss: rule.max_daily_loss || 0
                };

                if (rule.id) {
                    await supabase.from('automation_rules').update(payload).eq('id', rule.id);
                } else {
                    await supabase.from('automation_rules').insert(payload);
                }
            }
            setShowToast(true);
            fetchRules();
        } catch (e: any) {
            console.error("Failed to save rules", e);
            alert(`保存失败: ${e.message}`);
        } finally {
            setSaving(false);
        }
    };

    const addRule = () => {
        const newRule: AutomationRule = {
            symbol: 'NEW',
            is_enabled: false,
            fixed_lot_size: 0.01,
            max_spread_points: 50,
            ai_mode: 'pure_ai',
            ai_confidence_threshold: 0.75,
            // Kelly defaults
            use_kelly_sizing: false,
            kelly_fraction: 0.25,
            kelly_lookback_trades: 50,
            max_lot_size: 1.0,
            max_daily_loss: 0
        };
        setRules([...rules, newRule]);
        setActiveRuleId('NEW'); // Auto expand
    };

    const removeRule = async (index: number) => {
        const rule = rules[index];
        if (rule.id) {
            if (!confirm(`确定要删除 ${rule.symbol} 的规则吗？`)) return;
            await supabase.from('automation_rules').delete().eq('id', rule.id);
        }
        const newRules = [...rules];
        newRules.splice(index, 1);
        setRules(newRules);
    };

    const updateRule = (index: number, field: keyof AutomationRule, value: any) => {
        const newRules = [...rules];
        newRules[index] = { ...newRules[index], [field]: value };
        setRules(newRules);
    };

    if (loading) return (
        <Card className="animate-pulse">
            <div className="h-48 bg-white/5 rounded-lg"></div>
        </Card>
    );

    return (
        <div className="space-y-6">
            <Card>
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h2 className="text-xl font-bold text-white flex items-center gap-3">
                            <Bot className="text-accent-primary" size={28} />
                            自动化策略矩阵
                        </h2>
                        <p className="text-slate-400 text-sm mt-2">配置 AI 交易引擎的介入程度与风控参数</p>
                    </div>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex items-center gap-2 px-6 py-2.5 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-xl font-medium transition-all shadow-lg shadow-accent-primary/20 disabled:opacity-50 disabled:shadow-none"
                    >
                        {saving ? <Activity className="animate-spin" size={18} /> : <Save size={18} />}
                        {saving ? '保存中...' : '保存配置'}
                    </button>
                </div>

                <div className="grid gap-4">
                    {rules.map((rule, index) => {
                        const isExpanded = activeRuleId === rule.id || activeRuleId === rule.symbol;

                        return (
                            <div
                                key={index}
                                className={`group rounded-xl border transition-all duration-200 overflow-hidden ${rule.is_enabled
                                    ? 'bg-accent-primary/5 border-accent-primary/30'
                                    : 'bg-surface-glass border-surface-border hover:border-slate-600'
                                    }`}
                            >
                                {/* Header Row */}
                                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setActiveRuleId(isExpanded ? null : (rule.id || rule.symbol))}>
                                    <div className="flex items-center gap-4">
                                        <div className={`w-2 h-2 rounded-full ${rule.is_enabled ? 'bg-accent-success shadow-[0_0_8px_rgba(34,197,94,0.5)]' : 'bg-slate-600'}`} />

                                        <div className="flex flex-col">
                                            <span className="font-mono text-lg font-bold text-white">{rule.symbol}</span>
                                            <span className="text-xs text-slate-500">
                                                {rule.ai_mode === 'pure_ai' ? '🚀 Full Auto (Scanner)' :
                                                    rule.ai_mode === 'indicator_ai' ? '🛡️ AI Filter (Advisory)' :
                                                        '⚠️ Legacy'}
                                            </span>
                                        </div>

                                        <div className="h-6 w-[1px] bg-white/10 mx-2 hidden md:block"></div>

                                        <div className="hidden md:flex items-center gap-4 text-sm text-slate-400">
                                            <span className="flex items-center gap-1.5 bg-white/5 px-2 py-1 rounded">
                                                <Settings2 size={14} />
                                                {rule.fixed_lot_size} lots
                                            </span>
                                            {rule.ai_mode !== 'legacy' && (
                                                <span className="flex items-center gap-1.5 bg-white/5 px-2 py-1 rounded">
                                                    <Activity size={14} />
                                                    {rule.ai_confidence_threshold} Conf
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <div
                                            className="relative inline-flex h-6 w-11 items-center rounded-full bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-slate-900 cursor-pointer"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                updateRule(index, 'is_enabled', !rule.is_enabled);
                                            }}
                                        >
                                            <span
                                                className={`inline-block h-4 w-4 transform rounded-full bg-white transition duration-200 ease-in-out ${rule.is_enabled ? 'translate-x-6' : 'translate-x-1'
                                                    }`}
                                            />
                                            <span className={`absolute bg-accent-primary inset-0 rounded-full transition-opacity duration-200 ${rule.is_enabled ? 'opacity-100' : 'opacity-0'}`} />
                                            <span className={`absolute h-4 w-4 bg-white rounded-full transition-transform duration-200 top-1 left-1 ${rule.is_enabled ? 'translate-x-5' : 'translate-x-0'}`} />
                                        </div>
                                    </div>
                                </div>

                                {/* Expanded Detail Panel */}
                                {isExpanded && (
                                    <div className="border-t border-white/5 bg-black/20 p-6 space-y-6 animate-fade-in">
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                                            {/* Symbol Input */}
                                            <div>
                                                <label className="text-xs text-slate-500 font-medium mb-2 block uppercase tracking-wider">交易品种</label>
                                                <input
                                                    type="text"
                                                    value={rule.symbol}
                                                    onChange={(e) => updateRule(index, 'symbol', e.target.value.toUpperCase())}
                                                    className="w-full bg-surface-glass border border-surface-border rounded-lg px-4 py-2.5 text-white font-mono placeholder-slate-600 focus:border-accent-primary focus:ring-1 focus:ring-accent-primary outline-none transition-all"
                                                    placeholder="e.g. BTCUSD"
                                                />
                                            </div>

                                            {/* AI Mode Selector */}
                                            <div>
                                                <label className="text-xs text-slate-500 font-medium mb-2 block uppercase tracking-wider">AI 介入模式</label>
                                                <select
                                                    value={rule.ai_mode || 'pure_ai'}
                                                    onChange={(e) => updateRule(index, 'ai_mode', e.target.value)}
                                                    className="w-full bg-surface-glass border border-surface-border rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-primary outline-none appearance-none cursor-pointer"
                                                >
                                                    <option value="pure_ai">🚀 Full Auto (全自动 AI)</option>
                                                    <option value="indicator_ai">🛡️ AI Filter (仅过滤信号)</option>
                                                    <option value="legacy">⚙️ Legacy (纯指标)</option>
                                                </select>
                                                <p className="text-[10px] text-slate-500 mt-1.5">
                                                    {rule.ai_mode === 'pure_ai' && 'AI Scanner 实时扫描并由 AI 引擎全权决策交易。'}
                                                    {rule.ai_mode === 'indicator_ai' && '仅当指标信号产生时，使用 AI 进行二次确认。'}
                                                    {rule.ai_mode === 'legacy' && '不使用 AI 模型，完全依赖 MQL5 指标信号。'}
                                                </p>
                                            </div>

                                            {/* Lot Size */}
                                            <div>
                                                <label className="text-xs text-slate-500 font-medium mb-2 block uppercase tracking-wider">固定交易手数</label>
                                                <div className="relative">
                                                    <input
                                                        type="number"
                                                        step="0.01"
                                                        min="0.01"
                                                        value={rule.fixed_lot_size}
                                                        onChange={(e) => updateRule(index, 'fixed_lot_size', parseFloat(e.target.value))}
                                                        className="w-full bg-surface-glass border border-surface-border rounded-lg px-4 py-2.5 text-white font-mono focus:border-accent-primary outline-none"
                                                    />
                                                    <span className="absolute right-4 top-2.5 text-slate-500 text-sm">Lots</span>
                                                </div>
                                                {/* Smart Warning for Indices */}
                                                {['US30', 'NAS100', 'SPX500', 'GER30', 'UK100'].some(s => rule.symbol?.includes(s)) && rule.fixed_lot_size < 0.1 && (
                                                    <div className="flex items-center gap-2 mt-2 text-orange-400 text-[10px]">
                                                        <AlertTriangle size={12} />
                                                        <span>指数类商品最小手数通常为 0.1 或 1.0</span>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Confidence Threshold */}
                                            {rule.ai_mode !== 'legacy' && (
                                                <div className="space-y-3">
                                                    <div className="flex justify-between">
                                                        <label className="text-xs text-slate-500 font-medium uppercase tracking-wider">最低信心阈值</label>
                                                        <span className="text-xs font-mono text-accent-primary bg-accent-primary/10 px-1.5 rounded">
                                                            {(rule.ai_confidence_threshold || 0.75).toFixed(2)}
                                                        </span>
                                                    </div>
                                                    <input
                                                        type="range"
                                                        min="0.5"
                                                        max="0.95"
                                                        step="0.05"
                                                        value={rule.ai_confidence_threshold || 0.75}
                                                        onChange={(e) => updateRule(index, 'ai_confidence_threshold', parseFloat(e.target.value))}
                                                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-accent-primary hover:accent-accent-primary/80"
                                                    />
                                                    <div className="flex justify-between text-[10px] text-slate-600">
                                                        <span>宽松 (0.5)</span>
                                                        <span>严格 (0.95)</span>
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        {/* Kelly Criterion Position Sizing */}
                                        <div className="border-t border-white/5 pt-6 mt-6">
                                            <div className="flex items-center justify-between mb-4">
                                                <div>
                                                    <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                                                        📊 凯利公式仓位管理
                                                    </h4>
                                                    <p className="text-[10px] text-slate-500 mt-1">根据历史胜率动态计算最优仓位</p>
                                                </div>
                                                <div
                                                    className="relative inline-flex h-6 w-11 items-center rounded-full bg-slate-700 transition-colors cursor-pointer"
                                                    onClick={() => updateRule(index, 'use_kelly_sizing', !rule.use_kelly_sizing)}
                                                >
                                                    <span className={`absolute bg-accent-primary inset-0 rounded-full transition-opacity duration-200 ${rule.use_kelly_sizing ? 'opacity-100' : 'opacity-0'}`} />
                                                    <span className={`absolute h-4 w-4 bg-white rounded-full transition-transform duration-200 top-1 left-1 ${rule.use_kelly_sizing ? 'translate-x-5' : 'translate-x-0'}`} />
                                                </div>
                                            </div>

                                            {rule.use_kelly_sizing && (
                                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 animate-fade-in">
                                                    {/* Kelly Fraction */}
                                                    <div className="space-y-2">
                                                        <div className="flex justify-between">
                                                            <label className="text-xs text-slate-500 font-medium">凯利系数</label>
                                                            <span className="text-xs font-mono text-accent-primary bg-accent-primary/10 px-1.5 rounded">
                                                                {(rule.kelly_fraction || 0.25).toFixed(2)}
                                                            </span>
                                                        </div>
                                                        <input
                                                            type="range"
                                                            min="0.1"
                                                            max="1.0"
                                                            step="0.05"
                                                            value={rule.kelly_fraction || 0.25}
                                                            onChange={(e) => updateRule(index, 'kelly_fraction', parseFloat(e.target.value))}
                                                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-accent-primary"
                                                        />
                                                        <div className="flex justify-between text-[10px] text-slate-600">
                                                            <span>保守 (0.1)</span>
                                                            <span>激进 (1.0)</span>
                                                        </div>
                                                    </div>

                                                    {/* Lookback Trades */}
                                                    <div>
                                                        <label className="text-xs text-slate-500 font-medium mb-2 block">回溯交易数</label>
                                                        <input
                                                            type="number"
                                                            min="10"
                                                            max="200"
                                                            value={rule.kelly_lookback_trades || 50}
                                                            onChange={(e) => updateRule(index, 'kelly_lookback_trades', parseInt(e.target.value))}
                                                            className="w-full bg-surface-glass border border-surface-border rounded-lg px-4 py-2 text-white font-mono text-sm focus:border-accent-primary outline-none"
                                                        />
                                                    </div>

                                                    {/* Max Lot Size */}
                                                    <div>
                                                        <label className="text-xs text-slate-500 font-medium mb-2 block">最大手数上限</label>
                                                        <input
                                                            type="number"
                                                            step="0.1"
                                                            min="0.01"
                                                            value={rule.max_lot_size || 1.0}
                                                            onChange={(e) => updateRule(index, 'max_lot_size', parseFloat(e.target.value))}
                                                            className="w-full bg-surface-glass border border-surface-border rounded-lg px-4 py-2 text-white font-mono text-sm focus:border-accent-primary outline-none"
                                                        />
                                                    </div>

                                                    {/* Max Daily Loss */}
                                                    <div>
                                                        <label className="text-xs text-slate-500 font-medium mb-2 block">每日最大亏损 ($)</label>
                                                        <input
                                                            type="number"
                                                            min="0"
                                                            step="10"
                                                            value={rule.max_daily_loss || 0}
                                                            onChange={(e) => updateRule(index, 'max_daily_loss', parseFloat(e.target.value))}
                                                            className="w-full bg-surface-glass border border-surface-border rounded-lg px-4 py-2 text-white font-mono text-sm focus:border-accent-primary outline-none"
                                                        />
                                                        <p className="text-[10px] text-slate-600 mt-1">0 = 不限制</p>
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        <div className="flex justify-between items-center pt-4 border-t border-white/5">
                                            <p className="text-xs text-slate-500 italic">
                                                ID: {rule.id || 'Pending Save'}
                                            </p>
                                            <button
                                                onClick={() => removeRule(index)}
                                                className="flex items-center gap-2 px-4 py-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors text-sm"
                                            >
                                                <Trash2 size={16} />
                                                删除此规则
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                <div className="mt-8">
                    <button
                        onClick={addRule}
                        className="w-full py-4 border border-dashed border-slate-700/50 rounded-xl text-slate-400 hover:text-white hover:border-accent-primary/50 hover:bg-accent-primary/5 transition-all duration-300 flex items-center justify-center gap-3 group"
                    >
                        <div className="p-1 bg-slate-800 rounded-full group-hover:bg-accent-primary/20 transition-colors">
                            <Plus size={20} className="group-hover:text-accent-primary" />
                        </div>
                        <span className="font-medium">添加新的交易品种规则</span>
                    </button>
                </div>

                <Toast
                    open={showToast}
                    onOpenChange={setShowToast}
                    title="配置已同步"
                    description="自动化规则已成功更新并下发至交易执行引擎。"
                />
            </Card>

            {/* Risk Disclaimer */}
            <div className="mt-6 flex gap-4 p-4 rounded-xl bg-orange-500/5 border border-orange-500/20">
                <AlertTriangle className="text-orange-500 shrink-0 mt-1" size={20} />
                <div className="space-y-1">
                    <h4 className="text-sm font-semibold text-orange-500">高频交易风险提示</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">
                        启用 "Full Auto" 模式意味着 AI 将拥有完整的交易执行权限（只要满足预设的信心阈值）。
                        <br />
                        系统内置了 <span className="text-slate-300">Max Position Limit (2)</span> 和 <span className="text-slate-300">Daily Loss Limit</span> 硬性风控，
                        但请务必先在模拟环境中充分测试您的配置参数。
                    </p>
                </div>
            </div>
        </div>
    );
}
