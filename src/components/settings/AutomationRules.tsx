"use client";

import { useState, useEffect } from 'react';
import { Card } from '@/components/Card';
import { supabase } from '@/lib/supabase';
import { Plus, Trash2, Save, AlertTriangle, Bot } from 'lucide-react';
import { Toast } from '@/components/ui/Toast';

interface AutomationRule {
    id?: string;
    symbol: string;
    is_enabled: boolean;
    fixed_lot_size: number;
    max_spread_points: number;
    strategy_id?: string;
}

export function AutomationRules() {
    const [rules, setRules] = useState<AutomationRule[]>([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [showToast, setShowToast] = useState(false);

    useEffect(() => {
        fetchRules();
    }, []);

    const fetchRules = async () => {
        const { data } = await supabase
            .from('automation_rules')
            .select('*')
            .order('symbol');
        
        if (data && data.length > 0) {
            // Ensure proper types
            const typedData = data.map(r => ({
                ...r,
                fixed_lot_size: Number(r.fixed_lot_size),
                max_spread_points: Number(r.max_spread_points)
            }));
            setRules(typedData);
        } else {
            // Default GLOBAL rule if empty
            setRules([{
                symbol: 'GLOBAL',
                is_enabled: false,
                fixed_lot_size: 0.01,
                max_spread_points: 50
            }]);
        }
        setLoading(false);
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            for (const rule of rules) {
                // Basic validation
                if (!rule.symbol) continue;

                const payload = {
                    symbol: rule.symbol,
                    is_enabled: rule.is_enabled,
                    fixed_lot_size: rule.fixed_lot_size,
                    max_spread_points: rule.max_spread_points
                };

                if (rule.id) {
                    const { error } = await supabase
                        .from('automation_rules')
                        .update(payload)
                        .eq('id', rule.id);
                    if (error) throw error;
                } else {
                    const { error } = await supabase
                        .from('automation_rules')
                        .insert(payload);
                    if (error) throw error;
                }
            }
            setShowToast(true);
            fetchRules(); // Refresh to get IDs
        } catch (e: any) {
            console.error("Failed to save rules", e);
            alert(`保存失败: ${e.message || "未知错误，请检查 RLS 策略"}`);
        } finally {
            setSaving(false);
        }
    };

    const addRule = () => {
        setRules([...rules, {
            symbol: 'EURUSD',
            is_enabled: false,
            fixed_lot_size: 0.01,
            max_spread_points: 50
        }]);
    };

    const removeRule = async (index: number) => {
        const rule = rules[index];
        if (rule.id) {
            const confirmed = window.confirm(`确定要删除 ${rule.symbol} 的规则吗？`);
            if (!confirmed) return;
            
            await supabase.from('automation_rules').delete().eq('id', rule.id);
        }
        const newRules = [...rules];
        newRules.splice(index, 1);
        setRules(newRules);
    };

    const updateRule = (index: number, field: keyof AutomationRule, value: string | number | boolean) => {
        const newRules = [...rules];
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore - Dynamic key assignment
        newRules[index] = { ...newRules[index], [field]: value };
        setRules(newRules);
    };

    if (loading) return (
        <Card>
            <div className="p-8 text-center text-slate-400">Loading automation rules...</div>
        </Card>
    );

    return (
        <Card>
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                        <Bot size={24} className="text-accent-primary" />
                        自动化策略执行
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">配置 MQL5 信号的自动下单规则</p>
                </div>
                <button 
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center gap-2 px-4 py-2 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                    <Save size={18} />
                    {saving ? '保存中...' : '保存配置'}
                </button>
            </div>

            <div className="space-y-4">
                {rules.map((rule, index) => (
                    <div key={index} className="grid grid-cols-1 md:grid-cols-12 gap-4 p-4 bg-white/5 rounded-lg border border-surface-border items-center">
                        
                        {/* Symbol */}
                        <div className="md:col-span-3">
                            <label className="text-xs text-slate-500 mb-1 block">交易品种</label>
                            <input 
                                type="text" 
                                value={rule.symbol}
                                onChange={(e) => updateRule(index, 'symbol', e.target.value.toUpperCase())}
                                className="w-full bg-black/20 border border-surface-border rounded px-3 py-2 text-white font-mono uppercase"
                                placeholder="GLOBAL"
                            />
                        </div>

                        {/* Lot Size */}
                        <div className="md:col-span-2">
                            <label className="text-xs text-slate-500 mb-1 block">固定手数</label>
                            <input 
                                type="number" 
                                step="0.01"
                                min="0.01"
                                value={rule.fixed_lot_size}
                                onChange={(e) => updateRule(index, 'fixed_lot_size', parseFloat(e.target.value))}
                                className="w-full bg-black/20 border border-surface-border rounded px-3 py-2 text-white font-mono"
                            />
                        </div>

                        {/* Max Spread */}
                        <div className="md:col-span-2">
                            <label className="text-xs text-slate-500 mb-1 block">最大点差 (Points)</label>
                            <input 
                                type="number" 
                                min="0"
                                value={rule.max_spread_points}
                                onChange={(e) => updateRule(index, 'max_spread_points', parseInt(e.target.value))}
                                className="w-full bg-black/20 border border-surface-border rounded px-3 py-2 text-white font-mono"
                            />
                        </div>

                        {/* Enabled Switch */}
                        <div className="md:col-span-3 flex items-center justify-center gap-3 h-full pt-3">
                             <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={rule.is_enabled}
                                  onChange={(e) => updateRule(index, 'is_enabled', e.target.checked)}
                                  className="sr-only peer"
                                />
                                <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent-primary"></div>
                                <span className={`ml-3 text-sm font-medium ${rule.is_enabled ? 'text-accent-success' : 'text-slate-500'}`}>
                                    {rule.is_enabled ? '已启用' : '已禁用'}
                                </span>
                              </label>
                        </div>

                        {/* Delete */}
                        <div className="md:col-span-2 flex justify-end md:pt-3">
                            <button 
                                onClick={() => removeRule(index)}
                                className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded transition-colors"
                            >
                                <Trash2 size={18} />
                            </button>
                        </div>
                    </div>
                ))}

                <button 
                    onClick={addRule}
                    className="w-full py-3 border border-dashed border-slate-700 rounded-lg text-slate-400 hover:text-white hover:border-slate-500 hover:bg-white/5 transition-all flex items-center justify-center gap-2"
                >
                    <Plus size={18} />
                    添加规则
                </button>
            </div>

            <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg flex gap-3">
                <AlertTriangle className="text-yellow-500 shrink-0" size={24} />
                <div>
                    <h4 className="text-yellow-500 font-medium mb-1">风险提示</h4>
                    <p className="text-sm text-slate-400 leading-relaxed">
                        启用自动交易意味着系统将根据 MQL5 信号直接下单，无需人工确认。
                        <br />
                        1. 请确保您的 VPS/网络连接稳定。
                        <br />
                        2. 建议先使用小手数 (0.01) 进行测试。
                        <br />
                        3. &quot;GLOBAL&quot; 规则适用于所有未单独配置的品种。
                    </p>
                </div>
            </div>

            <Toast 
                open={showToast} 
                onOpenChange={setShowToast} 
                title="设置已保存" 
                description="自动化规则已成功更新并同步至交易桥。"
            />
        </Card>
    );
}

