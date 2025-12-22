"use client";

import React, { useState, useEffect } from 'react';
import { GlassCard, CardHeader, CardTitle, CardContent } from "@/components/ui/GlassCard";
import { Button } from "@/components/ui/Button";
import { 
    Palette, Monitor, Moon, Sun, Sparkles, Save, 
    Check, Layout, Type, Gauge, Bell, Globe 
} from 'lucide-react';
import { cn } from "@/lib/utils";

// 主题类型
type ThemeMode = 'dark' | 'light' | 'system';

interface SystemSettings {
    theme: ThemeMode;
    accentColor: string;
    fontSize: 'small' | 'medium' | 'large';
    compactMode: boolean;
    animations: boolean;
    notifications: boolean;
    language: 'zh-CN' | 'en-US';
}

const ACCENT_COLORS = [
    { name: '科技蓝', value: '#2563eb', class: 'bg-blue-500' },
    { name: '翠绿', value: '#10b981', class: 'bg-emerald-500' },
    { name: '琥珀', value: '#f59e0b', class: 'bg-amber-500' },
    { name: '玫红', value: '#ec4899', class: 'bg-pink-500' },
    { name: '紫罗兰', value: '#8b5cf6', class: 'bg-violet-500' },
    { name: '橙色', value: '#f97316', class: 'bg-orange-500' },
];

export default function SettingsPage() {
    const [settings, setSettings] = useState<SystemSettings>({
        theme: 'dark',
        accentColor: '#2563eb',
        fontSize: 'medium',
        compactMode: false,
        animations: true,
        notifications: true,
        language: 'zh-CN',
    });
    
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);

    // 加载设置
    useEffect(() => {
        try {
            const stored = localStorage.getItem('alphaos-settings');
            if (stored) {
                setSettings(JSON.parse(stored));
            }
        } catch (e) {
            console.error('Failed to parse settings', e);
        }
    }, []);

    // 保存设置
    const handleSave = async () => {
        setSaving(true);
        await new Promise(r => setTimeout(r, 300)); // 模拟保存延迟
        localStorage.setItem('alphaos-settings', JSON.stringify(settings));
            setSaving(false);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    const updateSetting = <K extends keyof SystemSettings>(key: K, value: SystemSettings[K]) => {
        setSettings(prev => ({ ...prev, [key]: value }));
    };

    return (
        <div className="max-w-3xl mx-auto flex flex-col gap-6 pb-20">
            {/* 页面标题 */}
            <div className="flex items-center justify-between sticky top-0 z-20 bg-bg-base/80 backdrop-blur-lg py-4 border-b border-white/5">
                <div>
                    <h1 className="text-2xl font-bold tracking-tight">系统设置</h1>
                    <p className="text-xs text-text-secondary">个性化您的界面风格与偏好</p>
                </div>
                <Button 
                    size="sm" 
                    onClick={handleSave} 
                    isLoading={saving} 
                    variant={saved ? "secondary" : "primary"} 
                    className={cn(saved && "bg-success/20 text-success border-success/30")}
                >
                    {saved ? <Check size={14} className="mr-2" /> : <Save size={14} className="mr-2" />}
                    {saved ? '已保存' : '保存更改'}
                    </Button>
            </div>

            {/* 主题模式 */}
            <GlassCard>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <Monitor size={18} className="text-primary" />
                        主题模式
                    </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                    <div className="grid grid-cols-3 gap-3">
                        {[
                            { mode: 'dark' as ThemeMode, icon: Moon, label: '深色模式', desc: '护眼暗色主题' },
                            { mode: 'light' as ThemeMode, icon: Sun, label: '浅色模式', desc: '明亮清新风格' },
                            { mode: 'system' as ThemeMode, icon: Monitor, label: '跟随系统', desc: '自动切换主题' },
                        ].map(({ mode, icon: Icon, label, desc }) => (
                            <button
                                key={mode}
                                onClick={() => updateSetting('theme', mode)}
                                className={cn(
                                    "p-4 rounded-xl border transition-all text-left",
                                    settings.theme === mode 
                                        ? "border-primary bg-primary/10 shadow-[0_0_20px_rgba(37,99,235,0.15)]" 
                                        : "border-white/5 bg-white/[0.02] hover:border-white/10"
                                )}
                            >
                                <Icon size={24} className={cn("mb-2", settings.theme === mode ? "text-primary" : "text-text-muted")} />
                                <div className="font-medium text-sm">{label}</div>
                                <div className="text-[10px] text-text-muted mt-0.5">{desc}</div>
                            </button>
                        ))}
            </div>
                </CardContent>
            </GlassCard>

            {/* 强调色 */}
            <GlassCard>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <Palette size={18} className="text-primary" />
                        强调色
                    </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                    <div className="flex flex-wrap gap-3">
                        {ACCENT_COLORS.map(color => (
                            <button
                                key={color.value}
                                onClick={() => updateSetting('accentColor', color.value)}
                                className={cn(
                                    "flex items-center gap-2 px-4 py-2 rounded-lg border transition-all",
                                    settings.accentColor === color.value
                                        ? "border-white/20 bg-white/10"
                                        : "border-transparent bg-white/[0.02] hover:bg-white/5"
                                )}
                            >
                                <div 
                                    className={cn("w-5 h-5 rounded-full shadow-lg", color.class)}
                                    style={{ boxShadow: settings.accentColor === color.value ? `0 0 12px ${color.value}` : undefined }}
                                />
                                <span className="text-sm">{color.name}</span>
                                {settings.accentColor === color.value && (
                                    <Check size={14} className="text-success ml-1" />
                                )}
                            </button>
                        ))}
                    </div>
                </CardContent>
            </GlassCard>

            {/* 界面布局 */}
            <GlassCard>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <Layout size={18} className="text-primary" />
                        界面布局
                    </CardTitle>
                </CardHeader>
                <CardContent className="pt-0 space-y-4">
                    {/* 字体大小 */}
                                    <div>
                        <label className="text-xs font-medium text-text-muted mb-2 block">字体大小</label>
                        <div className="grid grid-cols-3 gap-2">
                            {[
                                { size: 'small' as const, label: '小', sample: 'Aa' },
                                { size: 'medium' as const, label: '中', sample: 'Aa' },
                                { size: 'large' as const, label: '大', sample: 'Aa' },
                            ].map(({ size, label, sample }) => (
                                <button
                                    key={size}
                                    onClick={() => updateSetting('fontSize', size)}
                                    className={cn(
                                        "p-3 rounded-lg border transition-all flex flex-col items-center gap-1",
                                        settings.fontSize === size 
                                            ? "border-primary bg-primary/10" 
                                            : "border-white/5 bg-white/[0.02] hover:border-white/10"
                                    )}
                                >
                                    <span className={cn(
                                        "font-mono font-bold",
                                        size === 'small' && "text-sm",
                                        size === 'medium' && "text-base",
                                        size === 'large' && "text-lg"
                                    )}>{sample}</span>
                                    <span className="text-[10px] text-text-muted">{label}</span>
                                </button>
                            ))}
                                        </div>
                                    </div>

                    {/* 紧凑模式 */}
                    <div className="flex items-center justify-between p-3 bg-bg-subtle/50 rounded-lg">
                        <div className="flex items-center gap-3">
                            <Gauge size={18} className="text-text-muted" />
                            <div>
                                <div className="font-medium text-sm">紧凑模式</div>
                                <div className="text-[10px] text-text-muted">减少间距，显示更多内容</div>
                            </div>
                                        </div>
                        <button
                            onClick={() => updateSetting('compactMode', !settings.compactMode)}
                            className={cn(
                                "w-10 h-5 rounded-full transition-colors relative",
                                settings.compactMode ? "bg-primary" : "bg-bg-subtle"
                            )}
                        >
                            <div className={cn(
                                "absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform",
                                settings.compactMode ? "translate-x-5" : "translate-x-0.5"
                            )} />
                        </button>
                                        </div>
                </CardContent>
            </GlassCard>

            {/* 动效与通知 */}
            <GlassCard>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <Sparkles size={18} className="text-primary" />
                        动效与通知
                    </CardTitle>
                </CardHeader>
                <CardContent className="pt-0 space-y-3">
                    {/* 动画效果 */}
                    <div className="flex items-center justify-between p-3 bg-bg-subtle/50 rounded-lg">
                                                <div className="flex items-center gap-3">
                            <Sparkles size={18} className="text-text-muted" />
                            <div>
                                <div className="font-medium text-sm">动画效果</div>
                                <div className="text-[10px] text-text-muted">启用界面过渡动画</div>
                                                </div>
                                            </div>
                        <button
                            onClick={() => updateSetting('animations', !settings.animations)}
                            className={cn(
                                "w-10 h-5 rounded-full transition-colors relative",
                                settings.animations ? "bg-primary" : "bg-bg-subtle"
                            )}
                        >
                            <div className={cn(
                                "absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform",
                                settings.animations ? "translate-x-5" : "translate-x-0.5"
                            )} />
                        </button>
                                    </div>

                    {/* 系统通知 */}
                    <div className="flex items-center justify-between p-3 bg-bg-subtle/50 rounded-lg">
                        <div className="flex items-center gap-3">
                            <Bell size={18} className="text-text-muted" />
                            <div>
                                <div className="font-medium text-sm">系统通知</div>
                                <div className="text-[10px] text-text-muted">接收交易信号和系统提醒</div>
                                    </div>
                                </div>
                        <button
                            onClick={() => updateSetting('notifications', !settings.notifications)}
                            className={cn(
                                "w-10 h-5 rounded-full transition-colors relative",
                                settings.notifications ? "bg-primary" : "bg-bg-subtle"
                            )}
                        >
                            <div className={cn(
                                "absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform",
                                settings.notifications ? "translate-x-5" : "translate-x-0.5"
                            )} />
                        </button>
                    </div>
                </CardContent>
                        </GlassCard>

            {/* 语言设置 */}
            <GlassCard>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <Globe size={18} className="text-primary" />
                        语言
                    </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                    <div className="grid grid-cols-2 gap-3">
                        {[
                            { lang: 'zh-CN' as const, label: '简体中文', flag: '🇨🇳' },
                            { lang: 'en-US' as const, label: 'English', flag: '🇺🇸' },
                        ].map(({ lang, label, flag }) => (
                            <button
                                key={lang}
                                onClick={() => updateSetting('language', lang)}
                                className={cn(
                                    "p-3 rounded-lg border transition-all flex items-center gap-3",
                                    settings.language === lang 
                                        ? "border-primary bg-primary/10" 
                                        : "border-white/5 bg-white/[0.02] hover:border-white/10"
                                )}
                            >
                                <span className="text-2xl">{flag}</span>
                                <span className="font-medium text-sm">{label}</span>
                                {settings.language === lang && (
                                    <Check size={14} className="text-success ml-auto" />
                                )}
                            </button>
                        ))}
            </div>
                </CardContent>
            </GlassCard>

            {/* 版本信息 */}
            <div className="text-center text-[10px] text-text-muted pt-4 border-t border-white/5">
                <p>AlphaOS 量化交易系统 · v0.1.0</p>
                <p className="mt-1">AI 交易配置请前往 <a href="/ai" className="text-primary hover:underline">AI 管理中心</a></p>
            </div>
        </div>
    );
}
