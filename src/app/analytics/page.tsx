"use client";

import { Card } from "@/components/Card";
import { BarChart2, PieChart, TrendingUp, Activity } from "lucide-react";

export default function AnalyticsPage() {
    return (
        <div className="space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">数据分析</h1>
                    <p className="text-slate-400 mt-2">深入分析您的交易指标</p>
                </div>
                <div className="flex gap-2 bg-surface-glass p-1 rounded-xl border border-surface-border">
                    <button className="px-4 py-2 rounded-lg bg-accent-primary text-white shadow-lg shadow-accent-primary/20">概览</button>
                    <button className="px-4 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all">表现</button>
                    <button className="px-4 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all">风险</button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                    { label: '盈亏比', value: '2.45', icon: TrendingUp, color: 'text-accent-success' },
                    { label: '夏普比率', value: '1.82', icon: Activity, color: 'text-accent-primary' },
                    { label: '平均盈利', value: '$450', icon: BarChart2, color: 'text-accent-cyan' },
                    { label: '胜率', value: '62%', icon: PieChart, color: 'text-accent-secondary' },
                ].map((stat, i) => (
                    <Card key={i} className="hover:scale-105 transition-transform duration-300">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-sm font-medium text-slate-400">{stat.label}</p>
                                <h3 className={`text-2xl font-bold mt-2 ${stat.color}`}>{stat.value}</h3>
                            </div>
                            <div className={`p-3 rounded-xl bg-white/5 ${stat.color}`}>
                                <stat.icon size={20} />
                            </div>
                        </div>
                    </Card>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <Card>
                    <h3 className="text-lg font-semibold text-white mb-6">MAE / MFE 分析</h3>
                    <div className="h-64 flex items-center justify-center border border-dashed border-surface-border rounded-xl bg-white/5">
                        <p className="text-slate-500">散点图可视化占位符</p>
                    </div>
                </Card>

                <Card>
                    <h3 className="text-lg font-semibold text-white mb-6">回撤分析</h3>
                    <div className="h-64 flex items-center justify-center border border-dashed border-surface-border rounded-xl bg-white/5">
                        <p className="text-slate-500">回撤图表占位符</p>
                    </div>
                </Card>
            </div>
        </div>
    );
}
