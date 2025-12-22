"use client";

import React from "react";
import { GlassCard, CardHeader, CardTitle } from "@/components/ui/GlassCard";
import { MarketWatch } from "@/components/dashboard/MarketWatch";
import { TrendingUp, Globe, Clock } from "lucide-react";

export default function MarketPage() {
    return (
        <div className="flex flex-col gap-6 h-[calc(100vh-100px)]">
            <div className="flex items-center justify-between shrink-0">
                <h1 className="text-2xl font-bold tracking-tight">全球市场</h1>
                <div className="flex gap-4 text-xs text-text-secondary font-mono">
                    <span className="flex items-center gap-1"><Globe size={12} /> 伦敦: 开市</span>
                    <span className="flex items-center gap-1"><Clock size={12} /> 纽约: 开市</span>
                    <span className="flex items-center gap-1 text-text-muted"><Clock size={12} /> 东京: 休市</span>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-6 min-h-0">
                {/* Main Ticker */}
                <div className="col-span-1 md:col-span-2 h-full flex flex-col gap-6">
                    <GlassCard className="flex-1">
                        <MarketWatch />
                    </GlassCard>
                </div>

                {/* Market News / Sentiment */}
                <div className="col-span-1 flex flex-col gap-6">
                    <GlassCard className="h-1/3 p-6 flex flex-col justify-center gap-2 relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-8 opacity-10">
                            <TrendingUp size={64} className="text-success" />
                        </div>
                        <h3 className="text-text-muted text-xs uppercase font-bold">市场情绪</h3>
                        <div className="text-3xl font-bold text-success">看涨</div>
                        <p className="text-xs text-text-secondary">AI 置信度: 87%</p>
                    </GlassCard>

                    <GlassCard className="flex-1 p-6">
                        <CardHeader className="px-0 pt-0">
                            <CardTitle>突发新闻</CardTitle>
                        </CardHeader>
                        <div className="space-y-4 mt-4">
                            {[
                                { time: "10:30", text: "美国 CPI 数据低于预期。" },
                                { time: "11:15", text: "EURUSD 突破关键阻力位 1.0900。" },
                                { time: "12:00", text: "比特币因 ETF 交易量激增突破 6.8 万美元。" },
                            ].map((news, i) => (
                                <div key={i} className="flex gap-3 text-sm border-b border-white/5 pb-2 last:border-0">
                                    <span className="text-text-muted font-mono text-xs">{news.time}</span>
                                    <span className="text-text-primary">{news.text}</span>
                                </div>
                            ))}
                        </div>
                    </GlassCard>
                </div>
            </div>
        </div>
    );
}
