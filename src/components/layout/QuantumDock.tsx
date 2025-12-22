"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
    LayoutGrid,
    LineChart,
    BookOpen,
    Settings,
    Terminal,
    Activity,
    Brain
} from "lucide-react";

const NAV_ITEMS = [
    { icon: LayoutGrid, label: "仪表盘", href: "/dashboard" },
    { icon: Activity, label: "市场", href: "/market" },
    { icon: LineChart, label: "分析", href: "/analytics" },
    { icon: Brain, label: "AI", href: "/ai" },
    { icon: BookOpen, label: "日志", href: "/journal" },
    { icon: Settings, label: "设置", href: "/settings" },
];

export function QuantumDock() {
    const pathname = usePathname();

    return (
        <aside className="fixed left-0 top-0 h-full w-[72px] flex flex-col items-center bg-bg-base border-r border-border-subtle z-50">
            {/* Brand */}
            <div className="h-16 w-full flex items-center justify-center border-b border-border-subtle">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-indigo-600 flex items-center justify-center shadow-lg shadow-primary/20">
                    <Terminal className="text-white" size={20} />
                </div>
            </div>

            {/* Nav */}
            <nav className="flex-1 flex flex-col gap-4 py-6 w-full px-3">
                {NAV_ITEMS.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className="relative group w-full flex justify-center"
                        >
                            {isActive && (
                                <div className="absolute left-[-12px] top-1/2 -translate-y-1/2 w-1 h-8 bg-primary rounded-r-full shadow-[0_0_10px_var(--color-primary)]" />
                            )}

                            <div
                                className={cn(
                                    "w-12 h-12 flex items-center justify-center rounded-xl transition-all duration-300",
                                    isActive
                                        ? "bg-primary/10 text-primary shadow-[0_0_15px_rgba(37,99,235,0.15)]"
                                        : "text-text-secondary hover:text-white hover:bg-white/5"
                                )}
                            >
                                <item.icon size={22} strokeWidth={isActive ? 2.5 : 2} />
                            </div>

                            {/* Tooltip (CSS only for speed) */}
                            <div className="absolute left-14 top-1/2 -translate-y-1/2 px-2 py-1 bg-bg-card border border-border-subtle rounded text-xs text-white opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-[60]">
                                {item.label}
                            </div>
                        </Link>
                    );
                })}
            </nav>

            {/* User / Bot Status */}
            <div className="mb-6 w-full px-3 flex flex-col gap-3">
                <div className="w-12 h-12 rounded-full bg-surface-highlight border border-border-subtle flex items-center justify-center relative cursor-pointer hover:border-text-muted transition-colors">
                    <span className="font-bold text-xs text-text-secondary">AI</span>
                    <span className="absolute bottom-0 right-0 w-3 h-3 bg-success rounded-full border-2 border-bg-base" />
                </div>
            </div>
        </aside>
    );
}
