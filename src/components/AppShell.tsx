"use client";

import React from "react";
import { QuantumDock } from "@/components/layout/QuantumDock";
import { CommandHeader } from "@/components/layout/CommandHeader";
import { useBridgeSync } from "@/hooks/useBridgeSync";

export function AppShell({ children }: { children: React.ReactNode }) {
    // 启动后台数据同步（每秒轮询 Bridge 状态）
    useBridgeSync(1000);

    return (
        <div className="min-h-screen bg-bg-base text-text-primary selection:bg-primary/30">
            <QuantumDock />
            <CommandHeader />

            {/* Main Content Area */}
            <main className="pl-[72px] pt-16 h-screen overflow-hidden">
                <div className="p-4 lg:p-6 max-w-[1920px] mx-auto h-full animate-in fade-in duration-500">
                    {children}
                </div>
            </main>
        </div>
    );
}
