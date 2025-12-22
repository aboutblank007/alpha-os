"use client";

import React from "react";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Search, Wifi, Bell, Command } from "lucide-react";

export function CommandHeader() {
    return (
        <header className="h-16 w-full pl-[72px] fixed top-0 left-0 bg-bg-base/80 backdrop-blur-md border-b border-border-subtle z-40 flex items-center justify-between px-6">

            {/* Left: Context / Search */}
            <div className="flex items-center gap-6">
                <div className="hidden md:flex items-center gap-2 text-text-secondary">
                    <span className="text-sm font-medium text-text-primary">工作台</span>
                    <span className="text-border-active">/</span>
                    <span className="text-sm">主终端</span>
                </div>

                <div className="w-64">
                    <Input
                        placeholder="搜索市场或指令..."
                        leftIcon={<Search size={14} />}
                        className="h-8 bg-bg-subtle/50"
                    />
                </div>
            </div>

            {/* Right: System Status */}
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-bg-subtle border border-border-subtle">
                    <div className="w-2 h-2 rounded-full bg-success shadow-[0_0_8px_var(--color-success)] animate-pulse" />
                    <span className="text-xs font-medium text-text-secondary">系统正常</span>
                    <span className="text-[10px] font-mono text-text-muted ml-1">12ms</span>
                </div>

                <Button variant="ghost" size="icon" className="text-text-secondary relative">
                    <Bell size={18} />
                    <span className="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-danger" />
                </Button>

                <Button variant="secondary" size="sm" className="hidden md:flex gap-2">
                    <Command size={14} />
                    <span>操作</span>
                </Button>
            </div>
        </header>
    );
}
