"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface TabItem {
  value: string;
  label: string;
  content: React.ReactNode;
}

interface TabsProps {
  items: TabItem[];
  defaultValue?: string;
  className?: string;
}

export function Tabs({ items, defaultValue, className }: TabsProps) {
  const [active, setActive] = React.useState<string>(defaultValue ?? items[0]?.value ?? "");
  const listId = React.useId();
  return (
    <div className={cn("w-full", className)}>
      <div role="tablist" aria-orientation="horizontal" className="flex gap-2 bg-surface-glass p-1 rounded-xl border border-surface-border">
        {items.map(item => {
          const selected = active === item.value;
          return (
            <button
              key={item.value}
              role="tab"
              aria-selected={selected}
              aria-controls={`${listId}-${item.value}`}
              className={cn("px-4 py-2 rounded-lg text-sm", selected ? "bg-accent-primary text-white shadow-lg shadow-accent-primary/20" : "text-slate-400 hover:text-white hover:bg-white/5")}
              onClick={() => setActive(item.value)}
            >
              {item.label}
            </button>
          );
        })}
      </div>
      <div id={`${listId}-${active}`} role="tabpanel" className="mt-4">
        {items.find(i => i.value === active)?.content}
      </div>
    </div>
  );
}

