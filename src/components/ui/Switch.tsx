"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  className?: string;
}

export function Switch({ checked, onChange, label, className }: SwitchProps) {
  const id = React.useId();
  return (
    <div className={cn("inline-flex items-center gap-3", className)}>
      <button
        id={id}
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onChange(!checked);
          }
        }}
        className={cn("h-6 w-11 rounded-full transition-all", checked ? "bg-accent-primary" : "bg-white/10")}
      >
        <span className={cn("h-5 w-5 rounded-full bg-white shadow transform transition-all", checked ? "translate-x-6" : "translate-x-1")} />
      </button>
      {label && (
        <label htmlFor={id} className="text-sm text-slate-300">
          {label}
        </label>
      )}
    </div>
  );
}

