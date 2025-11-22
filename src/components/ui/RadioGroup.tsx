"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface RadioOption {
  value: string;
  label: string;
}

interface RadioGroupProps {
  name: string;
  value: string;
  onChange: (value: string) => void;
  options: RadioOption[];
  className?: string;
}

export function RadioGroup({ name, value, onChange, options, className }: RadioGroupProps) {
  return (
    <div role="radiogroup" className={cn("flex flex-wrap gap-2", className)}>
      {options.map(opt => (
        <label key={opt.value} className={cn("inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/10 bg-white/5 text-slate-300 cursor-pointer", value === opt.value && "bg-accent-primary/20 text-white border-accent-primary/30")}> 
          <input
            type="radio"
            name={name}
            value={opt.value}
            checked={value === opt.value}
            onChange={() => onChange(opt.value)}
            className="sr-only"
          />
          <span className={cn("h-4 w-4 rounded-full border", value === opt.value ? "border-accent-primary bg-accent-primary" : "border-white/20 bg-white/5")} />
          <span className="text-sm">{opt.label}</span>
        </label>
      ))}
    </div>
  );
}

