"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export function Checkbox({ id, checked, onChange, label, className, ...props }: CheckboxProps) {
  const autoId = React.useId();
  const inputId = id ?? autoId;
  return (
    <div className={cn("inline-flex items-center gap-3", className)}>
      <div className="relative">
        <input
          id={inputId}
          type="checkbox"
          className="peer sr-only"
          checked={checked}
          onChange={onChange}
          {...props}
        />
        <label
          htmlFor={inputId}
          className="block h-5 w-5 rounded-md border border-slate-300 bg-transparent peer-checked:bg-blue-500 peer-checked:border-blue-500 cursor-pointer transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-blue-400" 
        >
          <span className={cn("absolute inset-0 grid place-items-center text-white", checked ? "opacity-100" : "opacity-0")}>✔</span>
        </label>
      </div>
      {label && (
        <label htmlFor={inputId} className="text-sm text-slate-300">
          {label}
        </label>
      )}
    </div>
  );
}
