"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  description?: string;
  error?: string;
}

export function Input({ id, label, description, error, className, ...props }: InputProps) {
  const autoId = React.useId();
  const inputId = id ?? autoId;
  const descId = description ? `${inputId}-desc` : undefined;
  const errId = error ? `${inputId}-err` : undefined;
  const describedBy = [descId, errId].filter(Boolean).join(" ") || undefined;
  return (
    <div className="space-y-2">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-slate-300">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={cn(
          "w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-white/20",
          error && "border-accent-danger",
          className
        )}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
        {...props}
      />
      {description && (
        <p id={descId} className="text-xs text-slate-500">
          {description}
        </p>
      )}
      {error && (
        <p id={errId} className="text-xs text-accent-danger">
          {error}
        </p>
      )}
    </div>
  );
}
