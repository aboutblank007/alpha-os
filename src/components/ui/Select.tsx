"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface Option {
  value: string;
  label: string;
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  description?: string;
  error?: string;
  options?: Option[];
}

export function Select({ id, label, description, error, className, options, children, ...props }: SelectProps) {
  const autoId = React.useId();
  const selectId = id ?? autoId;
  const descId = description ? `${selectId}-desc` : undefined;
  const errId = error ? `${selectId}-err` : undefined;
  const describedBy = [descId, errId].filter(Boolean).join(" ") || undefined;
  return (
    <div className="space-y-2">
      {label && (
        <label htmlFor={selectId} className="text-sm font-medium text-slate-300">
          {label}
        </label>
      )}
      <select
        id={selectId}
        className={cn(
          "w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-white/20",
          error && "border-accent-danger",
          className
        )}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
        {...props}
      >
        {options ? options.map(o => (
          <option key={o.value} value={o.value} className="bg-background">
            {o.label}
          </option>
        )) : children}
      </select>
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
