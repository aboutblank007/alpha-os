import { cn } from "@/lib/utils";
import React from "react";

export type StatusVariant = 
  | "success" 
  | "danger" 
  | "warning" 
  | "info" 
  | "neutral"
  | "profit"
  | "loss";

interface StatusBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: StatusVariant;
  children: React.ReactNode;
  pulse?: boolean;
}

const variants: Record<StatusVariant, string> = {
  success: "bg-accent-success/10 text-accent-success border-accent-success/20",
  danger: "bg-accent-danger/10 text-accent-danger border-accent-danger/20",
  warning: "bg-accent-warning/10 text-accent-warning border-accent-warning/20",
  info: "bg-accent-primary/10 text-accent-primary border-accent-primary/20",
  neutral: "bg-white/5 text-slate-400 border-white/10",
  profit: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  loss: "bg-rose-500/10 text-rose-400 border-rose-500/20",
};

export function StatusBadge({ 
  variant = "neutral", 
  children, 
  className, 
  pulse = false,
  ...props 
}: StatusBadgeProps) {
  return (
    <span 
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border backdrop-blur-sm",
        variants[variant],
        className
      )}
      {...props}
    >
      {pulse && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current"></span>
          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current"></span>
        </span>
      )}
      {children}
    </span>
  );
}

