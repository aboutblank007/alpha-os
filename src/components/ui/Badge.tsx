import { cn } from "@/lib/utils";
import React from "react";

type Variant = "default" | "success" | "danger" | "warning" | "secondary";

const styles: Record<Variant, string> = {
  default: "bg-white/10 text-slate-300",
  success: "bg-accent-success/15 text-accent-success",
  danger: "bg-accent-danger/15 text-accent-danger",
  warning: "bg-accent-warning/15 text-accent-warning",
  secondary: "bg-white/5 text-slate-200",
};

export function Badge({ children, variant = "default", className }: { children: React.ReactNode; variant?: Variant; className?: string }) {
  return <span className={cn("inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium", styles[variant], className)}>{children}</span>;
}

