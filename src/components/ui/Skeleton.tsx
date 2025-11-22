import { cn } from "@/lib/utils";
import React from "react";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse bg-white/10 rounded", className)} />;
}

