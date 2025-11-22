import { cn } from "@/lib/utils";
import React from "react";

export function Divider({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-white/10", className)} />;
}

