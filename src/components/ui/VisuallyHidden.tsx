import React from "react";

export function VisuallyHidden({ children }: { children: React.ReactNode }) {
  return (
    <span className="absolute -m-px h-px w-px overflow-hidden p-0 whitespace-nowrap border-0 clip-[rect(0,0,0,0)]">
      {children}
    </span>
  );
}

