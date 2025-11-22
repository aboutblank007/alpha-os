import React from "react";

export function SkipNav() {
  return (
    <a href="#main" className="sr-only focus:not-sr-only fixed top-2 left-2 z-[100] rounded-lg bg-black/80 px-3 py-2 text-white">
      跳到主要内容
    </a>
  );
}

