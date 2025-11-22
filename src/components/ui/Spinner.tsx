import React from "react";

export function Spinner({ size = 16 }: { size?: number }) {
  return <span style={{ width: size, height: size }} className="inline-block rounded-full border-2 border-white/40 border-t-white animate-spin" />;
}

