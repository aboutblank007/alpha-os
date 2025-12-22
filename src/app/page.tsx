"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Terminal } from "lucide-react";

export default function Home() {
  const router = useRouter();
  const [status, setStatus] = useState("Initializing Secure Environment...");

  useEffect(() => {
    const timer1 = setTimeout(() => setStatus("Authenticating User Session..."), 800);
    const timer2 = setTimeout(() => setStatus("Connecting to Market Data Bridge..."), 1600);
    const timer3 = setTimeout(() => {
      setStatus("Ready.");
      router.push("/dashboard");
    }, 2400);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, [router]);

  return (
    <main className="h-screen w-full flex flex-col items-center justify-center bg-background relative overflow-hidden">
      {/* Subtle Background Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] pointer-events-none" />

      <div className="z-10 flex flex-col items-center">
        <div className="w-16 h-16 bg-bg-card border border-white/5 rounded-xl flex items-center justify-center mb-8 shadow-2xl backdrop-blur-md">
          <Terminal className="text-primary" size={32} />
        </div>

        <h1 className="text-2xl font-semibold tracking-tight text-white mb-2">
          Alpha<span className="text-primary">OS</span> <span className="text-text-muted font-normal">Quantum</span>
        </h1>

        <div className="flex items-center gap-3 mt-8">
          <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          <p className="text-sm text-text-secondary font-mono">{status}</p>
        </div>
      </div>

      <div className="absolute bottom-8 text-xs text-text-muted font-mono">
        v3.0.0-QUANTUM | Secured Connection
      </div>
    </main>
  );
}
