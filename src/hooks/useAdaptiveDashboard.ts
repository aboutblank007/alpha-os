"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";
import { useMarketStore } from "@/store/useMarketStore";
import { useTradeStore } from "@/store/useTradeStore";
import { useSignalStore } from "@/store/useSignalStore";
import { useSettingsStore } from "@/store/useSettingsStore";
import {
  computeDeviceProfile,
  computePriority,
  computeRiskState,
  computeSignalState,
  type TerminalMode,
} from "@/lib/terminal-adaptive";

const MODE_STORAGE_KEY = "alphaos_terminal_mode_v1";
const LOCAL_EVENT = "alphaos:localstorage";

// React useSyncExternalStore 要求 getSnapshot 在一次 render 内稳定。
// 直接读取 window.innerWidth 可能因为滚动条/布局抖动而在 render 过程中变化，触发 React #185。
let widthSnapshot = typeof window !== "undefined" ? window.innerWidth : 1440;

function safeJsonParse<T>(raw: string | null): T | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function emitLocalStorageKey(key: string) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(LOCAL_EVENT, { detail: { key } }));
}

function subscribeToKey(key: string, callback: () => void) {
  if (typeof window === "undefined") return () => {};
  const onStorage = (e: StorageEvent) => {
    if (e.key === key) callback();
  };
  const onLocal = (e: Event) => {
    const ce = e as CustomEvent<{ key?: string }>;
    if (ce?.detail?.key === key) callback();
  };
  window.addEventListener("storage", onStorage);
  window.addEventListener(LOCAL_EVENT, onLocal);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(LOCAL_EVENT, onLocal);
  };
}

function getLocalSnapshot(key: string) {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(key);
}

function subscribeToResize(callback: () => void) {
  if (typeof window === "undefined") return () => {};
  const onResize = () => {
    widthSnapshot = window.innerWidth;
    callback();
  };
  // 初始化一次，确保 snapshot 与当前一致
  widthSnapshot = window.innerWidth;
  window.addEventListener("resize", onResize);
  return () => window.removeEventListener("resize", onResize);
}

function getWidthSnapshot() {
  return widthSnapshot;
}

export function useAdaptiveDashboard(params: {
  workspace: string;
  baseLayout: string[];
  stats: {
    totalPnL?: number;
    maxDrawdown?: number;
    winRate?: number;
    profitFactor?: number;
    executionEfficiency?: number;
  };
}) {
  const { workspace, baseLayout, stats } = params;

  // 关键：避免 selector 返回新对象（SSR/hydration 时会触发 getServerSnapshot 未缓存警告，甚至导致无限更新）
  const riskLevel = useSettingsStore((s) => s.riskLevel);
  const userRole = useSettingsStore((s) => s.userRole);
  const density = useSettingsStore((s) => s.density);
  const defaultTerminalMode = useSettingsStore((s) => s.defaultTerminalMode);

  const isConnected = useMarketStore((s) => s.isConnected);
  const latency = useMarketStore((s) => s.latency);

  const account = useTradeStore((s) => s.account);
  const positions = useTradeStore((s) => s.positions);
  const signals = useSignalStore((s) => s.signals);

  // Device profile
  const width = useSyncExternalStore(subscribeToResize, getWidthSnapshot, () => 1440);
  const device = useMemo(() => computeDeviceProfile(width), [width]);

  // Terminal mode (local)
  const rawMode = useSyncExternalStore(
    (cb) => subscribeToKey(MODE_STORAGE_KEY, cb),
    () => getLocalSnapshot(MODE_STORAGE_KEY),
    () => null
  );
  const modeFromSettings: TerminalMode = defaultTerminalMode === "monitoring" ? "monitoring" : "trading";
  const mode = (safeJsonParse<TerminalMode>(rawMode) ?? modeFromSettings) as TerminalMode;
  const setMode = useCallback((next: TerminalMode) => {
    if (typeof window === "undefined") return;
    localStorage.setItem(MODE_STORAGE_KEY, JSON.stringify(next));
    emitLocalStorageKey(MODE_STORAGE_KEY);
  }, []);

  // Pinned modules (per workspace)
  const pinsKey = `alphaos_pins_v1_${workspace}`;
  const rawPins = useSyncExternalStore(
    (cb) => subscribeToKey(pinsKey, cb),
    () => getLocalSnapshot(pinsKey),
    () => null
  );
  const pinned = useMemo(() => {
    const list = safeJsonParse<string[]>(rawPins) ?? [];
    return new Set(list);
  }, [rawPins]);

  const togglePin = useCallback((key: string) => {
    if (typeof window === "undefined") return;
    const current = safeJsonParse<string[]>(localStorage.getItem(pinsKey)) ?? [];
    const next = new Set(current);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    localStorage.setItem(pinsKey, JSON.stringify(Array.from(next)));
    emitLocalStorageKey(pinsKey);
  }, [pinsKey]);

  const risk = useMemo(
    () =>
      computeRiskState({
        account,
        positions,
        stats,
        riskLevel,
        isConnected,
      }),
    [account, positions, stats, riskLevel, isConnected]
  );

  const signal = useMemo(() => computeSignalState(signals), [signals]);

  const ctx = useMemo(
    () => ({
      device,
      role: userRole,
      density,
      mode,
      risk,
      signal,
      isConnected,
      latencyMs: latency,
    }),
    [device, userRole, density, mode, risk, signal, isConnected, latency]
  );

  const priority = useMemo(
    () =>
      computePriority({
        baseLayout: baseLayout,
        pinned,
        ctx,
      }),
    [baseLayout, pinned, ctx]
  );

  return {
    device,
    mode,
    setMode,
    pinned,
    togglePin,
    risk,
    signal,
    ctx,
    priority,
  };
}


