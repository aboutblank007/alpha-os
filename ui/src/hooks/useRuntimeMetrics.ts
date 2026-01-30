import { useEffect, useRef, useState } from 'react';
import { RuntimeSnapshot } from '../shared/ws/types';

type RuntimeMetrics = {
    tickRate: number;
    lastUpdateMs: number | null;
    ageSeconds: number | null;
};

export function useRuntimeMetrics(runtime?: RuntimeSnapshot): RuntimeMetrics {
    const [tickRate, setTickRate] = useState(0);
    const [lastUpdateMs, setLastUpdateMs] = useState<number | null>(null);
    const lastTicksRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number | null>(null);

    useEffect(() => {
        if (!runtime) return;

        const ticks = runtime.ticks_total ?? 0;
        const tsSec = runtime.timestamp || 0;
        if (!tsSec) return;

        const tsMs = tsSec * 1000;
        if (lastTicksRef.current !== null && lastTimeRef.current !== null) {
            const dtSec = Math.max((tsMs - lastTimeRef.current) / 1000, 0.001);
            const rate = (ticks - lastTicksRef.current) / dtSec;
            if (Number.isFinite(rate)) {
                setTickRate(Math.max(0, rate));
            }
        }

        lastTicksRef.current = ticks;
        lastTimeRef.current = tsMs;
        setLastUpdateMs(tsMs);
    }, [runtime?.timestamp, runtime?.ticks_total]);

    const ageSeconds = lastUpdateMs ? Math.max(0, (Date.now() - lastUpdateMs) / 1000) : null;

    return { tickRate, lastUpdateMs, ageSeconds };
}

export default useRuntimeMetrics;
