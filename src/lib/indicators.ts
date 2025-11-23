export interface OHLC {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
}

export interface IndicatorSignal {
    time: number;
    type: 'BUY' | 'SELL' | 'RECLAIM_BUY' | 'RECLAIM_SELL';
    price: number;
    label: string;
}

export interface TrendState {
    trendUp: boolean;
    htfTrendUp: boolean; // Simulated or real
    tpPrice: number;
    slPrice: number;
}

export interface IndicatorResult {
    ema1: { time: number; value: number }[];
    ema2: { time: number; value: number }[];
    centerLine: { time: number; value: number; color: string }[];
    signals: IndicatorSignal[];
    currentTrend: TrendState;
}

export interface IndicatorSettings {
    prd: number; // Pivot Period
    Pd: number; // ATR Period
    length1: number; // EMA Short
    length2: number; // EMA Long
    filterMode: 'NO_FILTER' | 'BASIC' | 'STRICT';
    minDistance: number; // ATR Multiplier
    trendBufferMult: number;
    useSlopeFilter: boolean;
    slopeThreshold: number;
    strictSlope: boolean;
    minEmaSpreadMult: number;
    minBarsBetweenSignals: number;
    useHtfFilter: boolean;
    useReclaim: boolean;
    reclaimStrictMode: boolean;
    tpAtrMult: number;
    slAtrMult: number;
    tpSlBase: 'CLOSE' | 'CENTER';
}

export const DEFAULT_SETTINGS: IndicatorSettings = {
    prd: 2,
    Pd: 14,
    length1: 6,
    length2: 24,
    filterMode: 'BASIC',
    minDistance: 0.4,
    trendBufferMult: 0.1,
    useSlopeFilter: true,
    slopeThreshold: 0.5,
    strictSlope: true,
    minEmaSpreadMult: 0.1,
    minBarsBetweenSignals: 3,
    useHtfFilter: true,
    useReclaim: true,
    reclaimStrictMode: true,
    tpAtrMult: 1.5,
    slAtrMult: 1.0,
    tpSlBase: 'CLOSE',
};

// --- Helper Functions ---

function calculateEMA(data: number[], period: number): number[] {
    const k = 2 / (period + 1);
    const emaArray: number[] = [];
    let ema = data[0];
    emaArray.push(ema);

    for (let i = 1; i < data.length; i++) {
        ema = data[i] * k + ema * (1 - k);
        emaArray.push(ema);
    }
    return emaArray;
}

function calculateATR(high: number[], low: number[], close: number[], period: number): number[] {
    const tr: number[] = [high[0] - low[0]];
    for (let i = 1; i < high.length; i++) {
        const hl = high[i] - low[i];
        const hc = Math.abs(high[i] - close[i - 1]);
        const lc = Math.abs(low[i] - close[i - 1]);
        tr.push(Math.max(hl, hc, lc));
    }

    // First ATR is simple average of TR
    let atr = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
    const atrArray: number[] = new Array(period - 1).fill(NaN); // Pad initial NaNs
    atrArray.push(atr);

    // Subsequent ATRs: (Previous ATR * (n-1) + Current TR) / n
    for (let i = period; i < tr.length; i++) {
        atr = (atr * (period - 1) + tr[i]) / period;
        atrArray.push(atr);
    }

    // Fill beginning with NaNs to match data length
    const result = new Array(high.length).fill(NaN);
    for (let i = 0; i < atrArray.length; i++) {
        // The atrArray calculated above aligns such that atrArray[period-1] corresponds to data[period-1]
        // But we pushed NaNs to atrArray to align indices? No, let's simplify.
        // Let's just map back.
        // The loop above starts pushing at index `period-1`.
        // So result[i] should be valid from i = period-1.
        // Actually, standard RMA/ATR usually needs more data to stabilize, but this is the formula.
    }

    // Re-implementation for simpler array mapping
    const finalAtr = new Array(high.length).fill(NaN);
    let sumTr = 0;
    for (let i = 0; i < period; i++) {
        sumTr += tr[i];
    }
    let currentAtr = sumTr / period;
    finalAtr[period - 1] = currentAtr;

    for (let i = period; i < tr.length; i++) {
        currentAtr = (currentAtr * (period - 1) + tr[i]) / period;
        finalAtr[i] = currentAtr;
    }

    return finalAtr;
}

function calculateDMI(high: number[], low: number[], close: number[], period: number): { plusDI: number[], minusDI: number[], adx: number[] } {
    const plusDM: number[] = [0];
    const minusDM: number[] = [0];
    const tr: number[] = [high[0] - low[0]];

    for (let i = 1; i < high.length; i++) {
        const up = high[i] - high[i - 1];
        const down = low[i - 1] - low[i];

        plusDM.push((up > down && up > 0) ? up : 0);
        minusDM.push((down > up && down > 0) ? down : 0);

        const hl = high[i] - low[i];
        const hc = Math.abs(high[i] - close[i - 1]);
        const lc = Math.abs(low[i] - close[i - 1]);
        tr.push(Math.max(hl, hc, lc));
    }

    // Smooth TR, +DM, -DM using RMA (same as ATR smoothing)
    // RMA(x, n) = (prev * (n-1) + curr) / n
    const smooth = (data: number[], n: number) => {
        const res = new Array(data.length).fill(0);
        let sum = 0;
        for (let i = 0; i < n; i++) sum += data[i];
        res[n - 1] = sum / n;
        for (let i = n; i < data.length; i++) {
            res[i] = (res[i - 1] * (n - 1) + data[i]) / n;
        }
        return res;
    };

    const trSmooth = smooth(tr, period);
    const plusDMSmooth = smooth(plusDM, period);
    const minusDMSmooth = smooth(minusDM, period);

    const plusDI = new Array(high.length).fill(0);
    const minusDI = new Array(high.length).fill(0);
    const dx = new Array(high.length).fill(0);

    for (let i = period - 1; i < high.length; i++) {
        if (trSmooth[i] !== 0) {
            plusDI[i] = (plusDMSmooth[i] / trSmooth[i]) * 100;
            minusDI[i] = (minusDMSmooth[i] / trSmooth[i]) * 100;
        }
        const sumDI = plusDI[i] + minusDI[i];
        if (sumDI !== 0) {
            dx[i] = (Math.abs(plusDI[i] - minusDI[i]) / sumDI) * 100;
        }
    }

    const adx = smooth(dx, period); // ADX is smoothed DX

    return { plusDI, minusDI, adx };
}

function pivotHigh(high: number[], left: number, right: number): (number | typeof NaN)[] {
    const result = new Array(high.length).fill(NaN);
    for (let i = left; i < high.length - right; i++) {
        let isPivot = true;
        // Check left
        for (let j = 1; j <= left; j++) {
            if (high[i - j] > high[i]) { isPivot = false; break; }
        }
        // Check right
        if (isPivot) {
            for (let j = 1; j <= right; j++) {
                if (high[i + j] >= high[i]) { isPivot = false; break; } // strict inequality for right usually? Pine uses > or >= depending on version, usually >= for right to avoid duplicates? 
                // Pine Script `pivothigh` logic: returns the value at the pivot point, but it's only known `right` bars later.
                // However, the function returns a series where the value is present at the bar it occurred? 
                // No, `ta.pivothigh` returns a value at the moment it is confirmed (i.e., `right` bars later), but the value is of the pivot bar.
                // Wait, `ta.pivothigh(source, left, right)` returns the high of the pivot at the offset `right`.
                // So at `bar_index`, if a pivot happened at `bar_index - right`, it returns that high. Otherwise NaN.
            }
        }

        if (isPivot) {
            // The pivot is at index `i`. It is confirmed at index `i + right`.
            // So at index `i + right`, we should record `high[i]`.
            if (i + right < result.length) {
                result[i + right] = high[i];
            }
        }
    }
    return result;
}

function pivotLow(low: number[], left: number, right: number): (number | typeof NaN)[] {
    const result = new Array(low.length).fill(NaN);
    for (let i = left; i < low.length - right; i++) {
        let isPivot = true;
        for (let j = 1; j <= left; j++) {
            if (low[i - j] < low[i]) { isPivot = false; break; }
        }
        if (isPivot) {
            for (let j = 1; j <= right; j++) {
                if (low[i + j] <= low[i]) { isPivot = false; break; }
            }
        }

        if (isPivot) {
            if (i + right < result.length) {
                result[i + right] = low[i];
            }
        }
    }
    return result;
}

// --- Main Indicator Logic ---

export function calculatePivotTrendSignals(data: OHLC[], settings: IndicatorSettings = DEFAULT_SETTINGS): IndicatorResult {
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    // 1. Pivot Point & Center Line
    // Pine: float ph = ta.pivothigh(prd, prd)
    // Pine: float pl = ta.pivotlow(prd, prd)
    const ph = pivotHigh(highs, settings.prd, settings.prd);
    const pl = pivotLow(lows, settings.prd, settings.prd);

    const centerLine: number[] = new Array(data.length).fill(NaN);
    let lastpp = NaN;
    let center = NaN;

    for (let i = 0; i < data.length; i++) {
        const currPh = ph[i];
        const currPl = pl[i];

        if (!Number.isNaN(currPh)) lastpp = currPh;
        else if (!Number.isNaN(currPl)) lastpp = currPl;

        if (!Number.isNaN(lastpp)) {
            if (Number.isNaN(center)) {
                center = lastpp;
            } else {
                center = (center * 2 + lastpp) / 3;
            }
        }
        centerLine[i] = center;
    }

    // 2. EMAs & ATR
    const ema1 = calculateEMA(closes, settings.length1);
    const ema2 = calculateEMA(closes, settings.length2);
    const atrVal = calculateATR(highs, lows, closes, settings.Pd);
    const { adx } = calculateDMI(highs, lows, closes, 14);

    // 3. Logic Loop
    const signals: IndicatorSignal[] = [];
    let lastBuyBar = -100;
    let lastSellBar = -100;

    // Reclaim state
    let waitingForReclaimBuy = false;
    let waitingForReclaimSell = false;
    let reclaimRefPrice = NaN;

    // MinTick approximation (assuming forex/crypto standard, or derive from data)
    const minTick = 0.00001; // TODO: Make dynamic or pass in

    for (let i = Math.max(settings.length2, settings.Pd, 14); i < data.length; i++) {
        const close = closes[i];
        const high = highs[i];
        const low = lows[i];
        const e1 = ema1[i];
        const e2 = ema2[i];
        const e1_prev = ema1[i - 1];
        const e2_prev = ema2[i - 1];
        const atr = atrVal[i];
        const center = centerLine[i];
        const currentAdx = adx[i];

        // Trend Status
        const trendUp = e1 > e2;
        const crossUp = e1 > e2 && e1_prev <= e2_prev;
        const crossDn = e1 < e2 && e1_prev >= e2_prev;

        // 4. Filters

        // 4.1 Distance Filter
        let atrDistance = NaN;
        if (!Number.isNaN(center) && !Number.isNaN(atr) && atr !== 0) {
            atrDistance = Math.abs(close - center) / atr;
        }

        // Slope Filter
        const ema1Slope = Math.abs(e1 - e1_prev) / minTick;
        const ema1SlopePrev = Math.abs(e1_prev - ema1[i - 2]) / minTick;
        const slopeValOk = ema1Slope > settings.slopeThreshold;
        const slopePersistent = settings.strictSlope ? (slopeValOk && (ema1SlopePrev > settings.slopeThreshold)) : slopeValOk;
        const slopeOk = settings.useSlopeFilter ? slopePersistent : false;

        const distanceOk = (!Number.isNaN(atrDistance) && atrDistance >= settings.minDistance) || slopeOk;

        // 4.2 Trend Filter (Hysteresis)
        const useTrendFilter = settings.filterMode !== 'NO_FILTER';
        const trendBuffer = !Number.isNaN(atr) ? atr * settings.trendBufferMult : 0.0;
        const priceAboveCenter = !Number.isNaN(center) ? close > (center + trendBuffer) : false;
        const priceBelowCenter = !Number.isNaN(center) ? close < (center - trendBuffer) : false;

        let trendFilterOk = true;
        if (useTrendFilter) {
            if (!Number.isNaN(center)) {
                trendFilterOk = (trendUp && priceAboveCenter) || (!trendUp && priceBelowCenter);
            } else {
                trendFilterOk = trendUp;
            }
        }

        // 4.3 HTF Filter (Simplified: just use current trend for now, or same logic)
        // In Pine: request.security(..., htf_period, ...)
        // Here: We skip or assume true for V1
        const htfTrendOk = true;

        // 4.4 Volatility & Chop (Strict Mode)
        let volatilityOk = true;
        let notChop = true;
        if (settings.filterMode === 'STRICT') {
            const atrPercent = (!Number.isNaN(atr) && close !== 0) ? (atr / close) * 100 : 0;
            volatilityOk = atrPercent >= 0.3 && atrPercent <= 2.0;

            // Highest/Lowest of length2 (24)
            let highest = -Infinity;
            let lowest = Infinity;
            for (let j = 0; j < settings.length2; j++) {
                if (i - j >= 0) {
                    highest = Math.max(highest, highs[i - j]);
                    lowest = Math.min(lowest, lows[i - j]);
                }
            }
            const priceRangePercent = (close !== 0) ? ((highest - lowest) / close) * 100 : 0;
            notChop = priceRangePercent >= 0.5;
        }

        // EMA Spread Filter
        const spreadThreshold = !Number.isNaN(atr) ? atr * settings.minEmaSpreadMult : 0.0;
        const spreadOk = Math.abs(e1 - e2) > spreadThreshold;

        // 5. Signals
        const timeOkBuy = (i - lastBuyBar >= settings.minBarsBetweenSignals);
        const timeOkSell = (i - lastSellBar >= settings.minBarsBetweenSignals);

        const validBuy = crossUp && trendFilterOk && htfTrendOk && distanceOk && volatilityOk && notChop && timeOkBuy && spreadOk;
        const validSell = crossDn && trendFilterOk && htfTrendOk && distanceOk && volatilityOk && notChop && timeOkSell && spreadOk;

        // Update Reclaim State
        if (validBuy) {
            waitingForReclaimBuy = false;
            reclaimRefPrice = NaN;
        } else if (crossUp) {
            waitingForReclaimBuy = true;
            waitingForReclaimSell = false;
            reclaimRefPrice = high;
        }

        if (validSell) {
            waitingForReclaimSell = false;
            reclaimRefPrice = NaN;
        } else if (crossDn) {
            waitingForReclaimSell = true;
            waitingForReclaimBuy = false;
            reclaimRefPrice = low;
        }

        // Reclaim Signals
        let reclaimBuySig = false;
        let reclaimSellSig = false;

        if (settings.useReclaim && trendUp && !validBuy) {
            const adxOk = !settings.reclaimStrictMode || (currentAdx > 20);
            const breakout = !Number.isNaN(reclaimRefPrice) && close > reclaimRefPrice;

            if (!Number.isNaN(reclaimRefPrice)) {
                reclaimRefPrice = Math.max(reclaimRefPrice, high);
            }

            const recTimeOk = (i - lastBuyBar >= settings.minBarsBetweenSignals * 2);

            if (waitingForReclaimBuy && breakout && adxOk && recTimeOk && htfTrendOk && spreadOk) {
                reclaimBuySig = true;
                reclaimRefPrice = high;
            }
        }

        if (settings.useReclaim && !trendUp && !validSell) {
            const adxOk = !settings.reclaimStrictMode || (currentAdx > 20);
            const breakout = !Number.isNaN(reclaimRefPrice) && close < reclaimRefPrice;

            if (!Number.isNaN(reclaimRefPrice)) {
                reclaimRefPrice = Math.min(reclaimRefPrice, low);
            }

            const recTimeOk = (i - lastSellBar >= settings.minBarsBetweenSignals * 2);

            if (waitingForReclaimSell && breakout && adxOk && recTimeOk && htfTrendOk && spreadOk) {
                reclaimSellSig = true;
                reclaimRefPrice = low;
            }
        }

        // Record Signals
        if (validBuy) {
            signals.push({ time: data[i].time, type: 'BUY', price: low, label: '买入' });
            lastBuyBar = i;
        } else if (reclaimBuySig) {
            signals.push({ time: data[i].time, type: 'RECLAIM_BUY', price: low, label: 'Reclaim' });
            lastBuyBar = i;
        }

        if (validSell) {
            signals.push({ time: data[i].time, type: 'SELL', price: high, label: '卖出' });
            lastSellBar = i;
        } else if (reclaimSellSig) {
            signals.push({ time: data[i].time, type: 'RECLAIM_SELL', price: high, label: 'Reclaim' });
            lastSellBar = i;
        }
    }

    // Final Trend State for Table
    const lastIdx = data.length - 1;
    const currTrendUp = ema1[lastIdx] > ema2[lastIdx];
    const currClose = closes[lastIdx];
    const currAtr = atrVal[lastIdx];
    const currCenter = centerLine[lastIdx];

    const basePrice = settings.tpSlBase === 'CLOSE' ? currClose : currCenter;
    const tpPrice = currTrendUp ? basePrice + settings.tpAtrMult * currAtr : basePrice - settings.tpAtrMult * currAtr;
    const slPrice = currTrendUp ? basePrice - settings.slAtrMult * currAtr : basePrice + settings.slAtrMult * currAtr;

    return {
        ema1: ema1.map((v, i) => ({ time: data[i].time, value: v })).filter(d => !Number.isNaN(d.value)),
        ema2: ema2.map((v, i) => ({ time: data[i].time, value: v })).filter(d => !Number.isNaN(d.value)),
        centerLine: centerLine.map((v, i) => {
            const hl2 = (highs[i] + lows[i]) / 2;
            const color = v < hl2 ? '#26ba9f' : '#ba3026';
            return { time: data[i].time, value: v, color };
        }).filter(d => !Number.isNaN(d.value)),
        signals,
        currentTrend: {
            trendUp: currTrendUp,
            htfTrendUp: currTrendUp, // Mocked
            tpPrice,
            slPrice
        }
    };
}
