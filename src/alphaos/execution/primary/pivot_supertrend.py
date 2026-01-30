"""
Pivot Point SuperTrend (v4.1)

基于 TradingView 原版 "Pivot Point SuperTrend" by LonesomeTheBlue 实现
https://www.tradingview.com/script/n8bGFq8r-Pivot-Point-SuperTrend/

核心逻辑：
1. 检测 Pivot High/Low（结构性高低点）
2. 使用 Pivot 点的加权平均作为中心线：center = (center * 2 + lastpp) / 3
3. SuperTrend 带基于中心线计算：Up = center - ATR*factor, Dn = center + ATR*factor
4. 趋势切换：close > TDown[1] 为多头，close < TUp[1] 为空头

与标准 SuperTrend 的区别：
- 使用 Pivot 中心线而非 HL2 作为基准
- 中心线平滑跟踪结构性高低点，更稳定
- 避免单个 bar 波动导致的虚假信号

默认参数（来自 TradingView 原版）：
- prd = 2 (Pivot Point Period)
- Factor = 3 (ATR Factor)
- Pd = 10 (ATR Period)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Sequence

from alphaos.core.logging import get_logger
from alphaos.v4.types import Bar

logger = get_logger(__name__)


class TrendDirection(Enum):
    """Trend direction states."""
    LONG = "long"        # Uptrend confirmed
    SHORT = "short"      # Downtrend confirmed
    NONE = "none"        # No clear trend / initializing


@dataclass
class SuperTrendState:
    """
    Complete state output from PivotSuperTrend.
    
    Attributes:
        direction: Current trend direction
        supertrend_line: Current SuperTrend level (support in uptrend, resistance in downtrend)
        bandwidth: ATR-based bandwidth (volatility measure)
        trend_duration: Number of bars since last trend change
        pivot_high: Most recent confirmed pivot high
        pivot_low: Most recent confirmed pivot low
        center_line: Current pivot-based center line
        atr: Current ATR value
    """
    direction: TrendDirection = TrendDirection.NONE
    supertrend_line: float = 0.0
    bandwidth: float = 0.0
    trend_duration: int = 0
    pivot_high: float = 0.0
    pivot_low: float = 0.0
    center_line: float = 0.0
    atr: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "direction": self.direction.name,
            "supertrend_line": round(self.supertrend_line, 2),
            "bandwidth": round(self.bandwidth, 4),
            "trend_duration": self.trend_duration,
            "pivot_high": round(self.pivot_high, 2),
            "pivot_low": round(self.pivot_low, 2),
            "center_line": round(self.center_line, 2),
            "atr": round(self.atr, 4),
        }


@dataclass
class PivotSuperTrend:
    """
    Pivot Point SuperTrend - 基于 TradingView 原版实现
    
    核心改进：使用 Pivot 点的加权平均作为中心线，而非当前 bar 的 HL2。
    这使得 SuperTrend 线更稳定，减少虚假信号。
    
    Pine Script 原版逻辑：
    ```
    // Center line from pivot points
    center := (center * 2 + lastpp) / 3
    
    // Bands from center line
    Up = center - (Factor * atr(Pd))
    Dn = center + (Factor * atr(Pd))
    
    // Ratcheting bands
    TUp := close[1] > TUp[1] ? max(Up, TUp[1]) : Up
    TDown := close[1] < TDown[1] ? min(Dn, TDown[1]) : Dn
    
    // Trend detection
    Trend := close > TDown[1] ? 1: close < TUp[1]? -1: nz(Trend[1], 1)
    ```
    
    Args:
        pivot_lookback: Bars on each side to confirm pivot (default: 2, 原版默认)
        atr_period: ATR calculation period (default: 10, 原版默认)
        atr_factor: Multiplier for ATR bands (default: 3.0, 原版默认)
        min_bars_for_signal: Minimum bars before generating signals
    """
    
    # 使用 TradingView 原版默认参数
    pivot_lookback: int = 2      # prd = 2
    atr_period: int = 10         # Pd = 10
    atr_factor: float = 3.0      # Factor = 3
    min_bars_for_signal: int = 20
    
    # Internal state
    _bars: list[Bar] = field(default_factory=list, init=False)
    _highs: list[float] = field(default_factory=list, init=False)
    _lows: list[float] = field(default_factory=list, init=False)
    _closes: list[float] = field(default_factory=list, init=False)
    _tr_values: list[float] = field(default_factory=list, init=False)
    
    # Pivot points
    _last_pivot_high: float = field(default=0.0, init=False)
    _last_pivot_low: float = field(default=0.0, init=False)
    _last_pivot_price: float = field(default=0.0, init=False)  # 最近的 pivot（无论高低）
    
    # Center line (加权平均)
    _center: float = field(default=0.0, init=False)
    
    # SuperTrend state
    _direction: TrendDirection = field(default=TrendDirection.NONE, init=False)
    _trend: int = field(default=0, init=False)  # 1 = long, -1 = short, 0 = none
    _tup: float = field(default=0.0, init=False)  # Upper trailing stop (for downtrend)
    _tdown: float = field(default=0.0, init=False)  # Lower trailing stop (for uptrend)
    _prev_tup: float = field(default=0.0, init=False)
    _prev_tdown: float = field(default=0.0, init=False)
    _trend_duration: int = field(default=0, init=False)
    
    def update(self, bar: Bar) -> SuperTrendState:
        """
        Update with a new completed bar.
        
        Args:
            bar: Completed OHLC bar
            
        Returns:
            Current SuperTrend state
        """
        self._bars.append(bar)
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._closes.append(bar.close)
        
        bar_idx = len(self._bars) - 1
        
        # 1. Update True Range for ATR
        self._update_tr(bar)
        
        # 2. Check for pivot points (with confirmation delay)
        self._check_pivots(bar_idx)
        
        # 3. Update center line (weighted average of pivots)
        self._update_center()
        
        # 4. Update SuperTrend logic (Pine Script style)
        self._update_supertrend(bar)
        
        # Update trend duration
        self._trend_duration += 1
        
        # Keep buffers bounded
        self._trim_buffers()
        
        return self.current_state
    
    def _update_tr(self, bar: Bar) -> None:
        """Update True Range."""
        if len(self._bars) < 2:
            tr = bar.high - bar.low
        else:
            prev_close = self._closes[-2]
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
        self._tr_values.append(tr)
    
    def _get_atr(self) -> float:
        """Get current ATR value using SMA."""
        if len(self._tr_values) < self.atr_period:
            if not self._tr_values:
                return 0.0
            return sum(self._tr_values) / len(self._tr_values)
        
        return sum(self._tr_values[-self.atr_period:]) / self.atr_period
    
    def _check_pivots(self, current_idx: int) -> None:
        """
        Check for confirmed pivot points.
        
        Pine Script: pivothigh(prd, prd) / pivotlow(prd, prd)
        需要 prd 根 bar 在两侧确认
        """
        prd = self.pivot_lookback
        
        # Need at least 2*prd + 1 bars
        if len(self._bars) < 2 * prd + 1:
            return
        
        # Check the bar at position (current - prd) for pivot status
        check_idx = current_idx - prd
        if check_idx < prd:
            return
        
        check_high = self._highs[check_idx]
        check_low = self._lows[check_idx]
        
        # Check for pivot high: highest high in range
        is_pivot_high = True
        for i in range(check_idx - prd, check_idx + prd + 1):
            if i == check_idx:
                continue
            if i < 0 or i >= len(self._highs):
                is_pivot_high = False
                break
            if self._highs[i] >= check_high:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            self._last_pivot_high = check_high
            self._last_pivot_price = check_high
        
        # Check for pivot low: lowest low in range
        is_pivot_low = True
        for i in range(check_idx - prd, check_idx + prd + 1):
            if i == check_idx:
                continue
            if i < 0 or i >= len(self._lows):
                is_pivot_low = False
                break
            if self._lows[i] <= check_low:
                is_pivot_low = False
                break
        
        if is_pivot_low:
            self._last_pivot_low = check_low
            self._last_pivot_price = check_low
    
    def _update_center(self) -> None:
        """
        Update center line using pivot points.
        
        Pine Script:
        ```
        lastpp = ph ? ph : pl ? pl : na
        if lastpp
            if na(center)
                center := lastpp
            else
                center := (center * 2 + lastpp) / 3
        ```
        
        使用加权平均：新值权重 1/3，旧值权重 2/3
        """
        if self._last_pivot_price == 0:
            return
        
        if self._center == 0:
            # Initialize
            self._center = self._last_pivot_price
        else:
            # Weighted average: (center * 2 + lastpp) / 3
            self._center = (self._center * 2 + self._last_pivot_price) / 3
    
    def _update_supertrend(self, bar: Bar) -> None:
        """
        Update SuperTrend logic following Pine Script exactly.
        
        Pine Script:
        ```
        Up = center - (Factor * atr(Pd))
        Dn = center + (Factor * atr(Pd))
        
        TUp := close[1] > TUp[1] ? max(Up, TUp[1]) : Up
        TDown := close[1] < TDown[1] ? min(Dn, TDown[1]) : Dn
        
        Trend := close > TDown[1] ? 1: close < TUp[1]? -1: nz(Trend[1], 1)
        Trailingsl = Trend == 1 ? TUp : TDown
        ```
        """
        if self._center == 0:
            return
        
        atr = self._get_atr()
        if atr == 0:
            return
        
        # Calculate raw bands from center line
        up = self._center - (self.atr_factor * atr)  # Lower band (support)
        dn = self._center + (self.atr_factor * atr)  # Upper band (resistance)
        
        # Store previous values for comparison
        self._prev_tup = self._tup
        self._prev_tdown = self._tdown
        
        # Get previous close
        prev_close = self._closes[-2] if len(self._closes) >= 2 else bar.close
        
        # Ratcheting logic (Pine Script style)
        # TUp: 如果前一根 close > 前一个 TUp，则只能上移
        if self._tup == 0:
            self._tup = up
        else:
            if prev_close > self._prev_tup:
                self._tup = max(up, self._prev_tup)
            else:
                self._tup = up
        
        # TDown: 如果前一根 close < 前一个 TDown，则只能下移
        if self._tdown == 0:
            self._tdown = dn
        else:
            if prev_close < self._prev_tdown:
                self._tdown = min(dn, self._prev_tdown)
            else:
                self._tdown = dn
        
        # Trend detection (use previous bar's bands for current signal)
        # Trend := close > TDown[1] ? 1: close < TUp[1]? -1: nz(Trend[1], 1)
        prev_trend = self._trend
        
        if self._prev_tdown > 0 and bar.close > self._prev_tdown:
            self._trend = 1  # Long
        elif self._prev_tup > 0 and bar.close < self._prev_tup:
            self._trend = -1  # Short
        elif self._trend == 0:
            # Initialize: default to long if no prior trend
            self._trend = 1
        # else: keep previous trend
        
        # Update direction enum
        if self._trend == 1:
            self._direction = TrendDirection.LONG
        elif self._trend == -1:
            self._direction = TrendDirection.SHORT
        else:
            self._direction = TrendDirection.NONE
        
        # Reset duration on trend change
        if prev_trend != 0 and prev_trend != self._trend:
            self._trend_duration = 0
            logger.debug(
                "SuperTrend direction changed",
                old="LONG" if prev_trend == 1 else "SHORT",
                new=self._direction.name,
                close=round(bar.close, 2),
                tup=round(self._tup, 2),
                tdown=round(self._tdown, 2),
                center=round(self._center, 2),
            )
    
    def _trim_buffers(self) -> None:
        """Keep buffers bounded to prevent memory growth."""
        max_size = max(100, self.atr_period * 3, self.pivot_lookback * 10)
        
        if len(self._bars) > max_size:
            trim = len(self._bars) - max_size
            self._bars = self._bars[trim:]
            self._highs = self._highs[trim:]
            self._lows = self._lows[trim:]
            self._closes = self._closes[trim:]
        
        if len(self._tr_values) > max_size:
            self._tr_values = self._tr_values[-max_size:]
    
    @property
    def current_state(self) -> SuperTrendState:
        """Get current SuperTrend state."""
        atr = self._get_atr()
        
        # Trailing stop line: TUp in uptrend, TDown in downtrend
        if self._trend == 1:
            st_line = self._tup
        elif self._trend == -1:
            st_line = self._tdown
        else:
            st_line = self._center if self._center > 0 else 0.0
        
        return SuperTrendState(
            direction=self._direction,
            supertrend_line=st_line,
            bandwidth=atr * self.atr_factor * 2,
            trend_duration=self._trend_duration,
            pivot_high=self._last_pivot_high,
            pivot_low=self._last_pivot_low,
            center_line=self._center,
            atr=atr,
        )
    
    @property
    def current_direction(self) -> TrendDirection:
        """Get current trend direction."""
        return self._direction
    
    @property
    def is_ready(self) -> bool:
        """Check if enough bars for reliable signals."""
        return len(self._bars) >= self.min_bars_for_signal and self._center > 0
    
    def initialize_from_bars(self, bars: Sequence[Bar]) -> None:
        """
        Initialize from historical bars.
        
        Args:
            bars: Historical bars (oldest first)
        """
        logger.info(f"Initializing PivotSuperTrend with {len(bars)} bars")
        
        for bar in bars:
            self.update(bar)
        
        logger.info(
            "PivotSuperTrend initialized",
            direction=self._direction.name,
            supertrend_line=round(self.current_state.supertrend_line, 2),
            center_line=round(self._center, 2),
            pivot_high=round(self._last_pivot_high, 2),
            pivot_low=round(self._last_pivot_low, 2),
        )
    
    def reset(self) -> None:
        """Reset all state."""
        self._bars.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._tr_values.clear()
        self._last_pivot_high = 0.0
        self._last_pivot_low = 0.0
        self._last_pivot_price = 0.0
        self._center = 0.0
        self._direction = TrendDirection.NONE
        self._trend = 0
        self._tup = 0.0
        self._tdown = 0.0
        self._prev_tup = 0.0
        self._prev_tdown = 0.0
        self._trend_duration = 0
