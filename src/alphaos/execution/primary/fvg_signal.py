"""
Fair Value Gap (FVG) Signal Generator (v4.0)

Detects Fair Value Gaps and generates primary trading signals.

FVG Definition:
- Bullish FVG: Gap between Bar[i-2].high and Bar[i].low (imbalance zone)
- Bearish FVG: Gap between Bar[i-2].low and Bar[i].high (imbalance zone)

Entry Logic:
- Wait for price to retrace into FVG zone
- Enter when price touches the Consequent Encroachment (CE) midpoint
- Stop loss at structure (swing high/low)

This module combines with PivotSuperTrend to form the complete Primary engine:
- SuperTrend provides trend direction filter (high-level trend)
- FVG provides entry timing (pullback entries within trend)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Sequence

from alphaos.core.logging import get_logger
from alphaos.v4.types import Bar
from alphaos.execution.primary.pivot_supertrend import TrendDirection, SuperTrendState

logger = get_logger(__name__)


class FVGType(Enum):
    """Fair Value Gap types."""
    BULLISH = "bullish"    # Gap up - price should retrace down to fill
    BEARISH = "bearish"    # Gap down - price should retrace up to fill


@dataclass
class FVGSignal:
    """
    Detected Fair Value Gap.
    
    Attributes:
        fvg_type: Bullish or bearish
        top: Upper boundary of FVG zone
        bottom: Lower boundary of FVG zone
        ce_midpoint: Consequent Encroachment midpoint (50% of FVG)
        size_bps: FVG size in basis points
        bar_idx: Index of the bar that created the FVG
        created_time: Timestamp when FVG was detected
        is_active: Whether FVG is still valid (not filled)
        age_bars: Number of bars since creation
    """
    fvg_type: FVGType
    top: float
    bottom: float
    ce_midpoint: float
    size_bps: float
    bar_idx: int
    created_time: datetime
    is_active: bool = True
    age_bars: int = 0
    
    @property
    def width(self) -> float:
        """FVG width in price."""
        return self.top - self.bottom
    
    def contains_price(self, price: float, tolerance_bps: float = 0.0) -> bool:
        """Check if price is within FVG zone (with optional tolerance)."""
        tol = (self.top + self.bottom) / 2 * tolerance_bps / 10000
        return (self.bottom - tol) <= price <= (self.top + tol)
    
    def is_at_ce(self, price: float, tolerance_bps: float = 3.0) -> bool:
        """Check if price is at Consequent Encroachment midpoint."""
        tol = self.ce_midpoint * tolerance_bps / 10000
        return abs(price - self.ce_midpoint) <= tol
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.fvg_type.name,
            "top": round(self.top, 2),
            "bottom": round(self.bottom, 2),
            "ce_midpoint": round(self.ce_midpoint, 2),
            "size_bps": round(self.size_bps, 2),
            "age_bars": self.age_bars,
            "is_active": self.is_active,
        }


@dataclass
class FVGDetector:
    """
    Fair Value Gap detector.
    
    Scans bars for FVG patterns and manages active FVG zones.
    
    Args:
        min_size_bps: Minimum FVG size in basis points (default: 5.0)
        max_age_bars: Maximum bars before FVG expires (default: 20)
        ce_tolerance_bps: Tolerance for CE midpoint entry (default: 3.0)
    """
    
    min_size_bps: float = 5.0
    max_age_bars: int = 20
    ce_tolerance_bps: float = 3.0
    
    # Internal state
    _bars: list[Bar] = field(default_factory=list, init=False)
    _active_fvgs: list[FVGSignal] = field(default_factory=list, init=False)
    _filled_fvgs: list[FVGSignal] = field(default_factory=list, init=False)
    
    def update(self, bar: Bar) -> list[FVGSignal]:
        """
        Update with a new completed bar.
        
        Args:
            bar: Completed OHLC bar
            
        Returns:
            List of newly detected FVGs
        """
        self._bars.append(bar)
        bar_idx = len(self._bars) - 1
        
        # Age existing FVGs
        for fvg in self._active_fvgs:
            fvg.age_bars += 1
        
        # Expire old FVGs
        self._expire_old_fvgs()
        
        # Check if current bar filled any FVGs
        self._check_fvg_fills(bar)
        
        # Detect new FVGs (need at least 3 bars)
        new_fvgs = []
        if len(self._bars) >= 3:
            fvg = self._detect_fvg(bar_idx)
            if fvg:
                new_fvgs.append(fvg)
                self._active_fvgs.append(fvg)
        
        return new_fvgs
    
    def _detect_fvg(self, current_idx: int) -> FVGSignal | None:
        """
        Detect FVG in the last 3 bars.
        
        FVG is created by 3-bar pattern:
        - Bar[0] (2 bars ago): Reference bar
        - Bar[1] (1 bar ago): Impulse bar
        - Bar[2] (current): Confirmation bar
        
        Bullish FVG: Bar[0].high < Bar[2].low (gap between)
        Bearish FVG: Bar[0].low > Bar[2].high (gap between)
        """
        bar_0 = self._bars[current_idx - 2]  # 2 bars ago
        bar_1 = self._bars[current_idx - 1]  # 1 bar ago (impulse)
        bar_2 = self._bars[current_idx]      # Current bar
        
        mid_price = (bar_1.high + bar_1.low) / 2
        
        # Check for Bullish FVG (gap up)
        if bar_0.high < bar_2.low:
            gap_bottom = bar_0.high
            gap_top = bar_2.low
            size_bps = (gap_top - gap_bottom) / mid_price * 10000
            
            if size_bps >= self.min_size_bps:
                return FVGSignal(
                    fvg_type=FVGType.BULLISH,
                    top=gap_top,
                    bottom=gap_bottom,
                    ce_midpoint=(gap_top + gap_bottom) / 2,
                    size_bps=size_bps,
                    bar_idx=current_idx,
                    created_time=bar_2.time,
                )
        
        # Check for Bearish FVG (gap down)
        if bar_0.low > bar_2.high:
            gap_top = bar_0.low
            gap_bottom = bar_2.high
            size_bps = (gap_top - gap_bottom) / mid_price * 10000
            
            if size_bps >= self.min_size_bps:
                return FVGSignal(
                    fvg_type=FVGType.BEARISH,
                    top=gap_top,
                    bottom=gap_bottom,
                    ce_midpoint=(gap_top + gap_bottom) / 2,
                    size_bps=size_bps,
                    bar_idx=current_idx,
                    created_time=bar_2.time,
                )
        
        return None
    
    def _check_fvg_fills(self, bar: Bar) -> None:
        """Check if current bar fills any active FVGs."""
        still_active = []
        
        for fvg in self._active_fvgs:
            filled = False
            
            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG is filled when price drops below its bottom
                if bar.low <= fvg.bottom:
                    filled = True
            else:
                # Bearish FVG is filled when price rises above its top
                if bar.high >= fvg.top:
                    filled = True
            
            if filled:
                fvg.is_active = False
                self._filled_fvgs.append(fvg)
            else:
                still_active.append(fvg)
        
        self._active_fvgs = still_active
    
    def _expire_old_fvgs(self) -> None:
        """Remove FVGs that are too old."""
        still_active = []
        
        for fvg in self._active_fvgs:
            if fvg.age_bars < self.max_age_bars:
                still_active.append(fvg)
            else:
                fvg.is_active = False
        
        self._active_fvgs = still_active
    
    def get_fvgs_for_direction(self, direction: TrendDirection) -> list[FVGSignal]:
        """
        Get active FVGs that align with trend direction.
        
        Args:
            direction: Current trend direction
            
        Returns:
            FVGs that can be traded in this direction
        """
        if direction == TrendDirection.LONG:
            return [f for f in self._active_fvgs if f.fvg_type == FVGType.BULLISH]
        elif direction == TrendDirection.SHORT:
            return [f for f in self._active_fvgs if f.fvg_type == FVGType.BEARISH]
        return []
    
    def check_ce_entry(
        self, 
        price: float, 
        direction: TrendDirection,
    ) -> FVGSignal | None:
        """
        Check if price is at CE entry level for any aligned FVG.
        
        Args:
            price: Current price
            direction: Required trend direction
            
        Returns:
            FVG if entry condition met, None otherwise
        """
        aligned_fvgs = self.get_fvgs_for_direction(direction)
        
        for fvg in aligned_fvgs:
            if fvg.is_at_ce(price, self.ce_tolerance_bps):
                return fvg
        
        return None
    
    @property
    def active_fvg_count(self) -> int:
        """Number of active FVGs."""
        return len(self._active_fvgs)
    
    @property
    def active_fvgs(self) -> list[FVGSignal]:
        """Get all active FVGs (copy)."""
        return list(self._active_fvgs)
    
    def initialize_from_bars(self, bars: Sequence[Bar]) -> None:
        """Initialize from historical bars."""
        logger.info(f"Initializing FVGDetector with {len(bars)} bars")
        
        for bar in bars:
            self.update(bar)
        
        logger.info(
            "FVGDetector initialized",
            active_fvgs=self.active_fvg_count,
            bullish=[f.to_dict() for f in self._active_fvgs if f.fvg_type == FVGType.BULLISH],
            bearish=[f.to_dict() for f in self._active_fvgs if f.fvg_type == FVGType.BEARISH],
        )
    
    def reset(self) -> None:
        """Reset all state."""
        self._bars.clear()
        self._active_fvgs.clear()
        self._filled_fvgs.clear()


@dataclass
class PrimarySignal:
    """
    Primary signal output from the combined PivotSuperTrend + FVG engine.
    
    Attributes:
        direction: Signal direction (LONG/SHORT)
        entry_price: Recommended entry (CE midpoint)
        stop_loss: Structure-based stop loss
        supertrend_state: Full SuperTrend state
        fvg: The FVG that triggered this signal
        confidence_factors: Dict of factors affecting signal quality
    """
    direction: TrendDirection
    entry_price: float
    stop_loss: float
    supertrend_state: SuperTrendState
    fvg: FVGSignal
    confidence_factors: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "direction": self.direction.name,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "supertrend": self.supertrend_state.to_dict(),
            "fvg": self.fvg.to_dict(),
            "confidence_factors": self.confidence_factors,
        }


@dataclass
class PrimarySignalGenerator:
    """
    Combined Primary Signal Generator.
    
    Integrates PivotSuperTrend (trend filter) + FVGDetector (entry timing).
    
    Signal Generation Logic:
    1. SuperTrend provides trend direction (LONG/SHORT filter)
    2. FVG provides entry zones (pullback into value)
    3. Signal fires when:
       a. SuperTrend direction is clear
       b. Price enters aligned FVG zone (at CE midpoint)
       c. Structure stop is valid
    
    Args:
        supertrend: PivotSuperTrend instance
        fvg_detector: FVGDetector instance  
        min_trend_duration: Minimum bars in trend before signals (default: 3)
        sl_buffer_bps: Buffer added to structure SL (default: 5.0)
    """
    
    from alphaos.execution.primary.pivot_supertrend import PivotSuperTrend
    
    supertrend: PivotSuperTrend
    fvg_detector: FVGDetector
    min_trend_duration: int = 3
    sl_buffer_bps: float = 5.0
    
    # Internal state
    _bar_count: int = field(default=0, init=False)
    _signal_count: int = field(default=0, init=False)
    _cooldown_until_bar: int = field(default=0, init=False)
    _cooldown_bars: int = field(default=5, init=False)
    
    def update(self, bar: Bar, current_price: float | None = None) -> PrimarySignal | None:
        """
        Update with a new bar and check for signals.
        
        Args:
            bar: Completed OHLC bar
            current_price: Optional current price (if different from bar.close)
            
        Returns:
            PrimarySignal if conditions met, None otherwise
        """
        self._bar_count += 1
        
        # Update components
        st_state = self.supertrend.update(bar)
        new_fvgs = self.fvg_detector.update(bar)
        
        # Use bar close as current price if not provided
        price = current_price if current_price is not None else bar.close
        
        # Check for signal
        signal = self._check_signal(price, st_state, bar)
        
        if signal:
            self._signal_count += 1
            self._cooldown_until_bar = self._bar_count + self._cooldown_bars
        
        return signal
    
    def _check_signal(
        self, 
        price: float, 
        st_state: SuperTrendState,
        bar: Bar,
    ) -> PrimarySignal | None:
        """Check if signal conditions are met."""
        # Check cooldown
        if self._bar_count < self._cooldown_until_bar:
            return None
        
        # Need clear trend direction
        if st_state.direction == TrendDirection.NONE:
            return None
        
        # Need minimum trend duration
        if st_state.trend_duration < self.min_trend_duration:
            return None
        
        # Check for aligned FVG at CE entry
        entry_fvg = self.fvg_detector.check_ce_entry(
            price, 
            st_state.direction,
        )
        
        if entry_fvg is None:
            return None
        
        # Calculate structure stop loss
        sl = self._calculate_stop_loss(st_state, entry_fvg)
        
        if sl is None:
            return None
        
        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(
            st_state, entry_fvg, price, sl
        )
        
        return PrimarySignal(
            direction=st_state.direction,
            entry_price=entry_fvg.ce_midpoint,
            stop_loss=sl,
            supertrend_state=st_state,
            fvg=entry_fvg,
            confidence_factors=confidence_factors,
        )
    
    def _calculate_stop_loss(
        self, 
        st_state: SuperTrendState,
        fvg: FVGSignal,
    ) -> float | None:
        """
        Calculate structure-based stop loss.
        
        For LONG: SL below recent pivot low or FVG bottom
        For SHORT: SL above recent pivot high or FVG top
        """
        buffer = fvg.ce_midpoint * self.sl_buffer_bps / 10000
        
        if st_state.direction == TrendDirection.LONG:
            # Use lower of: pivot low, FVG bottom, or SuperTrend line
            candidates = [fvg.bottom]
            if st_state.pivot_low > 0:
                candidates.append(st_state.pivot_low)
            if st_state.supertrend_line > 0:
                candidates.append(st_state.supertrend_line)
            
            sl = min(candidates) - buffer
            
        elif st_state.direction == TrendDirection.SHORT:
            # Use higher of: pivot high, FVG top, or SuperTrend line
            candidates = [fvg.top]
            if st_state.pivot_high > 0:
                candidates.append(st_state.pivot_high)
            if st_state.supertrend_line > 0:
                candidates.append(st_state.supertrend_line)
            
            sl = max(candidates) + buffer
        else:
            return None
        
        return sl
    
    def _calculate_confidence_factors(
        self,
        st_state: SuperTrendState,
        fvg: FVGSignal,
        entry: float,
        sl: float,
    ) -> dict:
        """Calculate factors that affect signal quality."""
        # Risk (distance to SL)
        risk = abs(entry - sl)
        risk_bps = risk / entry * 10000
        
        return {
            "trend_duration": st_state.trend_duration,
            "fvg_age": fvg.age_bars,
            "fvg_size_bps": fvg.size_bps,
            "risk_bps": round(risk_bps, 2),
            "atr": round(st_state.atr, 4),
            "bandwidth": round(st_state.bandwidth, 4),
        }
    
    def check_signal_at_price(self, price: float) -> PrimarySignal | None:
        """
        Check for signal at a specific price (without bar update).
        
        Useful for tick-level checks between bar completions.
        
        Args:
            price: Current price to check
            
        Returns:
            PrimarySignal if conditions met, None otherwise
        """
        st_state = self.supertrend.current_state
        
        if st_state.direction == TrendDirection.NONE:
            return None
        
        if st_state.trend_duration < self.min_trend_duration:
            return None
        
        # Check cooldown
        if self._bar_count < self._cooldown_until_bar:
            return None
        
        # Check for aligned FVG at CE entry
        entry_fvg = self.fvg_detector.check_ce_entry(price, st_state.direction)
        
        if entry_fvg is None:
            return None
        
        sl = self._calculate_stop_loss(st_state, entry_fvg)
        if sl is None:
            return None
        
        confidence_factors = self._calculate_confidence_factors(
            st_state, entry_fvg, price, sl
        )
        
        return PrimarySignal(
            direction=st_state.direction,
            entry_price=entry_fvg.ce_midpoint,
            stop_loss=sl,
            supertrend_state=st_state,
            fvg=entry_fvg,
            confidence_factors=confidence_factors,
        )
    
    @property
    def current_direction(self) -> TrendDirection:
        """Get current trend direction."""
        return self.supertrend.current_direction
    
    @property
    def is_ready(self) -> bool:
        """Check if enough data for reliable signals."""
        return self.supertrend.is_ready
    
    def initialize_from_bars(self, bars: Sequence[Bar]) -> None:
        """Initialize from historical bars."""
        logger.info(f"Initializing PrimarySignalGenerator with {len(bars)} bars")
        
        for bar in bars:
            self.supertrend.update(bar)
            self.fvg_detector.update(bar)
            self._bar_count += 1
        
        logger.info(
            "PrimarySignalGenerator initialized",
            direction=self.supertrend.current_direction.name,
            active_fvgs=self.fvg_detector.active_fvg_count,
            bar_count=self._bar_count,
        )
    
    def reset(self) -> None:
        """Reset all state."""
        self.supertrend.reset()
        self.fvg_detector.reset()
        self._bar_count = 0
        self._signal_count = 0
        self._cooldown_until_bar = 0
    
    def get_stats(self) -> dict:
        """Get generator statistics."""
        return {
            "bar_count": self._bar_count,
            "signal_count": self._signal_count,
            "direction": self.supertrend.current_direction.name,
            "active_fvgs": self.fvg_detector.active_fvg_count,
            "trend_duration": self.supertrend.current_state.trend_duration,
            "is_ready": self.is_ready,
        }
