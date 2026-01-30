"""
AlphaOS Exit v2.1 - Enhanced Staged Exit with Bid/Ask + Cost Guard + Alignment Modulation

Key improvements over v2.0:
1. BID/ASK price selection:
   - LONG: Use BID for SL hit detection and PnL calculation (exit price)
   - SHORT: Use ASK for SL hit detection and PnL calculation (exit price)

2. Cost guard:
   - net_pnl_usd = unrealized_pnl_usd - est_commission - est_slippage - cost_buffer
   - BE/Partial/Trail triggers use NET P&L (not gross)

3. SL modification isolation:
   - sl_update_pending flag: Prevents sending MODIFY while awaiting broker ack
   - modify_cooldown_sec: Minimum time between SL modifications

4. Trend alignment modulation:
   - ALIGNED: Trade direction matches trend → wider thresholds (let winners run)
   - COUNTER: Trade direction opposes trend → tighter thresholds (protect profits)
   - UNKNOWN: Default thresholds

5. SL clamp rules:
   - LONG: new_sl < bid - min_sl_gap_price AND new_sl > current_sl
   - SHORT: new_sl > ask + min_sl_gap_price AND new_sl < current_sl

Architecture Principles (inherited from v2.0):
- Exit logic in OrderManager (single source of truth)
- Strictly causal (no future information)
- Single ExitDecision per tick
- Priority: FULL_CLOSE > PARTIAL_CLOSE > MOVE_SL > NOOP

Reference: Exit v2.1 白皮书
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Literal

from alphaos.core.config import ExitV21Config
from alphaos.core.types import SignalType, Tick


# ============================================================================
# Exit Decision Types
# ============================================================================

class ExitAction(IntEnum):
    """
    Exit action types (priority order, highest first).
    
    Single tick produces exactly ONE decision.
    """
    NOOP = 0           # No action needed
    MOVE_SL = 1        # Modify stop loss (BE or trailing)
    PARTIAL_CLOSE = 2  # Close partial position
    FULL_CLOSE = 3     # Close entire position


class ExitStage(IntEnum):
    """
    Exit stage tracking (monotonic progression).
    """
    INITIAL = 0        # Position just opened
    BE_DONE = 1        # Break-even completed
    PARTIAL_DONE = 2   # Partial close completed
    TRAILING = 3       # Trailing stop active


TrendAlignment = Literal["ALIGNED", "COUNTER", "UNKNOWN"]


@dataclass(slots=True)
class ExitDecision:
    """
    Decision output from ExitPolicyV21.evaluate().
    
    Includes audit fields for logging/replay.
    
    Attributes:
        action: What to do (NOOP/MOVE_SL/PARTIAL_CLOSE/FULL_CLOSE)
        new_sl: New stop loss price (only if action is MOVE_SL)
        close_lots: Lots to close (only if action is PARTIAL_CLOSE or FULL_CLOSE)
        reason: Human-readable reason for decision
        stage: Which stage triggered this (BE/PARTIAL/TRAILING/SL_HIT)
        
        # Audit fields (for logging/replay)
        price_used: Price used for decision (bid for LONG, ask for SHORT)
        bid: Current bid price
        ask: Current ask price
        threshold: Threshold that was compared against
        net_pnl_usd: Net P&L after costs
        gross_pnl_usd: Gross unrealized P&L
        alignment: Trend alignment at decision time
    """
    action: ExitAction = ExitAction.NOOP
    new_sl: float = 0.0
    close_lots: float = 0.0
    reason: str = ""
    stage: str = ""
    
    # Audit fields
    price_used: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    threshold: float = 0.0
    net_pnl_usd: float = 0.0
    gross_pnl_usd: float = 0.0
    alignment: str = "UNKNOWN"


# ============================================================================
# Position State v2.1 (Exit-aware with bid/ask tracking)
# ============================================================================

@dataclass
class PositionStateV21:
    """
    Extended position state for Exit v2.1.
    
    Key differences from v2.0:
    - Tracks current_bid/current_ask separately (not just mid)
    - Tracks best_price in direction-appropriate way (bid for LONG, ask for SHORT)
    - Includes cost estimation fields
    - Includes trend alignment context
    
    Field Categories:
    
    [immutable] - Set at position open:
        - ticket, symbol, direction, entry_price, entry_lots, entry_time_us
        - initial_sl, initial_tp
        - est_commission_usd, est_slippage_usd (calculated at open)
    
    [tick_must_update] - Updated every tick:
        - current_bid, current_ask, current_mid
        - unrealized_pnl_usd (gross), net_pnl_usd (after costs)
        - best_price (for trailing)
    
    [once_only_no_rollback] - Written once, never reverted:
        - stage, be_price, partial_price
        - trailing_active, trailing_sl
    
    [context] - External context (from InferenceResult):
        - trend_alignment, market_phase
    
    [isolation] - For pending/cooldown:
        - sl_update_pending, last_sl_update_time, last_modify_ack_time
    """
    
    # ======== Immutable (set at open) ========
    ticket: int = 0
    symbol: str = ""
    direction: SignalType = SignalType.NEUTRAL  # LONG or SHORT
    entry_price: float = 0.0
    entry_lots: float = 0.0
    entry_time_us: int = 0
    initial_sl: float = 0.0
    initial_tp: float = 0.0
    
    # Cost estimation (calculated at open based on config + lots)
    est_commission_usd: float = 0.0
    est_slippage_usd: float = 0.0
    
    # ======== tick_must_update ========
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_mid: float = 0.0
    current_lots: float = 0.0
    unrealized_pnl_usd: float = 0.0   # Gross (before costs)
    net_pnl_usd: float = 0.0          # After costs
    current_sl: float = 0.0           # Current broker SL
    
    # Best price tracking for trailing (direction-appropriate)
    best_price: float = 0.0           # Best favorable price since entry/partial
    
    # ======== once_only_no_rollback ========
    stage: ExitStage = field(default=ExitStage.INITIAL)
    be_price: float = 0.0             # SL after BE (locked once set)
    partial_price: float = 0.0        # Price at partial close
    partial_lots_closed: float = 0.0
    trailing_active: bool = False
    trailing_sl: float = 0.0          # Current trailing SL
    
    # ======== context (from InferenceResult) ========
    trend_alignment: TrendAlignment = "UNKNOWN"
    market_phase: str = ""            # e.g., "LAMINAR", "TURBULENT", etc.
    
    # ======== isolation (pending/cooldown) ========
    sl_update_pending: bool = False   # True while awaiting broker ack
    last_sl_update_time: float = 0.0  # time.time() of last SL update attempt
    last_modify_ack_time: float = 0.0 # time.time() of last successful ack
    last_exit_action_ts: float = 0.0  # time.time() of last any exit action
    
    def update_tick(
        self,
        tick: Tick,
        now: float,
        config: ExitV21Config,
    ) -> None:
        """
        Update tick-level fields (called every tick).
        
        Args:
            tick: Current tick with bid/ask
            now: Current time (time.time())
            config: Exit v2.1 configuration for cost calculation
        """
        self.current_bid = tick.bid
        self.current_ask = tick.ask
        self.current_mid = tick.mid
        
        # Calculate unrealized P&L using direction-appropriate price
        # LONG exits at BID (sell at bid), SHORT exits at ASK (buy at ask)
        if self.direction == SignalType.LONG:
            exit_price = tick.bid
            self.unrealized_pnl_usd = (exit_price - self.entry_price) * self.current_lots * config.tick_value_usd_per_lot
            # Update best price (highest bid for LONG)
            self.best_price = max(self.best_price, tick.bid)
        elif self.direction == SignalType.SHORT:
            exit_price = tick.ask
            self.unrealized_pnl_usd = (self.entry_price - exit_price) * self.current_lots * config.tick_value_usd_per_lot
            # Update best price (lowest ask for SHORT)
            if self.best_price == 0.0:
                self.best_price = tick.ask
            else:
                self.best_price = min(self.best_price, tick.ask)
        else:
            self.unrealized_pnl_usd = 0.0
        
        # Calculate net P&L after costs
        total_costs = self.est_commission_usd + self.est_slippage_usd + config.cost_buffer_usd
        self.net_pnl_usd = self.unrealized_pnl_usd - total_costs
    
    def get_exit_price(self) -> float:
        """Get the appropriate exit price based on direction."""
        if self.direction == SignalType.LONG:
            return self.current_bid
        else:  # SHORT
            return self.current_ask
    
    def mark_be_done(self, new_sl: float, now: float) -> None:
        """
        Mark BE stage as complete (once-only, no rollback).
        
        Args:
            new_sl: New stop loss price after BE trigger
            now: Current time (time.time())
        """
        if self.stage >= ExitStage.BE_DONE:
            return  # Already done, ignore
        
        self.stage = ExitStage.BE_DONE
        self.be_price = new_sl
        self.current_sl = new_sl
        self.trailing_sl = new_sl  # Initialize trailing from BE price
        self.last_sl_update_time = now
        self.last_exit_action_ts = now
    
    def mark_partial_done(self, close_price: float, lots_closed: float, now: float) -> None:
        """
        Mark partial close as complete (once-only, no rollback).
        
        Args:
            close_price: Price at which partial was executed
            lots_closed: Number of lots closed
            now: Current time (time.time())
        """
        if self.stage >= ExitStage.PARTIAL_DONE:
            return  # Already done, ignore
        
        self.stage = ExitStage.PARTIAL_DONE
        self.partial_price = close_price
        self.partial_lots_closed = lots_closed
        self.current_lots -= lots_closed
        self.last_exit_action_ts = now
        
        # Activate trailing after partial close
        self.trailing_active = True
        
        # Reset best_price for trailing from current price
        if self.direction == SignalType.LONG:
            self.best_price = self.current_bid
        else:
            self.best_price = self.current_ask
    
    def update_trailing_sl(self, new_sl: float, now: float) -> bool:
        """
        Update trailing stop loss (monotonic only).
        
        LONG: new_sl must be > trailing_sl (only moves up)
        SHORT: new_sl must be < trailing_sl (only moves down)
        
        Args:
            new_sl: Candidate new SL
            now: Current time
            
        Returns:
            True if update was valid and applied
        """
        if not self.trailing_active:
            return False
        
        if self.direction == SignalType.LONG:
            if new_sl > self.trailing_sl:
                self.trailing_sl = new_sl
                self.current_sl = new_sl
                self.last_sl_update_time = now
                return True
        else:  # SHORT
            if new_sl < self.trailing_sl:
                self.trailing_sl = new_sl
                self.current_sl = new_sl
                self.last_sl_update_time = now
                return True
        
        return False
    
    def set_pending(self) -> None:
        """Mark SL update as pending (awaiting broker ack)."""
        self.sl_update_pending = True
    
    def clear_pending(self, now: float) -> None:
        """Clear pending flag after broker ack."""
        self.sl_update_pending = False
        self.last_modify_ack_time = now
    
    @classmethod
    def from_position(
        cls,
        ticket: int,
        symbol: str,
        direction: SignalType,
        entry_price: float,
        entry_lots: float,
        entry_time_us: int,
        initial_sl: float,
        initial_tp: float,
        config: ExitV21Config,
        trend_alignment: TrendAlignment = "UNKNOWN",
        market_phase: str = "",
    ) -> "PositionStateV21":
        """
        Create PositionStateV21 from position data.
        
        Args:
            ticket: Position ticket
            symbol: Symbol
            direction: LONG or SHORT
            entry_price: Entry price
            entry_lots: Position size in lots
            entry_time_us: Entry timestamp in microseconds
            initial_sl: Initial stop loss
            initial_tp: Initial take profit
            config: Exit v2.1 config for cost estimation
            trend_alignment: Trend alignment context
            market_phase: Market phase context
        """
        # Calculate estimated costs
        est_commission = config.est_commission_usd_per_lot * entry_lots
        est_slippage = config.est_slippage_usd_per_lot * entry_lots
        
        return cls(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_lots=entry_lots,
            entry_time_us=entry_time_us,
            initial_sl=initial_sl,
            initial_tp=initial_tp,
            est_commission_usd=est_commission,
            est_slippage_usd=est_slippage,
            current_lots=entry_lots,
            current_sl=initial_sl,
            best_price=entry_price,  # Start with entry price
            trend_alignment=trend_alignment,
            market_phase=market_phase,
        )


# ============================================================================
# Exit Policy v2.1
# ============================================================================

class ExitPolicyV21:
    """
    Exit Policy v2.1 Implementation.
    
    Key features:
    - Bid/Ask price selection for direction-appropriate exit pricing
    - Cost guard: triggers use net P&L (after commission/slippage/buffer)
    - SL clamp rules: gap + monotonic + direction validity
    - Pending/cooldown isolation to prevent rapid MODIFY spam
    - Alignment multipliers for ALIGNED/COUNTER/UNKNOWN trend states
    
    Decision Priority (single tick): FULL_CLOSE > PARTIAL_CLOSE > MOVE_SL > NOOP
    """
    
    def __init__(self, config: ExitV21Config) -> None:
        """
        Initialize with configuration.
        
        Args:
            config: Exit v2.1 configuration
        """
        self.config = config
    
    def evaluate(
        self,
        state: PositionStateV21,
        now: float,
    ) -> ExitDecision:
        """
        Evaluate exit conditions and return single decision.
        
        Evaluation order (by priority):
        1. Check if SL hit → FULL_CLOSE
        2. Check partial trigger → PARTIAL_CLOSE
        3. Check BE trigger → MOVE_SL
        4. Check trailing update → MOVE_SL
        5. Otherwise → NOOP
        
        Args:
            state: Current position state (with tick already updated)
            now: Current time (time.time())
            
        Returns:
            Single ExitDecision for this tick
        """
        cfg = self.config
        
        # Get alignment multipliers
        alignment_mult = self._get_alignment_multipliers(state.trend_alignment)
        
        # Get direction-appropriate price
        price_used = state.get_exit_price()
        
        # ================================================================
        # 0. Check minimum hold time
        # ================================================================
        hold_time_sec = now - (state.entry_time_us / 1_000_000)
        if hold_time_sec < cfg.min_hold_seconds:
            return ExitDecision(
                action=ExitAction.NOOP,
                reason=f"Min hold time: {hold_time_sec:.1f}s < {cfg.min_hold_seconds}s",
                stage="HOLD_TIME",
                price_used=price_used,
                bid=state.current_bid,
                ask=state.current_ask,
                net_pnl_usd=state.net_pnl_usd,
                gross_pnl_usd=state.unrealized_pnl_usd,
                alignment=state.trend_alignment,
            )
        
        # ================================================================
        # 1. Check if SL hit (highest priority - FULL_CLOSE)
        # ================================================================
        if state.current_sl > 0:
            sl_hit = self._check_sl_hit(state, price_used)
            if sl_hit:
                return ExitDecision(
                    action=ExitAction.FULL_CLOSE,
                    close_lots=state.current_lots,
                    reason=f"SL hit at {price_used:.2f} (SL={state.current_sl:.2f})",
                    stage="SL_HIT",
                    price_used=price_used,
                    bid=state.current_bid,
                    ask=state.current_ask,
                    threshold=state.current_sl,
                    net_pnl_usd=state.net_pnl_usd,
                    gross_pnl_usd=state.unrealized_pnl_usd,
                    alignment=state.trend_alignment,
                )
        
        # ================================================================
        # 2. Check partial close trigger (after BE, before trailing active)
        # ================================================================
        partial_threshold = cfg.partial1_trigger_net_usd * alignment_mult.be_trigger_mult
        
        if (
            state.stage == ExitStage.BE_DONE
            and state.net_pnl_usd >= partial_threshold
        ):
            partial_lots = self._calculate_partial_lots(state)
            
            if partial_lots >= cfg.min_lots_to_partial:
                # Check post-partial cooldown
                time_since_be = now - state.last_exit_action_ts
                if time_since_be >= cfg.post_partial_cooldown_sec:
                    return ExitDecision(
                        action=ExitAction.PARTIAL_CLOSE,
                        close_lots=partial_lots,
                        reason=(
                            f"Partial trigger: net_pnl={state.net_pnl_usd:.2f} USD "
                            f">= {partial_threshold:.2f} USD, closing {partial_lots:.2f} lots"
                        ),
                        stage="PARTIAL",
                        price_used=price_used,
                        bid=state.current_bid,
                        ask=state.current_ask,
                        threshold=partial_threshold,
                        net_pnl_usd=state.net_pnl_usd,
                        gross_pnl_usd=state.unrealized_pnl_usd,
                        alignment=state.trend_alignment,
                    )
        
        # ================================================================
        # 3. Check BE trigger (first stage)
        # ================================================================
        be_threshold = cfg.be_trigger_net_usd * alignment_mult.be_trigger_mult
        
        if (
            state.stage == ExitStage.INITIAL
            and state.net_pnl_usd >= be_threshold
        ):
            # Check pending/cooldown
            if state.sl_update_pending:
                return ExitDecision(
                    action=ExitAction.NOOP,
                    reason="BE blocked: SL update pending",
                    stage="BE_PENDING",
                    price_used=price_used,
                    bid=state.current_bid,
                    ask=state.current_ask,
                    net_pnl_usd=state.net_pnl_usd,
                    gross_pnl_usd=state.unrealized_pnl_usd,
                    alignment=state.trend_alignment,
                )
            
            time_since_last_modify = now - state.last_sl_update_time
            if state.last_sl_update_time > 0 and time_since_last_modify < cfg.modify_cooldown_sec:
                return ExitDecision(
                    action=ExitAction.NOOP,
                    reason=f"BE cooldown: {time_since_last_modify:.2f}s < {cfg.modify_cooldown_sec}s",
                    stage="BE_COOLDOWN",
                    price_used=price_used,
                    bid=state.current_bid,
                    ask=state.current_ask,
                    net_pnl_usd=state.net_pnl_usd,
                    gross_pnl_usd=state.unrealized_pnl_usd,
                    alignment=state.trend_alignment,
                )
            
            # Calculate and validate BE SL
            be_sl = self._calculate_be_sl(state)
            
            if self._is_valid_sl(state, be_sl):
                return ExitDecision(
                    action=ExitAction.MOVE_SL,
                    new_sl=be_sl,
                    reason=(
                        f"BE trigger: net_pnl={state.net_pnl_usd:.2f} USD "
                        f">= {be_threshold:.2f} USD, moving SL to {be_sl:.2f}"
                    ),
                    stage="BE",
                    price_used=price_used,
                    bid=state.current_bid,
                    ask=state.current_ask,
                    threshold=be_threshold,
                    net_pnl_usd=state.net_pnl_usd,
                    gross_pnl_usd=state.unrealized_pnl_usd,
                    alignment=state.trend_alignment,
                )
        
        # ================================================================
        # 4. Check trailing stop update (after partial OR net_pnl >= trail_start)
        # ================================================================
        trail_start_threshold = cfg.trail_start_net_usd * alignment_mult.be_trigger_mult
        
        # Trailing can activate after partial done OR if net_pnl exceeds threshold
        should_trail = (
            state.stage >= ExitStage.PARTIAL_DONE
            or (state.stage >= ExitStage.BE_DONE and state.net_pnl_usd >= trail_start_threshold)
        )
        
        if should_trail and state.stage >= ExitStage.BE_DONE:
            # Activate trailing if not already
            if not state.trailing_active:
                state.trailing_active = True
                state.trailing_sl = state.current_sl
            
            # Check pending/cooldown
            if state.sl_update_pending:
                return ExitDecision(
                    action=ExitAction.NOOP,
                    reason="Trailing blocked: SL update pending",
                    stage="TRAILING_PENDING",
                    price_used=price_used,
                    bid=state.current_bid,
                    ask=state.current_ask,
                    net_pnl_usd=state.net_pnl_usd,
                    gross_pnl_usd=state.unrealized_pnl_usd,
                    alignment=state.trend_alignment,
                )
            
            time_since_last_modify = now - state.last_sl_update_time
            if time_since_last_modify < cfg.modify_cooldown_sec:
                return ExitDecision(
                    action=ExitAction.NOOP,
                    reason=f"Trailing cooldown: {time_since_last_modify:.2f}s < {cfg.modify_cooldown_sec}s",
                    stage="TRAILING_COOLDOWN",
                    price_used=price_used,
                    bid=state.current_bid,
                    ask=state.current_ask,
                    net_pnl_usd=state.net_pnl_usd,
                    gross_pnl_usd=state.unrealized_pnl_usd,
                    alignment=state.trend_alignment,
                )
            
            # Calculate candidate trailing SL
            trail_distance = cfg.trail_distance_price * alignment_mult.trail_distance_mult
            candidate_sl = self._calculate_trailing_sl(state, trail_distance)
            
            if candidate_sl is not None and self._is_valid_sl(state, candidate_sl):
                # Check minimum step
                if self._is_valid_trailing_move(state, candidate_sl):
                    return ExitDecision(
                        action=ExitAction.MOVE_SL,
                        new_sl=candidate_sl,
                        reason=(
                            f"Trailing update: best={state.best_price:.2f}, "
                            f"moving SL from {state.trailing_sl:.2f} to {candidate_sl:.2f}"
                        ),
                        stage="TRAILING",
                        price_used=price_used,
                        bid=state.current_bid,
                        ask=state.current_ask,
                        threshold=trail_distance,
                        net_pnl_usd=state.net_pnl_usd,
                        gross_pnl_usd=state.unrealized_pnl_usd,
                        alignment=state.trend_alignment,
                    )
        
        # ================================================================
        # 5. No action needed
        # ================================================================
        return ExitDecision(
            action=ExitAction.NOOP,
            reason="No exit condition met",
            stage="HOLD",
            price_used=price_used,
            bid=state.current_bid,
            ask=state.current_ask,
            net_pnl_usd=state.net_pnl_usd,
            gross_pnl_usd=state.unrealized_pnl_usd,
            alignment=state.trend_alignment,
        )
    
    def _get_alignment_multipliers(self, alignment: TrendAlignment) -> "AlignmentMultiplierConfig":
        """Get multipliers for trend alignment."""
        from alphaos.core.config import AlignmentMultiplierConfig
        
        mult_cfg = self.config.alignment_multipliers
        
        if alignment == "ALIGNED":
            return mult_cfg.aligned
        elif alignment == "COUNTER":
            return mult_cfg.counter
        else:
            return mult_cfg.unknown
    
    def _check_sl_hit(self, state: PositionStateV21, price_used: float) -> bool:
        """
        Check if current price has hit the stop loss.
        
        Uses direction-appropriate price (bid for LONG, ask for SHORT).
        """
        if state.current_sl <= 0:
            return False
        
        if state.direction == SignalType.LONG:
            # LONG: SL hit when bid <= SL
            return price_used <= state.current_sl
        else:  # SHORT
            # SHORT: SL hit when ask >= SL
            return price_used >= state.current_sl
    
    def _calculate_be_sl(self, state: PositionStateV21) -> float:
        """Calculate BE stop loss price."""
        cfg = self.config
        
        if state.direction == SignalType.LONG:
            # LONG: SL at entry + offset (locked in small profit)
            be_sl = state.entry_price + cfg.be_offset_price
        else:  # SHORT
            # SHORT: SL at entry - offset
            be_sl = state.entry_price - cfg.be_offset_price
        
        return round(be_sl, cfg.price_precision)
    
    def _calculate_partial_lots(self, state: PositionStateV21) -> float:
        """Calculate lots to close in partial stage."""
        cfg = self.config
        
        # Calculate partial close amount
        partial_lots = state.current_lots * cfg.partial1_ratio
        
        # Round to lot precision
        partial_lots = round(partial_lots, 2)
        
        # Ensure remaining position >= min_lots_to_partial
        remaining = state.current_lots - partial_lots
        if remaining < cfg.min_lots_to_partial:
            # Close less to keep minimum lot
            partial_lots = state.current_lots - cfg.min_lots_to_partial
            partial_lots = max(0.0, round(partial_lots, 2))
        
        return partial_lots
    
    def _calculate_trailing_sl(
        self,
        state: PositionStateV21,
        trail_distance: float,
    ) -> float | None:
        """
        Calculate candidate trailing SL.
        
        Returns None if no valid SL can be calculated.
        """
        cfg = self.config
        
        if state.direction == SignalType.LONG:
            # LONG: Trail below the best bid
            candidate = state.best_price - trail_distance
        else:  # SHORT
            # SHORT: Trail above the best ask
            candidate = state.best_price + trail_distance
        
        return round(candidate, cfg.price_precision)
    
    def _is_valid_sl(self, state: PositionStateV21, candidate_sl: float) -> bool:
        """
        Validate that SL is valid according to clamp rules.
        
        Rules:
        - LONG: candidate_sl < bid - min_sl_gap_price
        - SHORT: candidate_sl > ask + min_sl_gap_price
        - Monotonic: LONG SL only moves up, SHORT SL only moves down
        """
        cfg = self.config
        
        if state.direction == SignalType.LONG:
            # LONG: SL must be below bid with gap
            max_sl = state.current_bid - cfg.min_sl_gap_price
            if candidate_sl >= max_sl:
                return False
            
            # Monotonic: new SL must be >= current SL (only moves up)
            if state.current_sl > 0 and candidate_sl < state.current_sl:
                return False
            
        else:  # SHORT
            # SHORT: SL must be above ask with gap
            min_sl = state.current_ask + cfg.min_sl_gap_price
            if candidate_sl <= min_sl:
                return False
            
            # Monotonic: new SL must be <= current SL (only moves down)
            if state.current_sl > 0 and candidate_sl > state.current_sl:
                return False
        
        return True
    
    def _is_valid_trailing_move(
        self,
        state: PositionStateV21,
        candidate_sl: float,
    ) -> bool:
        """
        Check if trailing SL move meets minimum step requirement.
        """
        cfg = self.config
        
        if state.trailing_sl <= 0:
            return True  # First trailing SL, always valid
        
        if state.direction == SignalType.LONG:
            # Must move up by at least trail_step_price
            return candidate_sl >= state.trailing_sl + cfg.trail_step_price
        else:  # SHORT
            # Must move down by at least trail_step_price
            return candidate_sl <= state.trailing_sl - cfg.trail_step_price
    
    def get_config(self) -> ExitV21Config:
        """Get current configuration."""
        return self.config
