"""
AlphaOS Risk Manager

Implements risk controls:
- Position size limits
- Daily loss limits
- Consecutive loss tracking
- T-S phase-based filters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Any
from zoneinfo import ZoneInfo

from alphaos.core.config import RiskConfig
from alphaos.core.logging import get_logger
from alphaos.core.types import Signal, SignalType, MarketPhase, OrderResult, OrderStatus

logger = get_logger(__name__)


# ============================================================================
# Risk Events
# ============================================================================

class RiskEvent:
    """Risk event types for circuit breakers."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    FROZEN_MARKET = "frozen_market"
    HIGH_ENTROPY = "high_entropy"
    POSITION_LIMIT = "position_limit"


# ============================================================================
# Risk Manager
# ============================================================================

@dataclass
class RiskManager:
    """
    Risk management for live trading.
    
    Implements multiple layers of protection:
    1. Position sizing limits
    2. Daily P&L limits (circuit breaker)
    3. Consecutive loss limits
    4. Market phase filters
    """
    
    config: RiskConfig
    
    # State
    _daily_pnl: float = field(default=0.0, init=False)
    _current_date: date | None = field(default=None, init=False)
    _consecutive_losses: int = field(default=0, init=False)
    _total_trades: int = field(default=0, init=False)
    _winning_trades: int = field(default=0, init=False)
    _is_halted: bool = field(default=False, init=False)
    _halt_reason: str | None = field(default=None, init=False)
    _current_exposure: float = field(default=0.0, init=False)
    
    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._is_halted
    
    @property
    def halt_reason(self) -> str | None:
        """Get reason for trading halt."""
        return self._halt_reason
    
    def reset_daily(self) -> None:
        """Reset daily counters (call at start of new day)."""
        self._daily_pnl = 0.0
        self._current_date = self._current_trading_date()
        
        # Don't reset halt if it's from consecutive losses
        if self._halt_reason == RiskEvent.DAILY_LOSS_LIMIT:
            self._is_halted = False
            self._halt_reason = None
        
        logger.info("Daily risk counters reset")
    
    def check_signal(self, signal: Signal, context: str = "entry") -> tuple[bool, str | None]:
        """
        Check if a signal should be traded.
        
        Args:
            signal: Trading signal
            context: "entry" or "exit"
            
        Returns:
            Tuple of (allowed, rejection_reason)
        """
        gate_cfg = self.config.gate.entry if context == "entry" else self.config.gate.exit
        
        # Check if halted (loss limits, circuit breakers)
        if gate_cfg.enforce_loss_limits and self._is_halted:
            return False, f"Trading halted: {self._halt_reason}"
        
        # Reset daily counters if new day
        today = self._current_trading_date()
        if self._current_date != today:
            self.reset_daily()
        
        if gate_cfg.enforce_regime_filter:
            # Check market phase
            if signal.market_phase == MarketPhase.FROZEN:
                return False, RiskEvent.FROZEN_MARKET
            
            # Check entropy (optional additional filter)
            if signal.entropy > self.config.max_entropy:
                logger.debug(
                    "High entropy filter",
                    entropy=signal.entropy,
                    threshold=self.config.max_entropy,
                )
                return False, RiskEvent.HIGH_ENTROPY
            
            # Check temperature (avoid frozen markets)
            if signal.temperature < self.config.min_temperature:
                logger.debug(
                    "Low temperature filter",
                    temperature=signal.temperature,
                    threshold=self.config.min_temperature,
                )
                return False, RiskEvent.FROZEN_MARKET
        
        return True, None
    
    def check_position_size(self, requested_lots: float, context: str = "entry") -> float:
        """
        Validate and adjust position size.
        
        Args:
            requested_lots: Requested position size
            context: "entry" or "exit"
            
        Returns:
            Approved position size (may be reduced)
        """
        gate_cfg = self.config.gate.entry if context == "entry" else self.config.gate.exit
        if not gate_cfg.enforce_position_limits:
            return requested_lots
        
        # Apply maximum position limit
        approved = min(requested_lots, self.config.max_position_lots)
        
        if approved < requested_lots:
            logger.info(
                "Position size reduced",
                requested=requested_lots,
                approved=approved,
                limit=self.config.max_position_lots,
            )
        
        return approved
    
    def record_trade_result(self, result: OrderResult, pnl: float) -> None:
        """
        Record a completed trade for risk tracking.
        
        Args:
            result: Order execution result
            pnl: Realized P&L from this trade
        """
        if result.status != OrderStatus.FILLED:
            return
        
        self._total_trades += 1
        self._daily_pnl += pnl
        
        if pnl >= 0:
            self._winning_trades += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        
        # Check daily loss limit
        daily_limit = self.config.max_position_usd * (self.config.max_daily_loss_pct / 100)
        
        if self._daily_pnl < -daily_limit:
            self._is_halted = True
            self._halt_reason = RiskEvent.DAILY_LOSS_LIMIT
            logger.warning(
                "Daily loss limit reached",
                daily_pnl=self._daily_pnl,
                limit=-daily_limit,
            )
        
        # Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._is_halted = True
            self._halt_reason = RiskEvent.CONSECUTIVE_LOSSES
            logger.warning(
                "Consecutive loss limit reached",
                consecutive_losses=self._consecutive_losses,
                limit=self.config.max_consecutive_losses,
            )
    
    def resume_trading(self, reason: str = "manual") -> None:
        """
        Resume trading after halt.
        
        Args:
            reason: Reason for resuming
        """
        if not self._is_halted:
            return
        
        logger.info(
            "Trading resumed",
            previous_halt=self._halt_reason,
            resume_reason=reason,
        )
        
        self._is_halted = False
        self._halt_reason = None
        self._consecutive_losses = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get risk manager statistics."""
        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": (
                self._winning_trades / max(1, self._total_trades)
            ),
            "current_exposure": self._current_exposure,
        }
    
    def update_exposure(self, exposure_usd: float) -> None:
        """Update current market exposure."""
        self._current_exposure = exposure_usd
        
        if exposure_usd > self.config.max_position_usd:
            logger.warning(
                "Exposure exceeds limit",
                exposure=exposure_usd,
                limit=self.config.max_position_usd,
            )

    def _current_trading_date(self) -> date:
        """Get the current trading date using configured timezone/cutoff."""
        tzinfo = ZoneInfo(self.config.timezone)
        now = datetime.now(tzinfo)
        cutoff = self.config.trading_day_cutoff_hour
        if cutoff:
            now = now - timedelta(hours=cutoff)
        return now.date()
