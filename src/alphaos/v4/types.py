"""
v4.0 Core Types

定义 v4 架构使用的基础类型：
- Bar: OHLCV K线结构
- Timeframe: 时间框架枚举
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Timeframe(Enum):
    """Supported timeframes."""
    M1 = 60       # 1 minute in seconds
    M5 = 300      # 5 minutes
    M15 = 900     # 15 minutes
    M30 = 1800    # 30 minutes
    H1 = 3600     # 1 hour
    H4 = 14400    # 4 hours
    D1 = 86400    # 1 day


@dataclass
class Bar:
    """
    OHLCV bar structure.
    
    All prices are mid prices (average of bid/ask).
    """
    time: datetime  # Bar open time (UTC)
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = 0  # Number of ticks in bar
    spread_sum: float = 0.0  # Sum of spreads for averaging
    
    @property
    def range_pct(self) -> float:
        """Bar range as percentage of close."""
        if self.close <= 0:
            return 0.0
        return (self.high - self.low) / self.close * 100
    
    @property
    def body_pct(self) -> float:
        """Bar body as percentage of close."""
        if self.close <= 0:
            return 0.0
        return (self.close - self.open) / self.close * 100
    
    @property
    def is_bullish(self) -> bool:
        """True if bar closed higher than open."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """True if bar closed lower than open."""
        return self.close < self.open
    
    @property
    def avg_spread(self) -> float:
        """Average spread during bar."""
        if self.tick_volume <= 0:
            return 0.0
        return self.spread_sum / self.tick_volume
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "time": self.time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tick_volume": self.tick_volume,
            "avg_spread": self.avg_spread,
        }
