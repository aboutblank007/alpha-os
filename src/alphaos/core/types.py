"""
AlphaOS Core Type Definitions

Defines fundamental data structures used throughout the system:
- Tick: Raw market data
- Signal: Trading signal
- Order: Order request/response
- Position: Current position state
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import NamedTuple
import struct


# ============================================================================
# Enumerations
# ============================================================================

class TickFlag(IntEnum):
    """Flags indicating tick properties."""
    BID_CHANGED = 1
    ASK_CHANGED = 2
    LAST_CHANGED = 4
    VOLUME_CHANGED = 8
    BUY = 16
    SELL = 32


class SignalType(IntEnum):
    """Trading signal types."""
    NEUTRAL = 0
    LONG = 1
    SHORT = 2


class OrderAction(IntEnum):
    """Order action types."""
    BUY = 0
    SELL = 1
    CLOSE = 2
    MODIFY = 3


class OrderStatus(IntEnum):
    """Order execution status."""
    PENDING = 0
    FILLED = 1
    PARTIALLY_FILLED = 2
    REJECTED = 3
    CANCELLED = 4
    EXPIRED = 5


class MarketPhase(IntEnum):
    """Market phase based on T-S thermodynamics."""
    LAMINAR = 0          # Low T, Low S - Passive limit orders
    TURBULENT = 1        # High T, High S - Mean-reversion, reduce size
    PHASE_TRANSITION = 2 # High T, Low S - Breakout, IOC orders
    FROZEN = 3           # Low T, High S - Dead market, no trading


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(slots=True, frozen=True)
class Tick:
    """
    Immutable tick data structure.
    
    Represents a single L1 market data update from MT5.
    Volume fields may be zero in live trading (Sim2Real issue).
    
    Attributes:
        timestamp_us: Server timestamp in microseconds
        bid: Best bid price
        ask: Best ask price
        bid_volume: Bid volume (may be 0 in live)
        ask_volume: Ask volume (may be 0 in live)
        last: Last trade price
        last_volume: Last trade volume (may be 0 in live)
        flags: Bitfield of TickFlag values
    """
    timestamp_us: int
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    last: float = 0.0
    last_volume: float = 0.0
    flags: int = 0
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def timestamp_s(self) -> float:
        """Timestamp in seconds (float)."""
        return self.timestamp_us / 1_000_000
    
    @property
    def time_msc(self) -> int:
        """Timestamp in milliseconds (for compatibility with MT5 format)."""
        return self.timestamp_us // 1000
    
    @property
    def is_buy(self) -> bool:
        """Check if tick indicates a buy trade."""
        return bool(self.flags & TickFlag.BUY)
    
    @property
    def is_sell(self) -> bool:
        """Check if tick indicates a sell trade."""
        return bool(self.flags & TickFlag.SELL)


@dataclass(slots=True)
class TickFeatures:
    """
    Computed microstructure features for a tick.
    
    All features are designed to work without volume (Sim2Real compatible).
    """
    # Time features
    delta_t: float              # Inter-arrival time (seconds)
    log_delta_t: float          # ln(delta_t + epsilon)
    
    # Price features (relative units)
    dp: float                   # Percentage mid-price change (dp_pct)
    dp_bid: float               # Reserved (unused)
    dp_ask: float               # Reserved (unused)
    spread: float               # Spread in basis points (spread_bps)
    
    # Intensity features (Ghost Volume Proxies)
    tick_intensity: float       # Lambda_t (EWMA of 1/Δt), units: Hz
    
    # Direction features
    tick_direction: int         # Bayesian tick rule: -1, 0, +1
    ofi_count: float            # Volume-blind OFI
    pdi: float                  # Price-Driven Imbalance
    
    # Microstructure features
    kyle_lambda: float          # Micro Kyle's Lambda (percentage)
    
    # Rolling window High/Low (for Alpha191)
    rolling_high: float         # Window max price (mid)
    rolling_low: float          # Window min price (mid)
    
    # Thermodynamic features
    temperature: float          # Market temperature T (normalized 0-1)
    entropy: float              # Market entropy S (normalized 0-1)
    market_phase: MarketPhase   # Phase from T-S diagram
    
    # Trend features (v4.0 - per 交易模型研究.md Section 5.2)
    trend_deviation: float = 0.0  # (Close - SuperTrend_Line) / ATR


@dataclass(slots=True)
class Signal:
    """
    Trading signal from the model.
    
    Attributes:
        timestamp_us: When the signal was generated
        signal_type: LONG, SHORT, or NEUTRAL
        confidence: Probability/confidence of the signal [0, 1]
        temperature: Market temperature at signal time
        entropy: Market entropy at signal time
        market_phase: Current market phase
        features: Optional feature snapshot for logging
        probabilities: Optional (loss_prob, neutral_prob, win_prob) tuple
    """
    timestamp_us: int
    signal_type: SignalType
    confidence: float
    temperature: float
    entropy: float
    market_phase: MarketPhase
    features: TickFeatures | None = None
    probabilities: tuple[float, float, float] | None = None  # (loss, neutral, win)


@dataclass(slots=True)
class Order:
    """
    Order to be sent to MT5.
    
    Attributes:
        magic: Unique order identifier
        action: BUY, SELL, CLOSE, or MODIFY
        symbol: Trading symbol (e.g., XAUUSD)
        volume: Order size in lots
        price: Order price (0 for market orders)
        sl: Stop loss price
        tp: Take profit price
        deviation: Maximum allowed slippage in points
        comment: Order comment for logging
        ticket: Optional MT5 position ticket for CLOSE/MODIFY
    """
    magic: int
    action: OrderAction
    symbol: str
    volume: float
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 10
    comment: str = ""
    ticket: int | None = None


@dataclass(slots=True)
class OrderResult:
    """
    Result of an order execution from MT5.
    """
    magic: int
    ticket: int
    status: OrderStatus
    volume_filled: float
    price_filled: float
    error_code: int = 0
    error_message: str = ""


@dataclass(slots=True)
class Position:
    """
    Current position state.
    """
    symbol: str
    ticket: int
    direction: SignalType  # LONG or SHORT
    volume: float
    entry_price: float
    open_time_us: int
    current_price: float
    unrealized_pnl: float
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0


# ============================================================================
# Binary Protocol Structures (for ZeroMQ)
# ============================================================================

class TickPacket(NamedTuple):
    """
    Binary packet for tick data (MT5 -> Python).
    
    Format matches EA BinaryTick struct: '<ddqqi' (36 bytes)
    - bid: double (8 bytes)
    - ask: double (8 bytes)
    - time_msc: int64 (8 bytes) - milliseconds since epoch
    - volume: int64 (8 bytes)
    - flags: int32 (4 bytes)
    
    Note: No msg_type prefix - PUB/SUB only used for ticks.
    """
    bid: float
    ask: float
    time_msc: int
    volume: int
    flags: int
    
    STRUCT_FORMAT = '<ddqqi'
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "TickPacket":
        """Deserialize from binary data."""
        return cls(*struct.unpack(cls.STRUCT_FORMAT, data[:cls.STRUCT_SIZE]))
    
    def to_tick(self) -> Tick:
        """Convert to Tick object."""
        return Tick(
            timestamp_us=self.time_msc * 1000,  # Convert ms to us
            bid=self.bid,
            ask=self.ask,
            bid_volume=0.0,  # Not in EA packet
            ask_volume=0.0,  # Not in EA packet
            last=0.0,  # Not in EA packet
            last_volume=float(self.volume),
            flags=self.flags,
        )


class OrderPacket(NamedTuple):
    """
    Binary packet for orders (Python -> MT5).
    
    Format: '<BBddddQQ' (50 bytes)
    - msg_type: uint8 (2 = order)
    - action: uint8 (0=buy, 1=sell, 2=close)
    - volume: double
    - price: double
    - sl: double
    - tp: double
    - magic: uint64
    - ticket: uint64
    """
    msg_type: int
    action: int
    volume: float
    price: float
    sl: float
    tp: float
    magic: int
    ticket: int
    
    STRUCT_FORMAT = '<BBddddQQ'
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)
    
    def to_bytes(self) -> bytes:
        """Serialize to binary data."""
        return struct.pack(
            self.STRUCT_FORMAT,
            self.msg_type,
            self.action,
            self.volume,
            self.price,
            self.sl,
            self.tp,
            self.magic,
            self.ticket,
        )
    
    @classmethod
    def from_order(cls, order: Order) -> "OrderPacket":
        """Create packet from Order object."""
        return cls(
            msg_type=2,
            action=order.action.value,
            volume=order.volume,
            price=order.price,
            sl=order.sl,
            tp=order.tp,
            magic=order.magic,
            ticket=order.ticket or 0,
        )
