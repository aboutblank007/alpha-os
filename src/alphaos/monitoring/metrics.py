"""
AlphaOS Prometheus Metrics

Exposes operational metrics for monitoring:
- Inference latency
- Signal generation
- Order execution
- Risk state
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
)

from alphaos.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Metric Definitions
# ============================================================================

# Info metrics
SYSTEM_INFO = Info(
    "alphaos_system",
    "AlphaOS system information",
)

# Counters
TICK_COUNTER = Counter(
    "alphaos_ticks_total",
    "Total ticks processed",
    ["symbol"],
)

SIGNAL_COUNTER = Counter(
    "alphaos_signals_total",
    "Total signals generated",
    ["symbol", "signal_type"],
)

ORDER_COUNTER = Counter(
    "alphaos_orders_total",
    "Total orders sent",
    ["symbol", "action", "status"],
)

RISK_EVENT_COUNTER = Counter(
    "alphaos_risk_events_total",
    "Risk events triggered",
    ["event_type"],
)

# Gauges
WARMUP_PROGRESS = Gauge(
    "alphaos_warmup_progress",
    "Warmup progress (0-1)",
    ["symbol"],
)

CURRENT_POSITION = Gauge(
    "alphaos_position_lots",
    "Current position size in lots",
    ["symbol", "direction"],
)

DAILY_PNL = Gauge(
    "alphaos_daily_pnl_usd",
    "Daily P&L in USD",
    ["symbol"],
)

MARKET_TEMPERATURE = Gauge(
    "alphaos_market_temperature",
    "Current market temperature",
    ["symbol"],
)

MARKET_ENTROPY = Gauge(
    "alphaos_market_entropy",
    "Current market entropy",
    ["symbol"],
)

MODEL_CONFIDENCE = Gauge(
    "alphaos_model_confidence",
    "Latest model prediction confidence",
    ["symbol"],
)

# 直方图（延迟桶可配置）
_DEFAULT_INFERENCE_LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
_DEFAULT_TICK_PROCESSING_LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
_DEFAULT_ORDER_LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

INFERENCE_LATENCY: Histogram | None = None
TICK_PROCESSING_LATENCY: Histogram | None = None
ORDER_LATENCY: Histogram | None = None
_METRICS_INITIALIZED = False


def _ensure_histograms(metrics_config: dict | None) -> None:
    """按配置初始化直方图指标（只初始化一次）"""
    global INFERENCE_LATENCY, TICK_PROCESSING_LATENCY, ORDER_LATENCY, _METRICS_INITIALIZED
    if _METRICS_INITIALIZED:
        return
    cfg = metrics_config or {}
    inference_buckets = cfg.get("inference_latency_buckets", _DEFAULT_INFERENCE_LATENCY_BUCKETS)
    tick_buckets = cfg.get("tick_processing_latency_buckets", _DEFAULT_TICK_PROCESSING_LATENCY_BUCKETS)
    order_buckets = cfg.get("order_latency_buckets", _DEFAULT_ORDER_LATENCY_BUCKETS)

    INFERENCE_LATENCY = Histogram(
        "alphaos_inference_latency_seconds",
        "Model inference latency",
        ["symbol"],
        buckets=[float(x) for x in inference_buckets],
    )
    TICK_PROCESSING_LATENCY = Histogram(
        "alphaos_tick_processing_latency_seconds",
        "Full tick processing latency",
        ["symbol"],
        buckets=[float(x) for x in tick_buckets],
    )
    ORDER_LATENCY = Histogram(
        "alphaos_order_latency_seconds",
        "Order execution latency",
        ["symbol"],
        buckets=[float(x) for x in order_buckets],
    )
    _METRICS_INITIALIZED = True


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Centralized metrics collection for AlphaOS.
    
    Provides convenient methods for recording metrics
    and integrates with the gateway and model.
    """
    
    def __init__(self, symbol: str, version: str = "3.0.0", metrics_config: dict | None = None) -> None:
        """
        Initialize metrics collector.
        
        Args:
            symbol: Trading symbol
            version: AlphaOS version
        """
        self.symbol = symbol
        _ensure_histograms(metrics_config)
        
        # Set system info
        SYSTEM_INFO.info({
            "version": version,
            "symbol": symbol,
        })
        
        # Initialize gauges
        WARMUP_PROGRESS.labels(symbol=symbol).set(0)
        CURRENT_POSITION.labels(symbol=symbol, direction="long").set(0)
        CURRENT_POSITION.labels(symbol=symbol, direction="short").set(0)
        DAILY_PNL.labels(symbol=symbol).set(0)
    
    def record_tick(self) -> None:
        """Record a tick processed."""
        TICK_COUNTER.labels(symbol=self.symbol).inc()
    
    def record_signal(self, signal_type: str, confidence: float) -> None:
        """
        Record a generated signal.
        
        Args:
            signal_type: "long", "short", or "neutral"
            confidence: Prediction confidence
        """
        SIGNAL_COUNTER.labels(
            symbol=self.symbol,
            signal_type=signal_type,
        ).inc()
        
        MODEL_CONFIDENCE.labels(symbol=self.symbol).set(confidence)
    
    def record_order(self, action: str, status: str) -> None:
        """
        Record an order execution.
        
        Args:
            action: "buy", "sell", "close"
            status: "filled", "rejected", etc.
        """
        ORDER_COUNTER.labels(
            symbol=self.symbol,
            action=action,
            status=status,
        ).inc()
    
    def record_risk_event(self, event_type: str) -> None:
        """
        Record a risk event.
        
        Args:
            event_type: Type of risk event
        """
        RISK_EVENT_COUNTER.labels(event_type=event_type).inc()
    
    def update_warmup(self, progress: float) -> None:
        """
        Update warmup progress.
        
        Args:
            progress: Progress from 0 to 1
        """
        WARMUP_PROGRESS.labels(symbol=self.symbol).set(progress)
    
    def update_position(self, direction: str, lots: float) -> None:
        """
        Update current position.
        
        Args:
            direction: "long" or "short"
            lots: Position size
        """
        CURRENT_POSITION.labels(
            symbol=self.symbol,
            direction=direction,
        ).set(lots)
        
        # Clear opposite direction
        opposite = "short" if direction == "long" else "long"
        CURRENT_POSITION.labels(
            symbol=self.symbol,
            direction=opposite,
        ).set(0)
    
    def clear_position(self) -> None:
        """Clear all position gauges."""
        CURRENT_POSITION.labels(symbol=self.symbol, direction="long").set(0)
        CURRENT_POSITION.labels(symbol=self.symbol, direction="short").set(0)
    
    def update_pnl(self, pnl: float) -> None:
        """Update daily P&L."""
        DAILY_PNL.labels(symbol=self.symbol).set(pnl)
    
    def update_thermodynamics(self, temperature: float, entropy: float) -> None:
        """
        Update market thermodynamic state.
        
        Args:
            temperature: Market temperature
            entropy: Market entropy
        """
        MARKET_TEMPERATURE.labels(symbol=self.symbol).set(temperature)
        MARKET_ENTROPY.labels(symbol=self.symbol).set(entropy)
    
    @contextmanager
    def measure_inference(self) -> Generator[None, None, None]:
        """Context manager for measuring inference latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            INFERENCE_LATENCY.labels(symbol=self.symbol).observe(duration)
    
    @contextmanager
    def measure_tick_processing(self) -> Generator[None, None, None]:
        """Context manager for measuring full tick processing."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            TICK_PROCESSING_LATENCY.labels(symbol=self.symbol).observe(duration)
    
    @contextmanager
    def measure_order(self) -> Generator[None, None, None]:
        """Context manager for measuring order latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            ORDER_LATENCY.labels(symbol=self.symbol).observe(duration)


# ============================================================================
# Server Setup
# ============================================================================

def setup_metrics_server(port: int = 9090, metrics_config: dict | None = None) -> None:
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        port: HTTP port for metrics endpoint
    """
    try:
        _ensure_histograms(metrics_config)
        start_http_server(port)
        logger.info("Prometheus metrics server started", port=port)
    except Exception as e:
        logger.error("Failed to start metrics server", error=str(e))
        raise
