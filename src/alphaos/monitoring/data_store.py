from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from alphaos.core.config import DatabaseConfig, DataStoreConfig

logger = logging.getLogger(__name__)


class DataStore:
    """
    Unified async data store for ticks/bars/decisions/orders/fills/positions.
    Uses simple in-memory buffers with periodic batch flush.
    """

    def __init__(self, config: DatabaseConfig, symbol: str, settings: DataStoreConfig):
        self.config = config
        self.symbol = symbol
        self.settings = settings
        self.engine = create_async_engine(config.connection_string, echo=False)
        self._buffers: dict[str, list[dict[str, Any]]] = {
            "ticks": [],
            "bars": [],
            "decisions": [],
            "orders": [],
            "fills": [],
            "positions": [],
        }
        self._running = False
        self._flush_task: asyncio.Task | None = None
        self._dropped: dict[str, int] = {k: 0 for k in self._buffers.keys()}

    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS raw_ticks (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bid DOUBLE PRECISION,
                    ask DOUBLE PRECISION,
                    mid DOUBLE PRECISION,
                    spread DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    flags INTEGER
                );
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS bars (
                    time_open TIMESTAMPTZ NOT NULL,
                    time_close TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bar_idx INTEGER,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    tick_count INTEGER,
                    duration_ms DOUBLE PRECISION,
                    spread_avg DOUBLE PRECISION,
                    imbalance INTEGER,
                    imbalance_ratio DOUBLE PRECISION,
                    buy_count INTEGER,
                    sell_count INTEGER,
                    neutral_count INTEGER
                );
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS inference_decisions (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bar_idx INTEGER,
                    has_signal BOOLEAN,
                    direction INTEGER,
                    entry_price DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION,
                    meta_confidence DOUBLE PRECISION,
                    should_trade BOOLEAN,
                    filtered_reason TEXT,
                    market_phase TEXT,
                    market_temperature DOUBLE PRECISION,
                    market_entropy DOUBLE PRECISION,
                    trend_duration INTEGER,
                    st_trend_15m INTEGER,
                    fvg_event INTEGER,
                    trend_direction INTEGER,
                    model_version TEXT,
                    config_hash TEXT
                );
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS orders (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    request_id TEXT,
                    magic BIGINT,
                    action TEXT,
                    volume DOUBLE PRECISION,
                    price DOUBLE PRECISION,
                    sl DOUBLE PRECISION,
                    tp DOUBLE PRECISION,
                    deviation INTEGER,
                    comment TEXT,
                    context TEXT,
                    status TEXT,
                    error_code INTEGER,
                    error_message TEXT
                );
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fills (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    request_id TEXT,
                    magic BIGINT,
                    ticket BIGINT,
                    action TEXT,
                    volume_filled DOUBLE PRECISION,
                    price_filled DOUBLE PRECISION,
                    status TEXT,
                    error_code INTEGER,
                    error_message TEXT,
                    latency_ms DOUBLE PRECISION
                );
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS positions_state (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    ticket BIGINT,
                    direction TEXT,
                    entry_price DOUBLE PRECISION,
                    current_price DOUBLE PRECISION,
                    current_lots DOUBLE PRECISION,
                    unrealized_pnl DOUBLE PRECISION,
                    net_pnl DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION,
                    take_profit DOUBLE PRECISION,
                    stage TEXT,
                    trend_alignment TEXT,
                    market_phase TEXT
                );
            """))

            try:
                await conn.execute(text("SELECT create_hypertable('raw_ticks', 'time', if_not_exists => TRUE);"))
                await conn.execute(text("SELECT create_hypertable('bars', 'time_close', if_not_exists => TRUE);"))
                await conn.execute(text("SELECT create_hypertable('inference_decisions', 'time', if_not_exists => TRUE);"))
                await conn.execute(text("SELECT create_hypertable('orders', 'time', if_not_exists => TRUE);"))
                await conn.execute(text("SELECT create_hypertable('fills', 'time', if_not_exists => TRUE);"))
                await conn.execute(text("SELECT create_hypertable('positions_state', 'time', if_not_exists => TRUE);"))
            except Exception as e:
                logger.debug("Hypertable creation info: %s", e)

        logger.info("DataStore initialized")

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def close(self) -> None:
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(Exception):
                await self._flush_task
        await self.flush_all()
        await self.engine.dispose()

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.settings.flush_interval_sec)
            await self.flush_all()

    def _enqueue(self, key: str, item: dict[str, Any]) -> None:
        if len(self._buffers[key]) >= self.settings.max_queue_size:
            self._dropped[key] += 1
            return
        self._buffers[key].append(item)
        if len(self._buffers[key]) >= self.settings.batch_size:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush_all())
            except RuntimeError:
                pass

    async def flush_all(self) -> None:
        pending: dict[str, list[dict[str, Any]]] = {}
        for key, buf in self._buffers.items():
            if buf:
                pending[key] = buf.copy()
                self._buffers[key].clear()
        if not pending:
            return

        async with self.engine.begin() as conn:
            if pending.get("ticks"):
                await conn.execute(
                    text("""
                        INSERT INTO raw_ticks (
                            time, symbol, bid, ask, mid, spread, volume, flags
                        ) VALUES (
                            to_timestamp(:ts), :symbol, :bid, :ask, :mid, :spread, :volume, :flags
                        )
                    """),
                    pending["ticks"],
                )
            if pending.get("bars"):
                await conn.execute(
                    text("""
                        INSERT INTO bars (
                            time_open, time_close, symbol, bar_idx,
                            open, high, low, close, tick_count,
                            duration_ms, spread_avg, imbalance, imbalance_ratio,
                            buy_count, sell_count, neutral_count
                        ) VALUES (
                            to_timestamp(:ts_open), to_timestamp(:ts_close), :symbol, :bar_idx,
                            :open, :high, :low, :close, :tick_count,
                            :duration_ms, :spread_avg, :imbalance, :imbalance_ratio,
                            :buy_count, :sell_count, :neutral_count
                        )
                    """),
                    pending["bars"],
                )
            if pending.get("decisions"):
                await conn.execute(
                    text("""
                        INSERT INTO inference_decisions (
                            time, symbol, bar_idx, has_signal, direction, entry_price, stop_loss,
                            meta_confidence, should_trade, filtered_reason, market_phase,
                            market_temperature, market_entropy, trend_duration,
                            st_trend_15m, fvg_event, trend_direction, model_version, config_hash
                        ) VALUES (
                            to_timestamp(:ts), :symbol, :bar_idx, :has_signal, :direction, :entry_price, :stop_loss,
                            :meta_confidence, :should_trade, :filtered_reason, :market_phase,
                            :market_temperature, :market_entropy, :trend_duration,
                            :st_trend_15m, :fvg_event, :trend_direction, :model_version, :config_hash
                        )
                    """),
                    pending["decisions"],
                )
            if pending.get("orders"):
                await conn.execute(
                    text("""
                        INSERT INTO orders (
                            time, symbol, request_id, magic, action, volume, price, sl, tp,
                            deviation, comment, context, status, error_code, error_message
                        ) VALUES (
                            to_timestamp(:ts), :symbol, :request_id, :magic, :action, :volume, :price, :sl, :tp,
                            :deviation, :comment, :context, :status, :error_code, :error_message
                        )
                    """),
                    pending["orders"],
                )
            if pending.get("fills"):
                await conn.execute(
                    text("""
                        INSERT INTO fills (
                            time, symbol, request_id, magic, ticket, action, volume_filled, price_filled,
                            status, error_code, error_message, latency_ms
                        ) VALUES (
                            to_timestamp(:ts), :symbol, :request_id, :magic, :ticket, :action, :volume_filled, :price_filled,
                            :status, :error_code, :error_message, :latency_ms
                        )
                    """),
                    pending["fills"],
                )
            if pending.get("positions"):
                await conn.execute(
                    text("""
                        INSERT INTO positions_state (
                            time, symbol, ticket, direction, entry_price, current_price, current_lots,
                            unrealized_pnl, net_pnl, stop_loss, take_profit, stage, trend_alignment, market_phase
                        ) VALUES (
                            to_timestamp(:ts), :symbol, :ticket, :direction, :entry_price, :current_price, :current_lots,
                            :unrealized_pnl, :net_pnl, :stop_loss, :take_profit, :stage, :trend_alignment, :market_phase
                        )
                    """),
                    pending["positions"],
                )

    def enqueue_tick(self, ts: float, bid: float, ask: float, volume: float, flags: int) -> None:
        if not self.settings.enable_ticks:
            return
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread = ask - bid if bid > 0 and ask > 0 else 0.0
        self._enqueue("ticks", {
            "ts": ts,
            "symbol": self.symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "volume": volume,
            "flags": flags,
        })

    def enqueue_bar(self, bar: Any, bar_idx: int) -> None:
        if not self.settings.enable_bars:
            return
        self._enqueue("bars", {
            "ts_open": bar.time.timestamp(),
            "ts_close": bar.close_time.timestamp(),
            "symbol": self.symbol,
            "bar_idx": bar_idx,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "tick_count": bar.tick_count,
            "duration_ms": bar.duration_ms,
            "spread_avg": bar.avg_spread,
            "imbalance": bar.imbalance,
            "imbalance_ratio": bar.imbalance_ratio,
            "buy_count": bar.buy_count,
            "sell_count": bar.sell_count,
            "neutral_count": bar.neutral_count,
        })

    def enqueue_decision(self, ts: float, result: Any, model_version: str, config_hash: str) -> None:
        if not self.settings.enable_decisions:
            return
        self._enqueue("decisions", {
            "ts": ts,
            "symbol": self.symbol,
            "bar_idx": result.bar_idx,
            "has_signal": result.has_signal,
            "direction": result.direction,
            "entry_price": result.entry_price,
            "stop_loss": result.stop_loss,
            "meta_confidence": result.meta_confidence,
            "should_trade": result.should_trade,
            "filtered_reason": result.filtered_reason,
            "market_phase": result.market_phase,
            "market_temperature": result.market_temperature,
            "market_entropy": result.market_entropy,
            "trend_duration": result.trend_duration,
            "st_trend_15m": result.st_trend_15m,
            "fvg_event": result.fvg_event,
            "trend_direction": result.trend_direction,
            "model_version": model_version,
            "config_hash": config_hash,
        })

    def enqueue_order(self, ts: float, order: Any, context: str, status: str, error_code: int = 0, error_message: str = "") -> None:
        if not self.settings.enable_orders:
            return
        self._enqueue("orders", {
            "ts": ts,
            "symbol": self.symbol,
            "request_id": str(order.magic),
            "magic": order.magic,
            "action": order.action.name if hasattr(order.action, "name") else str(order.action),
            "volume": order.volume,
            "price": order.price,
            "sl": order.sl,
            "tp": order.tp,
            "deviation": order.deviation,
            "comment": order.comment,
            "context": context,
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
        })

    def enqueue_fill(self, ts: float, order: Any, result: Any, latency_ms: float) -> None:
        if not self.settings.enable_fills:
            return
        self._enqueue("fills", {
            "ts": ts,
            "symbol": self.symbol,
            "request_id": str(order.magic),
            "magic": order.magic,
            "ticket": result.ticket,
            "action": order.action.name if hasattr(order.action, "name") else str(order.action),
            "volume_filled": result.volume_filled,
            "price_filled": result.price_filled,
            "status": result.status.name if hasattr(result.status, "name") else str(result.status),
            "error_code": result.error_code,
            "error_message": result.error_message,
            "latency_ms": latency_ms,
        })

    def enqueue_positions(self, ts: float, positions: Iterable[dict[str, Any]]) -> None:
        if not self.settings.enable_positions:
            return
        for pos in positions:
            self._enqueue("positions", {
                "ts": ts,
                "symbol": self.symbol,
                "ticket": pos.get("ticket", 0),
                "direction": pos.get("direction"),
                "entry_price": pos.get("entry_price", 0.0),
                "current_price": pos.get("current_price", 0.0),
                "current_lots": pos.get("current_lots", 0.0),
                "unrealized_pnl": pos.get("unrealized_pnl", 0.0),
                "net_pnl": pos.get("net_pnl", 0.0),
                "stop_loss": pos.get("stop_loss", 0.0),
                "take_profit": pos.get("take_profit", 0.0),
                "stage": pos.get("stage"),
                "trend_alignment": pos.get("trend_alignment"),
                "market_phase": pos.get("market_phase"),
            })
