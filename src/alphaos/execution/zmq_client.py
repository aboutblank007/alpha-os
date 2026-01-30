"""
AlphaOS ZeroMQ Client (v4)

Handles binary communication with MT5 Expert Advisor:
- Tick subscription (SUB socket)
- Order sending (DEALER/ROUTER)
- Position queries
- Heartbeat monitoring
- Automatic reconnection
"""

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, NamedTuple

import zmq
import zmq.asyncio

from alphaos.core.config import ZeroMQConfig
from alphaos.core.logging import get_logger
from alphaos.core.types import (
    Tick,
    TickPacket,
    Order,
    OrderPacket,
    OrderResult,
    OrderStatus,
)

logger = get_logger(__name__)


# ============================================================================
# Message Types
# ============================================================================

class MessageType:
    """ZeroMQ message type identifiers."""
    TICK = 1
    ORDER = 2
    ORDER_RESULT = 3
    HEARTBEAT = 4
    POSITION_UPDATE = 5
    GET_POSITIONS = 6      # Request positions from MT5
    POSITIONS_RESPONSE = 7  # Positions response from MT5
    ERROR = 255


# ============================================================================
# Order Result Packet
# ============================================================================

@dataclass
class OrderResultPacket:
    """
    Binary packet for order results (MT5 -> Python).
    
    Format: '<BQQBddiI' (42 bytes)
    """
    msg_type: int
    magic: int
    ticket: int
    status: int
    volume_filled: float
    price_filled: float
    error_code: int
    reserved: int
    
    STRUCT_FORMAT = '<BQQBddiI'
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "OrderResultPacket":
        """Deserialize from binary data."""
        return cls(*struct.unpack(cls.STRUCT_FORMAT, data[:cls.STRUCT_SIZE]))
    
    def to_order_result(self) -> OrderResult:
        """Convert to OrderResult object."""
        return OrderResult(
            magic=self.magic,
            ticket=self.ticket,
            status=OrderStatus(self.status),
            volume_filled=self.volume_filled,
            price_filled=self.price_filled,
            error_code=self.error_code,
        )


# ============================================================================
# ZeroMQ Client
# ============================================================================

TickCallback = Callable[[Tick], Awaitable[None]]


class ZeroMQClient:
    """
    Async ZeroMQ client for MT5 communication.
    
    Uses two sockets:
    - SUB: Receive tick stream from MT5
    - REQ: Send orders to MT5
    
    Binary protocol for minimal latency (no JSON).
    """
    
    def __init__(self, config: ZeroMQConfig) -> None:
        """
        Initialize ZeroMQ client.
        
        Args:
            config: ZeroMQ configuration
        """
        self.config = config
        
        self._context: zmq.asyncio.Context | None = None
        self._tick_socket: zmq.asyncio.Socket | None = None
        self._order_socket: zmq.asyncio.Socket | None = None
        self._history_socket: zmq.asyncio.Socket | None = None  # For GET_HISTORY
        
        self._running = False
        self._connected = False
        self._tick_callbacks: list[TickCallback] = []
        
        # Stats
        self._tick_count = 0
        self._order_count = 0
        self._last_tick_time: float = 0
        self._last_heartbeat_time: float = 0
        
        # Order tracking
        self._pending_orders: dict[int, asyncio.Future[OrderResult]] = {}
        self._next_magic = int(time.time() * 1000) % 1000000000
    
    # ========================================================================
    # Connection Management
    # ========================================================================
    
    async def connect(self) -> None:
        """Connect to MT5 ZeroMQ endpoints."""
        logger.info("Connecting to MT5")
        
        self._context = zmq.asyncio.Context()
        
        # Tick subscription socket
        self._tick_socket = self._context.socket(zmq.SUB)
        self._tick_socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self._tick_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._tick_socket.setsockopt(zmq.LINGER, 0)
        self._tick_socket.connect(self.config.tick_endpoint)
        
        # 订单 socket（DEALER/ROUTER 模式，兼容 MT5 EA）
        # DEALER 比 REQ 更适合与 ROUTER 配对
        self._order_socket = self._context.socket(zmq.DEALER)
        self._order_socket.setsockopt(zmq.RCVTIMEO, self.config.order_recv_timeout_ms)
        self._order_socket.setsockopt(zmq.SNDTIMEO, self.config.order_snd_timeout_ms)
        self._order_socket.setsockopt(zmq.LINGER, 0)
        # Set identity for DEALER (so ROUTER can route responses back)
        import os
        import socket
        dealer_id = f"AlphaOS-{socket.gethostname()[:8]}-{os.getpid()}".encode("utf-8")
        self._order_socket.setsockopt(zmq.IDENTITY, dealer_id)
        self._order_socket.connect(self.config.order_endpoint)
        
        # History data socket (REQ pattern for GET_HISTORY requests)
        history_timeout = self.config.history_timeout_ms
        self._history_socket = self._context.socket(zmq.REQ)
        self._history_socket.setsockopt(zmq.RCVTIMEO, history_timeout)
        self._history_socket.setsockopt(zmq.SNDTIMEO, self.config.history_snd_timeout_ms)
        self._history_socket.setsockopt(zmq.LINGER, 0)
        history_endpoint = self.config.history_endpoint
        self._history_socket.connect(history_endpoint)
        
        self._connected = True
        
        logger.info(
            "Connected to MT5",
            tick_endpoint=self.config.tick_endpoint,
            order_endpoint=self.config.order_endpoint,
            history_endpoint=history_endpoint,
        )
    
    async def disconnect(self) -> None:
        """Disconnect from MT5."""
        self._running = False
        self._connected = False
        
        if self._tick_socket:
            self._tick_socket.close()
            self._tick_socket = None
        
        if self._order_socket:
            self._order_socket.close()
            self._order_socket = None
        
        # Close history socket
        if self._history_socket:
            self._history_socket.close()
            self._history_socket = None
        
        if self._context:
            self._context.term()
            self._context = None
        
        logger.info("Disconnected from MT5")
    
    async def reconnect(self) -> None:
        """Reconnect to MT5 after connection loss."""
        logger.warning("Reconnecting to MT5")
        
        await self.disconnect()
        await asyncio.sleep(self.config.reconnect_delay_ms / 1000)
        await self.connect()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self._connected
    
    # ========================================================================
    # Tick Stream
    # ========================================================================
    
    def add_tick_callback(self, callback: TickCallback) -> None:
        """Register a callback for incoming ticks."""
        self._tick_callbacks.append(callback)
    
    def remove_tick_callback(self, callback: TickCallback) -> None:
        """Remove a tick callback."""
        self._tick_callbacks.remove(callback)
    
    async def start_tick_stream(self) -> None:
        """Start receiving tick stream (blocking)."""
        if not self._connected:
            await self.connect()
        
        self._running = True
        logger.info("Starting tick stream")
        
        while self._running:
            try:
                await self._receive_tick()
            except zmq.ZMQError as e:
                if self._running:
                    logger.error("ZMQ error in tick stream", error=str(e))
                    await self.reconnect()
    
    async def _receive_tick(self) -> None:
        """Receive and process a single tick."""
        try:
            data = await self._tick_socket.recv()
            
            # EA sends 36-byte binary tick without msg_type prefix
            # Format: <ddqqi (bid, ask, time_msc, volume, flags)
            if len(data) >= TickPacket.STRUCT_SIZE:
                packet = TickPacket.from_bytes(data)
                tick = packet.to_tick()
                
                self._tick_count += 1
                self._last_tick_time = time.time()
                
                # Notify callbacks
                for callback in self._tick_callbacks:
                    try:
                        await callback(tick)
                    except Exception as e:
                        logger.error("Tick callback error", error=str(e))
            elif len(data) > 0:
                # Log unexpected data for debugging
                logger.warning(
                    "Unexpected tick data size",
                    received=len(data),
                    expected=TickPacket.STRUCT_SIZE,
                )
            
        except zmq.Again:
            # Timeout - check connection health
            await self._check_health()
    
    async def _check_health(self) -> None:
        """Check connection health and reconnect if needed."""
        now = time.time()
        
        # Check for stale connection
        if self._last_tick_time > 0:
            gap = now - self._last_tick_time
            
            if gap > self.config.tick_staleness_threshold_sec:
                logger.warning("Connection appears stale", gap_seconds=gap)
    
    # ========================================================================
    # Order Execution
    # ========================================================================
    
    def get_next_magic(self) -> int:
        """Generate unique order magic number."""
        magic = self._next_magic
        self._next_magic += 1
        return magic
    
    async def send_order(self, order: Order) -> OrderResult:
        """
        Send an order to MT5.
        
        Args:
            order: Order to execute
            
        Returns:
            OrderResult from MT5
        """
        import json
        if not self._connected:
            raise RuntimeError("Not connected to MT5")
        
        # Create JSON order request (compatible with MT5 EA's ROUTER)
        # action mapping: BUY/SELL/CLOSE/etc.
        from alphaos.core.types import OrderAction
        action_map = {
            OrderAction.BUY: "BUY",
            OrderAction.SELL: "SELL",
            OrderAction.CLOSE: "CLOSE",
            OrderAction.MODIFY: "MODIFY",
        }
        
        request_data = {
            "action": action_map.get(order.action, "BUY"),
            "symbol": order.symbol,
            "volume": order.volume,
            "price": order.price,
            "sl": order.sl,
            "tp": order.tp,
            "deviation": order.deviation,
            "magic": order.magic,
            "comment": order.comment,
            "request_id": str(order.magic)
        }
        
        request = json.dumps(request_data).encode('ascii')
        
        logger.debug(
            "Sending order",
            magic=order.magic,
            action=order.action.name,
            volume=order.volume,
        )
        
        try:
            # Send order (DEALER socket: [empty_frame, data])
            await self._order_socket.send_multipart([b"", request])
            self._order_count += 1
            
            # Wait for response (DEALER socket: [empty_frame, response])
            frames = await self._order_socket.recv_multipart()
            response = frames[-1] if frames else b""
            
            # Try to parse JSON response from EA
            try:
                res_data = json.loads(response.decode('utf-8'))
                
                # Map EA JSON status to OrderResult
                status_str = res_data.get("status", "REJECTED")
                status = OrderStatus.FILLED if status_str == "FILLED" else OrderStatus.REJECTED
                
                result = OrderResult(
                    magic=order.magic,
                    ticket=res_data.get("ticket", 0),
                    status=status,
                    volume_filled=res_data.get("volume_filled", 0.0),
                    price_filled=res_data.get("price_filled", 0.0),
                    error_code=res_data.get("error_code", 0),
                    error_message=res_data.get("error_message", ""),
                )
                
                logger.info(
                    "Order result (JSON)",
                    magic=result.magic,
                    status=result.status.name,
                    ticket=result.ticket,
                    price=result.price_filled,
                )
                
                return result  # [BUG FIX] 之前缺少 return，导致返回 None
                
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to binary result if EA still sends binary (though it should be JSON now)
                if len(response) >= OrderResultPacket.STRUCT_SIZE:
                    result_packet = OrderResultPacket.from_bytes(response)
                    result = result_packet.to_order_result()
                else:
                    result = OrderResult(
                        magic=order.magic,
                        ticket=0,
                        status=OrderStatus.REJECTED,
                        volume_filled=0,
                        price_filled=0,
                        error_code=-1,
                        error_message=f"Invalid JSON response: {response!r}",
                    )
                
                logger.info(
                    "Order result",
                    magic=result.magic,
                    status=result.status.name,
                    ticket=result.ticket,
                    price=result.price_filled,
                )
                
                return result
        
        except zmq.ZMQError as e:
            logger.error("Order send failed", error=str(e))
            return OrderResult(
                magic=order.magic,
                ticket=0,
                status=OrderStatus.REJECTED,
                volume_filled=0,
                price_filled=0,
                error_code=-1,
                error_message=str(e),
            )
    
    # ========================================================================
    # Position Query
    # ========================================================================
    
    async def query_positions(self, symbol: str | None = None) -> list["PositionInfo"]:
        """
        Query current open positions from MT5.
        
        This should be called at startup to sync position state.
        
        Args:
            symbol: Filter by symbol (None = all positions)
            
        Returns:
            List of PositionInfo objects for open positions
        """
        import json
        
        if not self._connected:
            raise RuntimeError("Not connected to MT5")
        
        # Create GET_POSITIONS request in JSON format (compatible with MT5 EA)
        request_json = json.dumps({
            "action": "GET_POSITIONS",
            "symbol": symbol or ""
        })
        request = request_json.encode('ascii')
        
        logger.debug("Querying positions from MT5", symbol=symbol)
        
        try:
            # DEALER socket: send [empty_frame, request] to ROUTER
            await self._order_socket.send_multipart([b"", request])
            
            # DEALER socket: receive [empty_frame, response] from ROUTER
            frames = await self._order_socket.recv_multipart()
            
            # Extract response (last frame, skip empty delimiter)
            if len(frames) < 1:
                logger.warning("Empty positions response from MT5")
                return []
            
            response = frames[-1]  # Response is the last frame
            
            if len(response) < 1:
                logger.warning("Empty positions response from MT5")
                return []
            
            # Try to parse JSON response
            try:
                data = json.loads(response.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback: try binary format for backwards compatibility
                msg_type = response[0]
                
                if msg_type == MessageType.ERROR:
                    error_code = struct.unpack('<i', response[1:5])[0] if len(response) >= 5 else -1
                    logger.error("MT5 returned error for positions query", error_code=error_code)
                    return []
                
                if msg_type != MessageType.POSITIONS_RESPONSE:
                    logger.warning("Unexpected response type for positions query", msg_type=msg_type)
                    return []
                
                # Parse binary positions response
                if len(response) < 5:
                    return []
                
                count = struct.unpack('<I', response[1:5])[0]
                
                if count == 0:
                    logger.info("No open positions in MT5")
                    return []
                
                positions: list[PositionInfo] = []
                offset = 5
                
                for _ in range(count):
                    if offset + PositionPacket.STRUCT_SIZE > len(response):
                        break
                    
                    packet = PositionPacket.from_bytes(response[offset:])
                    positions.append(packet.to_position_info())
                    offset += PositionPacket.STRUCT_SIZE
                
                logger.info(
                    "Positions synced from MT5 (binary)",
                    count=len(positions),
                    symbol=symbol,
                )
                
                return positions
            
            # Handle JSON response
            if data.get("error"):
                logger.error("MT5 returned error for positions query", error=data.get("error"))
                return []
            
            positions_data = data.get("positions", [])
            
            if not positions_data:
                logger.info("No open positions in MT5")
                return []
            
            positions: list[PositionInfo] = []
            for pos in positions_data:
                positions.append(PositionInfo(
                    ticket=pos.get("ticket", 0),
                    symbol=pos.get("symbol", ""),
                    direction="LONG" if pos.get("type", 0) == 0 else "SHORT",
                    volume=pos.get("volume", 0.0),
                    entry_price=pos.get("price_open", 0.0),
                    current_price=pos.get("price_current", 0.0),
                    sl=pos.get("sl", 0.0),
                    tp=pos.get("tp", 0.0),
                    profit=pos.get("profit", 0.0),
                    open_time_us=pos.get("time", 0),
                    magic=pos.get("magic", 0),
                    comment=pos.get("comment", ""),
                ))
            
            logger.info(
                "Positions synced from MT5",
                count=len(positions),
                symbol=symbol,
            )
            
            return positions
            
        except zmq.ZMQError as e:
            logger.error("Position query failed", error=str(e))
            return []
        except Exception as e:
            logger.error("Position query error", error=str(e))
            return []

    # ========================================================================
    # Account/Status Query
    # ========================================================================

    async def get_status(self) -> dict:
        """
        Query account/runtime status from MT5 EA.
        
        Uses the EA's STATUS action on the order socket.
        """
        import json
        
        if not self._connected:
            raise RuntimeError("Not connected to MT5")
        
        request = json.dumps({"action": "STATUS"}).encode("ascii")
        
        try:
            await self._order_socket.send_multipart([b"", request])
            frames = await self._order_socket.recv_multipart()
            response = frames[-1] if frames else b""
            data = json.loads(response.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error("Status query failed", error=str(e))
            return {}

    # ========================================================================
    # Symbol Info Query
    # ========================================================================

    async def get_symbol_info(self, symbol: str) -> dict:
        """
        Query symbol info from MT5 EA via history socket.
        
        Uses: GET_SYMBOL_INFO|SYMBOL
        """
        import json
        
        if not self._connected or self._history_socket is None:
            raise RuntimeError("Not connected to MT5 or history socket not available")
        
        request = f"GET_SYMBOL_INFO|{symbol}"
        
        try:
            await self._history_socket.send_string(request)
            response = await self._history_socket.recv_string()
            data = json.loads(response)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error("Symbol info query failed", error=str(e))
            return {}
    
    # ========================================================================
    # History Data Request
    # ========================================================================
    
    async def get_history(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Request historical OHLC data from MT5 EA.
        
        Uses the EA's GET_HISTORY protocol:
        - Request: GET_HISTORY|SYMBOL|TIMEFRAME|START_DATE|END_DATE
        - Response: CSV|COUNT|COLUMNS|DATA
        
        Args:
            symbol: Symbol name (e.g., "XAUUSD")
            timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1
            start_date: Start date in format "YYYY-MM-DD" or "YYYY.MM.DD HH:MM:SS"
            end_date: End date in same format
            
        Returns:
            List of bar dicts with keys: time, open, high, low, close, tick_volume, spread, real_volume
        """
        if not self._connected or self._history_socket is None:
            raise RuntimeError("Not connected to MT5 or history socket not available")
        
        # Build request string
        request = f"GET_HISTORY|{symbol}|{timeframe}|{start_date}|{end_date}"
        
        logger.debug(
            "Requesting history data",
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )
        
        try:
            # Send request (REQ socket)
            await self._history_socket.send_string(request)
            
            # Receive response
            response = await self._history_socket.recv_string()
            
            # Parse response
            if response.startswith("ERROR|"):
                error_msg = response.split("|", 1)[1] if "|" in response else response
                logger.error("History request failed", error=error_msg)
                return []
            
            if not response.startswith("CSV|"):
                logger.warning("Unexpected history response format", response=response[:100])
                return []
            
            # Parse CSV response: CSV|COUNT|COLUMNS|DATA
            parts = response.split("|", 3)
            if len(parts) < 4:
                logger.warning("Malformed history response")
                return []
            
            count = int(parts[1])
            columns = parts[2].split(",")
            csv_data = parts[3]
            
            # Parse CSV lines into dicts
            bars = []
            for line in csv_data.split("\n"):
                if not line.strip():
                    continue
                values = line.split(",")
                if len(values) != len(columns):
                    continue
                
                bar = {}
                for col, val in zip(columns, values):
                    col = col.strip()
                    val = val.strip()
                    if col == "time":
                        bar[col] = val
                    elif col in ("tick_volume", "spread", "real_volume"):
                        bar[col] = int(val) if val else 0
                    else:
                        bar[col] = float(val) if val else 0.0
                bars.append(bar)
            
            logger.info(
                "History data received",
                symbol=symbol,
                timeframe=timeframe,
                bars=len(bars),
                expected=count,
            )
            
            return bars
            
        except zmq.ZMQError as e:
            logger.error("History request ZMQ error", error=str(e))
            return []
        except Exception as e:
            logger.error("History request error", error=str(e))
            return []
    
    async def ping_history(self) -> bool:
        """
        Ping the history socket to check connectivity.
        
        Returns:
            True if connected and responsive
        """
        if not self._connected or self._history_socket is None:
            return False
        
        try:
            await self._history_socket.send_string("PING")
            response = await self._history_socket.recv_string()
            return response == "PONG"
        except Exception:
            return False
    
    # ========================================================================
    # v4.0: Tick History Replay Control (for BOOTSTRAP_REPLAY)
    # ========================================================================
    
    async def start_tick_replay(
        self,
        symbol: str,
        window_sec: int = 86400,
        end_eps_ms: int = 1000,
        max_ticks: int = 2000000,
        pace_tps: int = 50000,
    ) -> dict:
        """
        Start tick history replay from MT5 EA (v4.0).
        
        Instructs EA to load historical ticks and stream them through 
        the same PUB socket used for live ticks. This ensures exact
        parity between replay and live tick processing.
        
        Protocol: START_REPLAY_TICKS|SYMBOL|WINDOW_SEC|END_EPS_MS|MAX_TICKS|PACE_TPS
        Response: OK|REPLAY_STARTED|count=...|start=...|end=...|pace=...
        
        Args:
            symbol: Symbol name (e.g., "XAUUSD")
            window_sec: History window in seconds (default: 86400 = 24h)
            end_eps_ms: Epsilon offset from TimeCurrent() in ms (default: 1000)
                        Ensures replay ticks are strictly in the past.
            max_ticks: Maximum ticks to load (safety limit, default: 2M)
            pace_tps: Ticks per second to send (default: 50000)
            
        Returns:
            dict with keys: success, count, start, end, pace, error
        """
        if not self._connected or self._history_socket is None:
            return {"success": False, "error": "Not connected to MT5"}
        
        request = f"START_REPLAY_TICKS|{symbol}|{window_sec}|{end_eps_ms}|{max_ticks}|{pace_tps}"
        
        logger.info(
            "Starting tick replay",
            symbol=symbol,
            window_sec=window_sec,
            end_eps_ms=end_eps_ms,
            max_ticks=max_ticks,
            pace_tps=pace_tps,
        )
        
        try:
            await self._history_socket.send_string(request)
            response = await self._history_socket.recv_string()
            
            # Parse response: OK|REPLAY_STARTED|count=...|start=...|end=...|pace=...
            if response.startswith("ERROR|"):
                error_msg = response.split("|", 1)[1] if "|" in response else response
                logger.error("Replay start failed", error=error_msg)
                return {"success": False, "error": error_msg}
            
            if response.startswith("OK|REPLAY_STARTED"):
                result = {"success": True}
                for part in response.split("|")[2:]:
                    if "=" in part:
                        key, val = part.split("=", 1)
                        if key in ("count", "pace"):
                            result[key] = int(val)
                        else:
                            result[key] = val
                
                logger.info(
                    "Tick replay started",
                    count=result.get("count"),
                    start=result.get("start"),
                    end=result.get("end"),
                )
                return result
            
            logger.warning("Unexpected replay response", response=response[:100])
            return {"success": False, "error": f"Unexpected response: {response[:100]}"}
            
        except zmq.ZMQError as e:
            logger.error("Replay start ZMQ error", error=str(e))
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("Replay start error", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def stop_tick_replay(self) -> dict:
        """
        Stop tick history replay (v4.0).
        
        Response: OK|REPLAY_STOPPED|sent=...|remaining=...
        
        Returns:
            dict with keys: success, sent, remaining, error
        """
        if not self._connected or self._history_socket is None:
            return {"success": False, "error": "Not connected to MT5"}
        
        try:
            await self._history_socket.send_string("STOP_REPLAY_TICKS")
            response = await self._history_socket.recv_string()
            
            if response.startswith("OK|REPLAY_STOPPED") or response.startswith("OK|REPLAY_NOT_ACTIVE"):
                result = {"success": True}
                for part in response.split("|")[2:]:
                    if "=" in part:
                        key, val = part.split("=", 1)
                        result[key] = int(val) if val.isdigit() else val
                
                logger.info(
                    "Tick replay stopped",
                    sent=result.get("sent"),
                    remaining=result.get("remaining"),
                )
                return result
            
            return {"success": False, "error": f"Unexpected response: {response[:100]}"}
            
        except Exception as e:
            logger.error("Replay stop error", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_replay_status(self) -> dict:
        """
        Get tick replay status (v4.0).
        
        Response: OK|REPLAY_STATUS|active=...|sent=...|total=...|progress=...
        
        Returns:
            dict with keys: active, sent, total, progress
        """
        if not self._connected or self._history_socket is None:
            return {"active": False, "error": "Not connected"}
        
        try:
            await self._history_socket.send_string("GET_REPLAY_STATUS")
            response = await self._history_socket.recv_string()
            
            if response.startswith("OK|REPLAY_STATUS"):
                result = {}
                for part in response.split("|")[2:]:
                    if "=" in part:
                        key, val = part.split("=", 1)
                        if key == "active":
                            result[key] = val.lower() == "true"
                        elif key in ("sent", "total"):
                            result[key] = int(val) if val.isdigit() else 0
                        elif key == "progress":
                            result[key] = float(val) if val else 0.0
                        else:
                            result[key] = val
                return result
            
            return {"active": False, "error": f"Unexpected response: {response[:100]}"}
            
        except Exception as e:
            return {"active": False, "error": str(e)}
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "connected": self._connected,
            "tick_count": self._tick_count,
            "order_count": self._order_count,
            "last_tick_age_s": time.time() - self._last_tick_time if self._last_tick_time else None,
        }


# ============================================================================
# Position Packet and Info
# ============================================================================

class PositionInfo(NamedTuple):
    """
    Position information from MT5.
    
    Used for syncing position state at startup.
    """
    ticket: int
    symbol: str
    direction: str  # "LONG" or "SHORT"
    volume: float
    entry_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    open_time_us: int
    magic: int
    comment: str


@dataclass
class PositionPacket:
    """
    Binary packet for position data (MT5 -> Python).
    
    Format: '<Q20sBdddddqQ64s' (168 bytes)
    - ticket: uint64 (8)
    - symbol: char[20] (20)
    - direction: uint8 (1) - 0=BUY/LONG, 1=SELL/SHORT
    - volume: double (8)
    - open_price: double (8)
    - current_price: double (8)
    - sl: double (8)
    - tp: double (8)
    - profit: double (8)
    - open_time_us: int64 (8)
    - magic: uint64 (8)
    - comment: char[64] (64)
    
    Total: 157 bytes (padded to 168 for alignment)
    """
    ticket: int
    symbol: bytes
    direction: int
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    open_time_us: int
    magic: int
    comment: bytes
    
    STRUCT_FORMAT = '<Q20sBdddddqQ64s'
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "PositionPacket":
        """Deserialize from binary data."""
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data[:cls.STRUCT_SIZE])
        return cls(
            ticket=unpacked[0],
            symbol=unpacked[1],
            direction=unpacked[2],
            volume=unpacked[3],
            open_price=unpacked[4],
            current_price=unpacked[5],
            sl=unpacked[6],
            tp=unpacked[7],
            profit=unpacked[8],
            open_time_us=unpacked[9],
            magic=unpacked[10],
            comment=unpacked[11],
        )
    
    def to_position_info(self) -> PositionInfo:
        """Convert to PositionInfo object."""
        return PositionInfo(
            ticket=self.ticket,
            symbol=self.symbol.rstrip(b'\x00').decode('utf-8', errors='ignore'),
            direction="LONG" if self.direction == 0 else "SHORT",
            volume=self.volume,
            entry_price=self.open_price,
            current_price=self.current_price,
            sl=self.sl,
            tp=self.tp,
            profit=self.profit,
            open_time_us=self.open_time_us,
            magic=self.magic,
            comment=self.comment.rstrip(b'\x00').decode('utf-8', errors='ignore'),
        )
