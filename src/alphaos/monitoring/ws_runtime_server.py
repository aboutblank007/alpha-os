import asyncio
import json
import logging
from typing import Set, Any, Iterable
from urllib.parse import parse_qs, urlparse

import websockets
from alphaos.monitoring.runtime_state import RuntimeSnapshot

logger = logging.getLogger(__name__)

class WSRuntimeServer:
    """
    WebSocket Server for pushing Runtime State Snapshots to UI (port 8765).
    Acts as a bridge between Backend SSOT and Frontend UI.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        allowed_origins: Iterable[str] | None = None,
        auth_tokens: Iterable[str] | None = None,
    ):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self._last_snapshot: RuntimeSnapshot | None = None
        self._last_positions: list[dict[str, Any]] | None = None
        self.allowed_origins = {origin for origin in (allowed_origins or []) if origin}
        self.auth_tokens = {token for token in (auth_tokens or []) if token}
        
    async def start(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
        logger.info(f"WSRuntimeServer started on ws://{self.host}:{self.port}")
        
    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
    async def broadcast_message(self, msg_type: str, data: Any) -> None:
        """Broadcast a generic WS message to all connected clients."""
        if not self.clients:
            return
        message = json.dumps({
            "type": msg_type,
            "data": data,
        })
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )

    async def broadcast(self, snapshot: RuntimeSnapshot) -> None:
        """Broadcast snapshot to all connected clients."""
        self._last_snapshot = snapshot
        await self.broadcast_message("runtime_snapshot", snapshot.model_dump())

    async def broadcast_positions(self, positions: Iterable[dict[str, Any]]) -> None:
        """Broadcast open positions list to all connected clients."""
        positions_list = list(positions)
        self._last_positions = positions_list
        await self.broadcast_message("position", {"positions": positions_list})

    def _get_origin(self, websocket: websockets.WebSocketServerProtocol) -> str | None:
        return websocket.request_headers.get("Origin")

    def _get_token(self, websocket: websockets.WebSocketServerProtocol) -> str | None:
        auth_header = websocket.request_headers.get("Authorization")
        if auth_header:
            if auth_header.lower().startswith("bearer "):
                return auth_header.split(" ", 1)[1].strip()
            return auth_header.strip()

        header_token = websocket.request_headers.get("X-Auth-Token")
        if header_token:
            return header_token.strip()

        query = urlparse(websocket.path or "").query
        params = parse_qs(query)
        for key in ("token", "auth_token", "access_token"):
            if key in params and params[key]:
                return params[key][0]
        return None

    async def _reject(self, websocket: websockets.WebSocketServerProtocol, reason: str) -> None:
        logger.warning("WS connection rejected", reason=reason, path=websocket.path)
        await websocket.close(code=1008, reason=reason)

    async def _validate_connection(self, websocket: websockets.WebSocketServerProtocol) -> bool:
        if self.allowed_origins:
            origin = self._get_origin(websocket)
            if origin not in self.allowed_origins:
                await self._reject(websocket, "origin_not_allowed")
                return False

        if self.auth_tokens:
            token = self._get_token(websocket)
            if token not in self.auth_tokens:
                await self._reject(websocket, "invalid_token")
                return False

        return True

    async def _handler(self, websocket: websockets.WebSocketServerProtocol):
        if not await self._validate_connection(websocket):
            return

        self.clients.add(websocket)
        try:
            # Send welcome & last snapshot immediately
            await websocket.send(json.dumps({"type": "welcome", "message": "AlphaOS v4 Runtime Stream"}))
            if self._last_snapshot:
                await websocket.send(json.dumps({
                    "type": "runtime_snapshot",
                    "data": self._last_snapshot.model_dump()
                }))
            if self._last_positions is not None:
                await websocket.send(json.dumps({
                    "type": "position",
                    "data": {"positions": self._last_positions}
                }))
            
            # Keep connection open
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

if __name__ == "__main__":
    import struct
    import zmq
    import zmq.asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    server = WSRuntimeServer()
    
    async def zmq_broadcast_loop(server: WSRuntimeServer):
        """Listen to MT5 ZMQ execution ticks and broadcast to WS"""
        context = zmq.asyncio.Context()
        sock = context.socket(zmq.SUB)
        # Default MT5 ZMQ port is 5555 for ticks
        sock.connect("tcp://127.0.0.1:5555")
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        
        logger.info("Connected to MT5 ZMQ at tcp://127.0.0.1:5555")
        
        from alphaos.monitoring.runtime_state import RuntimeSnapshot, MarketPhase
        import time
        from datetime import datetime
        
        fake_snapshot = RuntimeSnapshot(
            timestamp=0,
            symbol="XAUUSD",
            ticks_total=0,
            market_phase=MarketPhase.UNKNOWN,
            temperature=0.0,
            entropy=0.0
        )

        try:
            while True:
                msg = await sock.recv()
                if len(msg) == 36: # Struct format for Tick
                     # double bid, double ask, long long time, unsigned int volume, unsigned int flags
                    bid, ask, timestamp, volume, flags = struct.unpack("ddqqI", msg)
                    
                    # Update snapshot with live data
                    fake_snapshot.timestamp = timestamp / 1000.0 if timestamp > 1e12 else float(timestamp)
                    fake_snapshot.ticks_total += 1
                    
                    # Basic "fake" metrics for UI demo purposes if actual engine isn't running
                    # In a real setup, the engine computes these and broadcasts the snapshot.
                    # Here we just want to prove the UI works with live ticks.
                    fake_snapshot.temperature = abs(ask - bid) * 100 # Dummy logic
                    fake_snapshot.entropy = (volume % 100) / 100.0
                    
                    await server.broadcast(fake_snapshot)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ZMQ loop error: {e}")
            
    async def main():
        await server.start()
        
        # Start ZMQ bridge task
        zmq_task = asyncio.create_task(zmq_broadcast_loop(server))
        
        # Keep alive
        try:
            await asyncio.Future() # run forever
        except asyncio.CancelledError:
            pass
        finally:
            zmq_task.cancel()
            await server.stop()

    # Run server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopping WebSocket Server...")
    except Exception as e:
        logger.error(f"WebSocket Server error: {e}")
