import asyncio
import json
import logging
from typing import Set, Any, Iterable
from urllib.parse import parse_qs, urlparse

import websockets
from alphaos.monitoring.runtime_state import RuntimeSnapshot
from alphaos.core.logging import get_logger

# Use structlog-style logger (supports key=value context)
logger = get_logger(__name__)

class WSRuntimeServer:
    """
    WebSocket Server for pushing Runtime State Snapshots to UI (port 8765).
    Acts as a bridge between Backend SSOT and Frontend UI.
    """
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        allowed_origins: Iterable[str] | None = None,
        auth_tokens: Iterable[str] | None = None,
    ):
        self.host = host
        self.port = port
        self.allowed_origins = set(allowed_origins or [])
        self.auth_tokens = set(auth_tokens or [])
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self._last_snapshot: RuntimeSnapshot | None = None
        self._last_positions: list[dict[str, Any]] | None = None
        self.allowed_origins = {origin for origin in (allowed_origins or []) if origin}
        self.auth_tokens = {token for token in (auth_tokens or []) if token}
        
    async def start(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
        logger.info("WSRuntimeServer started", ws=f"ws://{self.host}:{self.port}")
        
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
        clients = list(self.clients)
        results = await asyncio.gather(
            *[client.send(message) for client in clients],
            return_exceptions=True
        )
        for client, result in zip(clients, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("WebSocket send failed; removing client: %s", result)
                self.clients.discard(client)

    async def broadcast(self, snapshot: RuntimeSnapshot) -> None:
        """Broadcast snapshot to all connected clients."""
        self._last_snapshot = snapshot
        await self.broadcast_message("runtime_snapshot", snapshot.model_dump())

    async def broadcast_positions(self, positions: Iterable[dict[str, Any]]) -> None:
        """Broadcast open positions list to all connected clients."""
        positions_list = list(positions)
        self._last_positions = positions_list
        await self.broadcast_message("position", {"positions": positions_list})

    def _is_origin_allowed(self, origin: str | None) -> bool:
        if not self.allowed_origins:
            return True
        return origin in self.allowed_origins

    def _get_path(self, conn: Any) -> str:
        """
        websockets compatibility:
        - websockets<=11: conn.path + conn.request_headers
        - websockets>=12 (e.g. 16.0): conn.request.path + conn.request.headers
        """
        path = getattr(conn, "path", None)
        if isinstance(path, str):
            return path
        req = getattr(conn, "request", None)
        if req is not None:
            req_path = getattr(req, "path", None)
            if isinstance(req_path, str):
                return req_path
        return ""

    def _get_headers(self, conn: Any) -> Any:
        headers = getattr(conn, "request_headers", None)
        if headers is not None:
            return headers
        req = getattr(conn, "request", None)
        if req is not None:
            req_headers = getattr(req, "headers", None)
            if req_headers is not None:
                return req_headers
        return {}

    def _extract_token(self, conn: Any) -> str | None:
        parsed = urlparse(self._get_path(conn) or "")
        params = parse_qs(parsed.query)
        token = params.get("token", [None])[0]
        if token:
            return token
        headers = self._get_headers(conn)
        auth_header = headers.get("Authorization", "") if hasattr(headers, "get") else ""
        if auth_header.lower().startswith("bearer "):
            candidate = auth_header[7:].strip()
            return candidate or None
        header_token = headers.get("X-AlphaOS-Token") if hasattr(headers, "get") else None
        return header_token

    def _is_token_valid(self, token: str | None) -> bool:
        if not self.auth_tokens:
            return True
        return token in self.auth_tokens

    async def _handler(self, websocket: Any):
        headers = self._get_headers(websocket)
        origin = headers.get("Origin") if hasattr(headers, "get") else None
        token = self._extract_token(websocket)
        if not self._is_origin_allowed(origin) or not self._is_token_valid(token):
            logger.warning(
                "Rejected WS client",
                origin=origin,
                has_token=bool(token),
                path=self._get_path(websocket),
            )
            await websocket.close(code=1008, reason="Forbidden")
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
            wait_closed = getattr(websocket, "wait_closed", None)
            if callable(wait_closed):
                await wait_closed()
            else:
                await asyncio.Future()
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
        
        from alphaos.monitoring.runtime_state import RuntimeSnapshot
        import time
        from datetime import datetime
        
        fake_snapshot = RuntimeSnapshot(
            timestamp=0,
            symbol="XAUUSD",
            ticks_total=0,
            market_phase="UNKNOWN",
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
