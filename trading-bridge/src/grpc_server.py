import grpc
import logging
import asyncio
import time
from typing import Dict, AsyncIterable

# These will be generated at runtime inside the container
import alphaos_pb2
import alphaos_pb2_grpc

logger = logging.getLogger(__name__)

class AlphaZeroService(alphaos_pb2_grpc.AlphaZeroServicer):
    def __init__(self):
        self.clients: Dict[str, asyncio.Queue] = {} # client_id -> Queue of SignalRequest
        self.responses: Dict[str, asyncio.Future] = {} # request_id -> Future(SignalResponse)

    async def HealthCheck(self, request, context):
        return alphaos_pb2.Pong(
            server_id="cloud-bridge-1",
            timestamp=int(time.time() * 1000),
            ready=True
        )

    async def StreamSignals(self, request_iterator, context):
        client_id = "unknown"
        
        # 1. Handle the stream connection
        try:
            # We expect the first message or metadata to identify the client, 
            # but for bi-streaming, we can just assign an ID or wait for first ping.
            # For simplicity, we'll generate one.
            client_id = f"client-{int(time.time())}"
            logger.info(f"New gRPC client connected: {client_id}")
            
            self.clients[client_id] = asyncio.Queue()

            # Create a task to read from the client (Responses)
            read_task = asyncio.create_task(self._read_stream(request_iterator, client_id))
            
            # Main loop: Write requests to the client
            while True:
                signal_request = await self.clients[client_id].get()
                yield signal_request
                
        except asyncio.CancelledError:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error in StreamSignals for {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            logger.info(f"Cleaned up client {client_id}")

    async def _read_stream(self, request_iterator, client_id):
        """Reads SignalResponse from the client."""
        try:
            async for response in request_iterator:
                req_id = response.request_id
                if req_id in self.responses:
                    if not self.responses[req_id].done():
                        self.responses[req_id].set_result(response)
                else:
                    logger.warning(f"Received response for unknown request {req_id}")
        except Exception as e:
            logger.error(f"Error reading stream from {client_id}: {e}")

    async def send_signal_and_wait(self, signal_request: alphaos_pb2.SignalRequest, timeout=5.0) -> alphaos_pb2.SignalResponse:
        """
        Sends a request to the first available client and waits for response.
        """
        if not self.clients:
            return None # No AI engine connected

        # Simple strategy: Round robin or just pick first
        client_id = next(iter(self.clients))
        queue = self.clients[client_id]
        
        # Register the future
        req_id = signal_request.request_id
        future = asyncio.get_running_loop().create_future()
        self.responses[req_id] = future
        
        try:
            await queue.put(signal_request)
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for AI response for {req_id}")
            return None
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return None
        finally:
            if req_id in self.responses:
                del self.responses[req_id]

async def start_grpc_server(service: AlphaZeroService, port=50051):
    server = grpc.aio.server()
    alphaos_pb2_grpc.add_AlphaZeroServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    logger.info(f"🚀 gRPC Server starting on port {port}...")
    await server.start()
    # Run in background
    asyncio.create_task(server.wait_for_termination())
    return server

