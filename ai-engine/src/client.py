import asyncio
import grpc
import logging
import os
import sys
import pandas as pd
import numpy as np
import time

# Adjust path to import generated proto
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import alphaos_pb2
import alphaos_pb2_grpc

# Configure Logging
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'
# os.environ['GRPC_TRACE'] = 'connectivity_state,http'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AI-Engine")

# Configuration
# Default to remote IP if not set
DEFAULT_REMOTE_IP = "49.235.153.73:50051" 
CLOUD_BRIDGE_URL = os.environ.get("CLOUD_BRIDGE_URL", DEFAULT_REMOTE_IP)

class LocalAIEngine:
    def __init__(self):
        self.channel = None
        self.stub = None
        self.is_connected = False

    async def connect(self):
        """Establish connection to Cloud Bridge"""
        logger.info(f"Connecting to Cloud Bridge at {CLOUD_BRIDGE_URL}...")
        self.channel = grpc.aio.insecure_channel(CLOUD_BRIDGE_URL)
        self.stub = alphaos_pb2_grpc.AlphaZeroStub(self.channel)
        self.is_connected = True
        # Note: insecure_channel is non-blocking by default, actual connection check happens on first call
        # But we can wait for ready
        try:
             await asyncio.wait_for(self.channel.channel_ready(), timeout=10.0)
             logger.info("✅ Connected to Cloud Bridge.")
        except asyncio.TimeoutError:
             logger.warning("⚠️ Connection handshake timed out after 10s. Server might be reachable but not speaking gRPC.")
        except Exception as e:
             logger.warning(f"⚠️ Connection handshake failed: {e}")

    async def run(self):
        await self.connect()
        
        while True:
            try:
                # Start the bi-directional stream
                # We use an iterator to send responses
                response_queue = asyncio.Queue()
                
                async def request_generator():
                    # Send an initial ping or empty response to establish stream presence if needed
                    # But our proto StreamSignals expects SignalResponse.
                    # We can send a dummy one or just wait.
                    while True:
                        response = await response_queue.get()
                        yield response

                logger.info("🎧 Waiting for signals...")
                
                # This iterator will block until a message comes from server
                async for signal_request in self.stub.StreamSignals(request_generator()):
                    logger.info(f"📥 Received Signal: {signal_request.symbol} {signal_request.action}")
                    
                    # Process the signal
                    response = await self.process_signal(signal_request)
                    
                    # Send response back
                    await response_queue.put(response)
                    logger.info(f"📤 Sent Response: {response.should_execute}")
            
            except grpc.RpcError as e:
                # Check status code
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.error(f"❌ Cloud Bridge Unavailable (Connection Refused). Retrying in 5s...")
                else:
                    logger.error(f"gRPC Error: {e}")
                
                await asyncio.sleep(5)
                # Re-create channel on hard failures
                await self.connect()
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(5)

    async def process_signal(self, request: alphaos_pb2.SignalRequest) -> alphaos_pb2.SignalResponse:
        """
        Core Logic: Feature Engineering + Inference
        """
        start_time = time.time()
        
        # 1. Convert Context to DataFrame
        data = []
        for c in request.market_context:
            data.append({
                'time': c.time,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            })
        df = pd.DataFrame(data)
        
        # 2. Feature Engineering (Simplified Port)
        features = self.calculate_features(df)
        
        # 3. Inference (Mock for now)
        # In real implementation, load LightGBM model here
        confidence = self.predict_dummy(features)
        
        should_execute = confidence > 0.6
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"🧠 Inference done in {latency_ms:.2f}ms | Conf: {confidence:.2f}")
        
        return alphaos_pb2.SignalResponse(
            request_id=request.request_id,
            client_id="local-m2-pro",
            should_execute=should_execute,
            action=request.action,
            confidence=confidence,
            reason=f"Local AI Decision (Latency: {latency_ms:.2f}ms)"
        )

    def calculate_features(self, df: pd.DataFrame):
        if df.empty: return {}
        
        # Simple RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        return {
            "rsi": latest.get('rsi', 50),
            "close": latest['close']
        }

    def predict_dummy(self, features):
        # Mock logic
        rsi = features.get('rsi', 50)
        # Trend Following Logic for test
        if rsi > 70: return 0.2 # Overbought, don't buy
        if rsi < 30: return 0.8 # Oversold, buy
        return 0.65 # Neutral

if __name__ == "__main__":
    engine = LocalAIEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Stopping AI Engine...")
