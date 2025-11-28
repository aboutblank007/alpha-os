import asyncio
import grpc
import logging
import os
import sys
import pandas as pd
import numpy as np
import time
import lightgbm as lgb

# Adjust path to import generated proto and features
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import alphaos_pb2
import alphaos_pb2_grpc
from features import FeatureEngineer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AI-Engine")

# Configuration
# Default to remote IP if not set
DEFAULT_REMOTE_IP = "49.235.153.73:50051" 
CLOUD_BRIDGE_URL = os.environ.get("CLOUD_BRIDGE_URL", DEFAULT_REMOTE_IP)
MODEL_PATH = "ai-engine/models/lgbm_scalping_v1.txt"

# Scalping Rules
TP_PCT = 0.001   # 0.1%
SL_PCT = 0.0008  # 0.08%

class LocalAIEngine:
    def __init__(self):
        self.channel = None
        self.stub = None
        self.is_connected = False
        self.fe = FeatureEngineer()
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = lgb.Booster(model_file=MODEL_PATH)
                logger.info(f"✅ Loaded LightGBM model from {MODEL_PATH}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
        else:
            logger.warning(f"⚠️ Model file {MODEL_PATH} not found. Will run in dummy mode.")

    async def connect(self):
        """Establish connection to Cloud Bridge"""
        logger.info(f"Connecting to Cloud Bridge at {CLOUD_BRIDGE_URL}...")
        self.channel = grpc.aio.insecure_channel(CLOUD_BRIDGE_URL)
        self.stub = alphaos_pb2_grpc.AlphaZeroStub(self.channel)
        self.is_connected = True
        
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
                response_queue = asyncio.Queue()
                
                async def request_generator():
                    while True:
                        response = await response_queue.get()
                        yield response

                logger.info("🎧 Waiting for signals...")
                
                async for signal_request in self.stub.StreamSignals(request_generator()):
                    logger.info(f"📥 Received Signal: {signal_request.symbol} {signal_request.action}")
                    
                    # Process the signal
                    response = await self.process_signal(signal_request)
                    
                    # Send response back
                    await response_queue.put(response)
                    logger.info(f"📤 Sent Response: {response.should_execute} (Conf: {response.confidence:.2f})")
            
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.error(f"❌ Cloud Bridge Unavailable (Connection Refused). Retrying in 5s...")
                else:
                    logger.error(f"gRPC Error: {e}")
                
                await asyncio.sleep(5)
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
        
        # 2. Feature Engineering
        # Extract DOM data if available
        dom_bids = request.dom_bids
        dom_asks = request.dom_asks
        
        features = self.fe.get_latest_features(df, dom_bids, dom_asks)
        
        # 3. Inference
        confidence = 0.0
        if self.model and features:
            # Ensure feature order matches model expectation (simple dict -> list conversion)
            # LightGBM Booster.predict usually wants 2D array.
            # We need to align keys with model feature names.
            # For now, passing values assuming dict-based prediction support or conversion.
            # Note: Booster.predict needs raw data list in correct order.
            model_features = self.model.feature_name()
            feature_vector = []
            for name in model_features:
                feature_vector.append(features.get(name, 0.0))
            
            try:
                # predict returns list of probs
                confidence = self.model.predict([feature_vector])[0]
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                confidence = 0.5
        else:
            # Dummy fallback
            rsi = features.get('rsi', 50)
            confidence = 0.8 if (rsi < 30 and request.action == 'BUY') or (rsi > 70 and request.action == 'SELL') else 0.4

        should_execute = confidence > 0.6
        
        # 4. Adjust Trade Params for Scalping
        # Calculate price levels based on scalping rules (0.1% TP, 0.08% SL)
        current_price = features.get('close', 0)
        adjusted_tp = 0.0
        adjusted_sl = 0.0
        
        if current_price > 0:
            if request.action == 'BUY':
                adjusted_tp = current_price * (1 + TP_PCT)
                adjusted_sl = current_price * (1 - SL_PCT)
            elif request.action == 'SELL':
                adjusted_tp = current_price * (1 - TP_PCT)
                adjusted_sl = current_price * (1 + SL_PCT)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return alphaos_pb2.SignalResponse(
            request_id=request.request_id,
            client_id="local-m2-pro-scalper",
            should_execute=should_execute,
            action=request.action,
            confidence=confidence,
            adjusted_sl=adjusted_sl,
            adjusted_tp=adjusted_tp,
            reason=f"Scalping Model v1 (Latency: {latency_ms:.2f}ms)"
        )

if __name__ == "__main__":
    engine = LocalAIEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Stopping AI Engine...")
