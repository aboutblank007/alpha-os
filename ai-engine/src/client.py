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
# We will dynamic adjust TP based on MFE prediction
BASE_SL_PCT = 0.0008  # 0.08% Base SL

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
                    logger.info(f"📤 Sent Response: {response.should_execute} (MFE Pred: {response.confidence:.2f})")
            
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
        
        # 1. Extract Technical Context from Request (Protobuf -> Dict)
        tc = request.technical_context
        
        # Current Features (Snapshot)
        # Map protobuf fields to model feature names
        # Note: 'rsi', 'tick_volume', 'spread', 'candle_size' are NEW in protobuf
        # If Cloud Bridge hasn't been updated to send these, we might need fallback or compute from history
        
        # Assuming Cloud Bridge sends raw features in `technical_context` if updated, 
        # OR we compute from `market_context` (candles).
        
        # Let's compute dynamic features from market_context just to be safe/fresh
        candles = []
        for c in request.market_context:
            candles.append({
                'time': c.time,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'tick_volume': c.volume
            })
        df = pd.DataFrame(candles)
        
        # Basic Features from protobuf (provided by EA)
        current_features = {
            'price_vs_center': tc.price_vs_center,
            'adx': tc.adx,
            'atr_percent': tc.atr_percent,
            'cloud_width': tc.cloud_width,
            'ema_spread': tc.ema_spread,
            'rsi': getattr(tc, 'rsi', 50.0), # Fallback if proto not updated
            'reclaim_state': tc.reclaim_state,
            'is_reclaim_signal': 1 if tc.is_reclaim_signal else 0,
            'bars_since_last': tc.bars_since_last,
            'trend_direction': tc.trend_direction,
            'ema_cross_event': tc.ema_cross_event,
            # Microstructure (from EA or computed)
            'tick_volume': getattr(tc, 'tick_volume', 0), 
            'spread': getattr(tc, 'spread', 0.0),
            'candle_size': getattr(tc, 'candle_size', 0.0),
            'wick_upper': getattr(tc, 'wick_upper', 0.0),
            'wick_lower': getattr(tc, 'wick_lower', 0.0),
        }
        
        # Add Time Features
        current_dt = pd.to_datetime(time.time(), unit='s')
        current_features['hour'] = current_dt.hour
        current_features['day_of_week'] = current_dt.dayofweek

        # 3. Inference
        predicted_mfe = 0.0
        should_execute = False
        reason = "Model not loaded"
        
        if self.model:
            model_features = self.model.feature_name()
            feature_vector = []
            missing_feats = []
            
            for name in model_features:
                val = current_features.get(name)
                if val is None:
                    # Try to compute from history if missing in EA payload
                    # (Simplified for now: just fill 0)
                    val = 0.0
                    missing_feats.append(name)
                feature_vector.append(val)
            
            if missing_feats:
                logger.warning(f"⚠️ Missing features for inference: {missing_feats}")

            try:
                # Predict MFE (Regression)
                preds = self.model.predict([feature_vector])
                predicted_mfe = preds[0]
                
                # Decision Logic (Regression based)
                # If Predicted MFE > Threshold (e.g., 2.0 points), EXECUTE
                # Threshold should be tuned. From training: Top 20% had mean MFE ~3.96
                MFE_THRESHOLD = 2.5 
                
                should_execute = predicted_mfe > MFE_THRESHOLD
                reason = f"Pred MFE: {predicted_mfe:.2f} (Thresh: {MFE_THRESHOLD})"
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                reason = f"Inference Error: {e}"
        else:
            # Dummy fallback
            should_execute = True # Default allow in dry run
            reason = "Dummy Mode"

        # 4. Adjust Trade Params (Dynamic TP)
        current_price = request.suggested_entry
        adjusted_tp = 0.0
        adjusted_sl = request.suggested_sl
        
        if current_price > 0 and should_execute:
            # Set TP based on Predicted MFE (Conservative: 80% of prediction)
            # MFE is in points/price diff.
            conservative_mfe = predicted_mfe * 0.8
            
            if request.action == 'BUY':
                adjusted_tp = current_price + conservative_mfe
                # Ensure SL is at least min distance? EA handles this usually.
            elif request.action == 'SELL':
                adjusted_tp = current_price - conservative_mfe
        
        latency_ms = (time.time() - start_time) * 1000
        
        return alphaos_pb2.SignalResponse(
            request_id=request.request_id,
            client_id="local-m2-pro-scalper-v2",
            should_execute=should_execute,
            action=request.action,
            confidence=predicted_mfe, # Return MFE as confidence score
            adjusted_sl=adjusted_sl,
            adjusted_tp=adjusted_tp,
            reason=f"{reason} | {latency_ms:.2f}ms"
        )

if __name__ == "__main__":
    engine = LocalAIEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Stopping AI Engine...")
