import asyncio
import grpc
import logging
import os
import sys
import pandas as pd
import numpy as np
import time
import torch
import lightgbm as lgb
import json
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from supabase import create_client

# Adjust path to import generated proto and features
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import alphaos_pb2
import alphaos_pb2_grpc
# Use Polars for High Performance
from features_polars import FeatureEngineerPolars
from models.quantum_net import QuantumNetLite
from models.dqn import DQNAgent
from models.time_series import StreamingARIMA_GARCH
from models.online_lgbm import OnlineLGBM
from dqn_trainer import DQNTrainer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AI-Engine-1m")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration
DEFAULT_REMOTE_IP = "192.168.3.8:50051" 
_env_url = os.environ.get("CLOUD_BRIDGE_URL", DEFAULT_REMOTE_IP)
CLOUD_BRIDGE_URL = _env_url.replace("http://", "").replace("https://", "")
# Global risk toggles
RISK_OFF = os.environ.get("RISK_OFF", "false").lower() == "true"
MAX_VOL_MULT = float(os.environ.get("MAX_VOL_MULT", "1.5"))  # tighter cap vs 2.0
REJECT_VOL_TIER1 = float(os.environ.get("REJECT_VOL_TIER1", "10.0"))
REJECT_VOL_TIER1_SCORE = float(os.environ.get("REJECT_VOL_TIER1_SCORE", "0.3"))
REJECT_VOL_TIER2 = float(os.environ.get("REJECT_VOL_TIER2", "6.0"))
REJECT_VOL_TIER2_SCORE = float(os.environ.get("REJECT_VOL_TIER2_SCORE", "0.15"))

# Model Paths
MODEL_DIR = "/app/models" if os.path.exists("/app/models") else "ai-engine/models"
LGBM_PATH = os.path.join(MODEL_DIR, "lgbm_scalping_v2.txt")
DQN_PATH = os.path.join(MODEL_DIR, "dqn_agent.pth")

# Supabase for Feature Storage
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Core Features for QuantumNet (Global shared def)
CORE_FEATURES_QNET = ['open','high','low','close','tick_volume','ema_short','ema_long','atr','adx','rsi',
                      'center','price_vs_center','cloud_width','ema_spread','atr_percent','candle_size',
                      'wick_upper','wick_lower','log_return','volatility_5','spread_to_atr','volume_density',
                      'cloud_dist_atr','wick_ratio','volume_shock','volatility_shock','order_imbalance_proxy',
                      # Pad with others or duplicate to reach 33
                      'ema_short', 'ema_long', 'atr', 'adx', 'rsi', 'close']

def _run_feature_extraction(candles_list):
    """
    Top-level function for ProcessPool.
    Returns: (latest_row_dict, input_seq_numpy)
    """
    fe = FeatureEngineerPolars()
    df = fe.process_all(candles_list)
    
    if df.height == 0:
        return None, None

    # Get latest row as dict
    # Polars row(idx, named=True)
    latest_row = df.row(-1, named=True)
    
    # Prepare Input Seq for QuantumNet
    # Handle missing cols
    # This is slightly expensive in loop loop, but Polars select is fast.
    
    # Check explicitly for missing columns to zero-fill
    # Polars 'select' with fallback?
    # Easier: Iterate and construct list of expressions
    import polars as pl
    exprs = []
    current_cols = df.columns
    exprs = []
    current_cols = df.columns
    for i, feature in enumerate(CORE_FEATURES_QNET[:33]):
        # Check explicitly
        if feature in current_cols:
            # If duplication exists in list, we must alias uniquely if we want to include it multiple times
            # However, for input tensor, name doesn't matter as much as position, but Polars needs unique names in select.
            # We can use original column expression but alias it to ensure uniqueness if needed, 
            # OR better: just select by positional expression without forcing alias collision.
            # But wait, `pl.col(feature)` keeps the name. If I select `pl.col('ema_short')` twice, output DF has duplicate cols?
            # Polars DF doesn't support duplicate column names.
            # So we MUST alias duplicates.
            exprs.append(pl.col(feature).alias(f"feat_{i}_{feature}"))
        else:
            exprs.append(pl.lit(0.0).alias(f"feat_{i}_{feature}"))
            
    # Select last 33 rows or whatever length
    # QuantumNet expects sequence? Yes. (Batch, Seq, Features)
    # Usually Seq=64 or so.
    # Let's return the whole sequence (up to say 128 latest) to save IPC bandwidth
    
    # We need to return numpy array of shape (Seq, 33)
    # Efficient select
    df_lite = df.select(exprs).tail(128)
    input_seq = df_lite.to_numpy().astype(np.float32)
    
    return latest_row, input_seq

# Log Path
DECISION_LOG_PATH = "/app/ai_decisions.csv" if os.path.exists("/app") else "ai-engine/ai_decisions.csv"

class DecisionLogger:
    def __init__(self, filename=DECISION_LOG_PATH):
        self.filename = filename
        # Ensure dir exists if path includes dir
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("timestamp,symbol,action,price,conf_qnet,conf_lgbm,conf_dqn,final_vote,latency_ms,dom_imb\n")

    def log(self, symbol, action, price, q_conf, l_conf, d_conf, vote, latency, dom_imb=0.0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.filename, 'a') as f:
                f.write(f"{timestamp},{symbol},{action},{price},{q_conf:.4f},{l_conf:.4f},{d_conf:.4f},{vote},{latency:.2f},{dom_imb:.4f}\n")
        except Exception as e:
            logger.error(f"Failed to write to decision log: {e}")

def numpy_converter(obj):
    """Helper to serialize numpy/pandas types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class LocalAIEngine:
    def __init__(self):
        self.channel = None
        self.stub = None
        self.is_connected = False
        self.decision_logger = DecisionLogger(filename=DECISION_LOG_PATH)
        self.process_executor = ProcessPoolExecutor(max_workers=2) # 2 workers for features
        
        # Device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info("🚀 AI Engine Initialized on " + str(self.device) + " [v2.5.0 AI-Kelly Release]")
        
        # DEBUG: List models
        try:
            logger.info(f"📂 Model Dir: {MODEL_DIR}")
            logger.info(f"📂 Files found: {os.listdir(MODEL_DIR)}")
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")

        # === Initialize Ensemble Models ===
        
        # 1. QuantumNet (PyTorch)
        self.quantum = QuantumNetLite(input_dim=33, hidden_dim=96).to(self.device)
        self.quantum.eval()
        
        # 2. LightGBM (Online)
        self.lgbm = OnlineLGBM(model_dir=MODEL_DIR, default_model_path=LGBM_PATH)
        
        # 3. DQN Agent
        # Input dim depends on feature set size, approx 40-50
        self.dqn = DQNAgent(input_dim=50, device=self.device, model_path=DQN_PATH) 
        
        # 4. Time Series (ARIMA-GARCH)
        # Dictionary to hold state per symbol
        self.ts_models = {} 
        
        # Pending Experiences for DQN Training
        self.pending_experiences = {} # {request_id: (state, action, timestamp)}
        
        # Thread Pool for Parallel Execution (for IO/Threads)
        from concurrent.futures import ThreadPoolExecutor
        self.thread_executor = ThreadPoolExecutor(max_workers=4)

        # 5. Initialize DQN Trainer (Background Loop)
        self.trainer = DQNTrainer(self.dqn, self)
        
        # 6. Supabase Client for Feature Storage
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("✅ Supabase connected for feature storage")
            except Exception as e:
                logger.warning(f"⚠️ Supabase connection failed: {e}")

    # ... (other methods)

    def _store_ai_features(self, symbol: str, timestamp: int, features: dict, action: str, score: float):
        """Store computed AI features to Supabase for training pipeline."""
        if not self.supabase:
            return
        
        try:
            # Generate deterministic signal_id from symbol and timestamp
            signal_id = f"{symbol}_{timestamp}"
            
            # Use raw features, let json.dumps handle conversion via default
            ai_features = features.copy()
            
            # Add model scores
            ai_features['ai_score'] = float(score)
            ai_features['ai_action'] = action
            
            # Use UPSERT to create or update the record
            # This solves the race condition where AI Engine processes before Bridge creates the record
            signal_data = {
                "signal_id": signal_id,
                "symbol": symbol,
                "action": action,
                "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                "signal_price": float(features.get('price', 0.0)),
                "sl": float(features.get('sl', 0.0)),
                "tp": float(features.get('tp', 0.0)),
                "ai_features": json.dumps(ai_features, default=numpy_converter)
            }
            
            action_upper = action.upper()
        
            # Determine target table based on action
            # - BUY/SELL -> 'training_signals' (Real Trades)
            # - WAIT/SCAN -> 'market_scans' (Negative/Potential Samples)
            target_table = "training_signals"
            if action_upper in ["WAIT", "SCAN"]:
                target_table = "market_scans"

            self.supabase.table(target_table) \
                .upsert(signal_data, on_conflict="signal_id") \
                .execute()
            
            logger.info(f"📊 Stored {len(ai_features)} AI features for {signal_id} in {target_table}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store AI features: {e}")

    def _check_reclaim_signal(self, features: dict) -> tuple[str, float]:
        """
        Rule-based Reclaim Strategy (Migrated from Bridge).
        Logic:
        1. RECLAIM_BUY: Price crosses ABOVE Center line AND Trend is UP (EMA Short > Long)
        2. RECLAIM_SELL: Price crosses BELOW Center line AND Trend is DOWN
        
        Returns: (Action, Score)
        """
        price = features.get('price', 0.0)
        center = features.get('center', 0.0)
        ema_short = features.get('ema_short', 0.0)
        ema_long = features.get('ema_long', 0.0)
        
        # Trend Direction
        trend_up = ema_short > ema_long
        trend_down = ema_short < ema_long
        
        # Center Distance threshold (e.g. within 0.1% or just crossed)
        # Using price_vs_center from features: (Price - Center) / ATR
        price_vs_center = features.get('price_vs_center', 0.0)
        
        # Reclaim State (from feature engineering): 
        # 1 = Price > Center, -1 = Price < Center
        # is_reclaim = 1 means we just crossed? 
        # Let's rely on 'is_reclaim_signal' feature if available, else derive it.
        is_reclaim = features.get('is_reclaim_signal', 0.0) == 1.0
        
        if is_reclaim:
            if price > center and trend_up:
                return "RECLAIM_BUY", 1.0
            elif price < center and trend_down:
                return "RECLAIM_SELL", 1.0
                
        return None, 0.0

    async def connect(self):
        logger.info(f"Connecting to Cloud Bridge at {CLOUD_BRIDGE_URL}...")
        options = [
            ('grpc.keepalive_time_ms', 300000),
            ('grpc.keepalive_timeout_ms', 20000),
            ('grpc.keepalive_permit_without_calls', 1)
        ]
        self.channel = grpc.aio.insecure_channel(CLOUD_BRIDGE_URL, options=options)
        self.stub = alphaos_pb2_grpc.AlphaZeroStub(self.channel)
        
        try:
            await asyncio.wait_for(self.channel.channel_ready(), timeout=10.0)
            logger.info("✅ Connected to Cloud Bridge.")
        except Exception as e:
            logger.warning(f"⚠️ Connection handshake failed: {e}")

    async def run(self):
        # Start Trainer Loop
        asyncio.create_task(self.trainer.start())
        
        await self.connect()
        while True:
            try:
                response_queue = asyncio.Queue()
                
                async def request_generator():
                    logger.info("🤝 Sending Handshake...")
                    yield alphaos_pb2.SignalResponse(
                        request_id="HANDSHAKE",
                        client_id="alphaos-1m-ensemble",
                        should_execute=False,
                        action="PING",
                        confidence=0.0,
                        reason="Init"
                    )
                    while True:
                        response = await response_queue.get()
                        yield response

                logger.info("🎧 Waiting for inference requests...")
                async for signal_req in self.stub.StreamSignals(request_generator()):
                    # Process
                    response = await self.process_signal(signal_req)
                    await response_queue.put(response)
            
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(5)
                await self.connect()

    def _get_ts_model(self, symbol):
        if symbol not in self.ts_models:
            self.ts_models[symbol] = StreamingARIMA_GARCH()
        return self.ts_models[symbol]

    def complete_experience(self, request_id, pnl, volatility=None):
        """
        Called by DQNTrainer when a trade is closed.
        """
        if request_id in self.pending_experiences:
            data = self.pending_experiences.pop(request_id)
            if len(data) == 5:
                state, action, timestamp, symbol, atr = data
            else:
                # Legacy fallback
                state, action, timestamp = data[:3]
                atr = 1.0
            
            # Use stored ATR for normalization if volatility not passed
            vol = volatility if volatility else atr
            
            # Calculate Reward (Sharpe-like)
            # Reward = PnL / (Volatility + epsilon)
            reward = pnl / (vol + 1e-8)
            
            # Time Penalty: -0.001 per minute held
            # Fix: timestamp is from time.time() in pending_expr, so this is correct.
            # (Note: client.py fix was for features, DQN uses system time for training loop which is OK)
            hold_time = time.time() - timestamp
            time_penalty = -0.001 * (hold_time / 60.0)
            
            total_reward = reward + time_penalty
            
            # For scalping, we treat close as terminal state for this 'episode'
            next_state = np.zeros_like(state)
            done = True
            
            # Push to Replay Buffer
            self.dqn.memory.push(state, action, total_reward, next_state, done)
            
            logger.info(f"🎓 DQN Experience Added: ReqID={request_id[:8]} Reward={total_reward:.4f} (PnL={pnl:.2f})")
            return True
        return False

    async def process_signal(self, request: alphaos_pb2.SignalRequest) -> alphaos_pb2.SignalResponse:
        start_time = time.time()
        symbol = request.symbol
        
        # Risk-off master switch
        if RISK_OFF:
            return alphaos_pb2.SignalResponse(
                request_id=request.request_id,
                client_id="alphaos-1m",
                should_execute=False,
                action="SCAN",
                confidence=0.0,
                adjusted_sl=0.0,
                adjusted_tp=0.0,
                reason="RiskOff enabled"
            )
        
        if len(request.market_context) < 50:
            return alphaos_pb2.SignalResponse(request_id=request.request_id, should_execute=False, action="SCAN", reason="Warmup: Insufficient Data (<50)")

        # 1. Data Prep & Feature Engineering (ASYNC / PROCESS POOL)
        candles_list = []
        for c in request.market_context:
            candles_list.append({
                'time': c.time, 
                'open': c.open, 
                'high': c.high, 
                'low': c.low, 
                'close': c.close, 
                'tick_volume': c.volume,
                # Unpack new microstructure fields from Proto
                'real_volume': getattr(c, 'volume_real', 0),
                'tick_count': getattr(c, 'tick_count', 0),
                'aggressor_buy_vol': getattr(c, 'aggressor_buy', 0),
                'aggressor_sell_vol': getattr(c, 'aggressor_sell', 0)
            })

        # Run Heavy Calculation in Process Pool
        loop = asyncio.get_running_loop()
        
        # 1. Feature Extraction (CPU Heavy) -> Offload
        try:
            latest_row, input_seq = await loop.run_in_executor(
                self.process_executor, 
                _run_feature_extraction, 
                candles_list
            )
        except Exception as e:
            logger.error(f"❌ Feature Engineering Failed: {e}")
            # We can't proceed without features
            return alphaos_pb2.SignalResponse(request_id=request.request_id, should_execute=False, action="WAIT", reason=f"FeatureError: {e}")
        
        if latest_row is None:
             return alphaos_pb2.SignalResponse(request_id=request.request_id, should_execute=False, reason="Feature Error")

        # Capture Last Candle Time for Temporal Features (Fix Look-Ahead Bias)
        last_candle_time = candles_list[-1]['time']
        
        # 1. Extract Technical Context from Request (Protobuf -> Dict)
        tc = request.technical_context
        
        # Basic Features - Prefer Computed over TC (if available)
        # tc values might be 0 if source didn't compute them
        
        def get_feat(key, tc_val):
            if key in latest_row: return latest_row[key]
            return tc_val

        def clamp(val, default=0.0, low=None, high=None):
            if val is None or isinstance(val, str):
                val = default
            try:
                v = float(val)
            except Exception:
                v = default
            if np.isnan(v) or np.isinf(v):
                v = default
            if low is not None:
                v = max(low, v)
            if high is not None:
                v = min(high, v)
            return v

        current_features = {
            # Signal Data (from request)
            'price': request.suggested_entry if request.suggested_entry > 0 else latest_row['close'],
            'sl': request.suggested_sl,
            'tp': request.suggested_tp,
            
            # Core Trend Indicators
            'ema_short': get_feat('ema_short', tc.ema_short),
            'ema_long': get_feat('ema_long', tc.ema_long),
            'atr': get_feat('atr', tc.atr),
            'adx': get_feat('adx', tc.adx),
            'rsi': get_feat('rsi', tc.rsi), 
            'center': get_feat('center', tc.center),
            
            # Boolean Flags (convert to int for model)
            'distance_ok': 1 if tc.distance_ok else 0,
            'slope_ok': 1 if tc.slope_ok else 0,
            'trend_filter_ok': 1 if tc.trend_filter_ok else 0,
            'htf_trend_ok': 1 if tc.htf_trend_ok else 0,
            'volatility_ok': 1 if tc.volatility_ok else 0,
            'chop_ok': 1 if tc.chop_ok else 0,
            'spread_ok': 1 if tc.spread_ok else 0,
            
            # State & Structure
            'bars_since_last': tc.bars_since_last,
            'trend_direction': tc.trend_direction,
            'ema_cross_event': tc.ema_cross_event,
            'ema_spread': get_feat('ema_spread', tc.ema_spread),
            'atr_percent': get_feat('atr_percent', tc.atr_percent),
            'reclaim_state': tc.reclaim_state,
            'is_reclaim_signal': 1 if tc.is_reclaim_signal else 0,
            'price_vs_center': get_feat('price_vs_center', tc.price_vs_center),
            'cloud_width': get_feat('cloud_width', tc.cloud_width),
            
            # Microstructure (from EA or Computed)
            'tick_volume': get_feat('tick_volume', tc.tick_volume), 
            'spread': get_feat('spread', tc.spread),
            'candle_size': get_feat('candle_size', tc.candle_size),
            'wick_upper': get_feat('wick_upper', tc.wick_upper),
            'wick_lower': get_feat('wick_lower', tc.wick_lower),
            
            # New Microstructure (Raw from EA via DF)
            'real_volume': latest_row.get('real_volume', 0),
            'tick_count': latest_row.get('tick_count', 0),
            'aggressor_buy_vol': latest_row.get('aggressor_buy_vol', 0),
            'aggressor_sell_vol': latest_row.get('aggressor_sell_vol', 0),
            
            # Derived Microstructure
            'volume_shock': latest_row.get('volume_shock', 0),
            'volatility_shock': latest_row.get('volatility_shock', 0),
            'order_imbalance_proxy': latest_row.get('order_imbalance_proxy', 0),
            'volatility_skew': latest_row.get('volatility_skew', 0), # If computed
        }
        
        # Add Time Features (Fix: Use Candle Time instead of System Time)
        # last_candle_time is Unix Timestamp (seconds)
        current_dt = pd.to_datetime(last_candle_time, unit='s')
        current_features['hour'] = current_dt.hour
        current_features['day_of_week'] = current_dt.dayofweek

        # Add Derived Features (v2 Model)
        atr = current_features.get('atr', 0.0001)  # Use small default to avoid div by zero
        spread = current_features.get('spread', 0.0)
        tick_vol = current_features.get('tick_volume', 0)
        candle_size = current_features.get('candle_size', 0.0001)
        pvc = current_features.get('price_vs_center', 0.0)
        wick_u = current_features.get('wick_upper', 0.0)
        wick_l = current_features.get('wick_lower', 0.0)
        
        # 1. spread_to_atr
        current_features['spread_to_atr'] = spread / (atr + 1e-8)
        
        # 2. volume_density
        current_features['volume_density'] = tick_vol / (candle_size + 1e-8)
        
        # 3. cloud_dist_atr (normalized price distance from center)
        current_features['cloud_dist_atr'] = abs(pvc) / (atr + 1e-8)
        
        # 4. wick_ratio
        body = abs(candle_size - wick_u - wick_l)
        current_features['wick_ratio'] = (wick_u + wick_l) / (body + 1e-8)
        
        # === PHASE 7: JUMP TRADING SYNTHETIC FEATURES (L1 Data Optimization) ===
        # -----------------------------------------------------------------------
        
        # 5. Absorption Ratio (Volume / Price Change)
        # Logic: High Volume + Small Range = Hidden Limit Orders (Absorption/Distribution)
        # We normalize by ATR to make it scale-invariant
        price_range = latest_row['high'] - latest_row['low']
        current_features['absorption_ratio'] = tick_vol / (price_range + 1e-8)
        
        # 6. Tick Velocity Ratio (Momentum of Activity)
        # Logic: Current Volume vs Recent Average. Spikes precede breakouts.
        # We need rolling average, assuming 'tick_vol_MA5' might be in df, if not compute on fly?
        # For now, simplistic approximation: compare to tick_count (density) or valid if passed from scanner
        # Let's use a proxy: tick_volume / 1000 (normalized) if no history. 
        # Better: use tick_count directly if available.
        # Let's rely on 'tick_count' provided by MT5.
        current_features['tick_velocity'] = current_features.get('tick_count', 0) / 60.0 # Ticks per sec for 1m bar
        
        # 7. Tick Imbalance Proxy (Price Action based)
        # Logic: If Close is near High, Buying dominated.
        # Formula: (Close - Open) / (High - Low) * Volume
        candle_direction = (latest_row['close'] - latest_row['open']) / (price_range + 1e-8)
        current_features['tick_imbalance_proxy'] = candle_direction * tick_vol
        
        # === PHASE 7: CITADEL RISK CONTROLS ===
        # -------------------------------------
        
        # 8. Spread Protection (Liquidity Check)
        # If Spread > 0.2 ATR, liquidity is too thin for safe scalping
        # Use a soft flag 'liquidity_ok'
        spread_threshold = 0.2 * atr
        current_features['liquidity_ok'] = 1.0 if spread < spread_threshold else 0.0
        
        # 9. Stalemate Indicator (Time-Based Exit Proxy)
        # Identify if price is chopping (ADX < 15) AND Volume is dropping
        # This will be used in decision logic to force WAIT or CLOSE
        adx_val = current_features.get('adx', 20.0)
        current_features['is_stalemate'] = 1.0 if (adx_val < 15 and current_features['tick_velocity'] < 5.0) else 0.0

        # Sanitize Microstructure / DOM ranges to avoid explosions across symbols
        sanitize_bounds = {
            'volume_shock': (0.0, 50.0),
            'volatility_shock': (0.0, 50.0),
            'order_imbalance_proxy': (-1e7, 1e7),
            'volume_density': (-1e5, 1e5),
            'spread_to_atr': (0.0, 200.0),
            'cloud_dist_atr': (0.0, 1e3),
            'wick_ratio': (0.0, 50.0),
            'dom_imbalance': (-1.0, 1.0),
            'avg_dom_imbalance': (-1.0, 1.0),
            'best_bid_vol': (0.0, 1e7),
            'best_ask_vol': (0.0, 1e7),
            'aggressor_buy_vol': (0.0, 1e7),
            'aggressor_sell_vol': (0.0, 1e7),
            'tick_volume': (0.0, 1e9),
            'real_volume': (0.0, 1e9),
            'tick_count': (0.0, 1e9),
            # New Phase 7 Bounds
            'absorption_ratio': (0.0, 1e9),
            'tick_velocity': (0.0, 1000.0),
            'tick_imbalance_proxy': (-1e9, 1e9),
        }
        for k, bounds in sanitize_bounds.items():
            low, high = bounds
            current_features[k] = clamp(current_features.get(k, 0.0), 0.0, low, high)
        
        # If all filters are zero, emit a warning for upstream debugging
        filter_keys = ['distance_ok', 'trend_filter_ok', 'volatility_ok', 'spread_ok']
        if sum([current_features.get(k, 0) for k in filter_keys]) == 0:
            logger.warning("⚠️ Filters all zero; check scanner/bridge inputs for validity.")

        # 5. DOM Features (New)
        # From Proto (if available)
        dom_bids = request.dom_bids
        dom_asks = request.dom_asks
        
        # Calculate DOM Imbalance if not provided in TC or if we want fresh calc
        # (Bid - Ask) / (Bid + Ask)
        total_bid_vol = sum([b.volume for b in dom_bids])
        total_ask_vol = sum([a.volume for a in dom_asks])
        
        if total_bid_vol + total_ask_vol > 0:
            current_features['dom_imbalance'] = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        else:
            current_features['dom_imbalance'] = tc.dom_imbalance # Fallback to TC
            
        current_features['best_bid_vol'] = dom_bids[0].volume if dom_bids else tc.best_bid_vol
        current_features['best_ask_vol'] = dom_asks[0].volume if dom_asks else tc.best_ask_vol
        # Derived DOM
        current_features['dom_pressure'] = current_features['dom_imbalance'] 
        # Map for model compatibility if needed (avg_dom_imbalance -> dom_imbalance)
        current_features['avg_dom_imbalance'] = current_features['dom_imbalance']

        # === 日志输出: 特征提取结果 ===
        logger.info(f"   📊 Features Extracted:")
        logger.info(f"      💰 Signal: P={current_features['price']:.5f} SL={current_features['sl']:.5f} TP={current_features['tp']:.5f}")
        logger.info(f"      📈 Techs: EMA={current_features['ema_short']:.5f}/{current_features['ema_long']:.5f} ATR={current_features['atr']:.5f} RSI={current_features['rsi']:.1f}")
        logger.info(f"      ✅ Filters: Dist={current_features['distance_ok']} Trend={current_features['trend_filter_ok']} Vol={current_features['volatility_ok']} Spr={current_features['spread_ok']}")
        logger.info(f"      🔢 Derived: Spr/ATR={current_features['spread_to_atr']:.2f} VolDen={current_features['volume_density']:.2f} CldDist={current_features['cloud_dist_atr']:.2f}")
        logger.info(f"      🏛️ DOM: Imb={current_features['dom_imbalance']:.2f} BidV={current_features['best_bid_vol']:.0f} AskV={current_features['best_ask_vol']:.0f}")
        logger.info(f"      ⚡ Micro: VolShock={current_features['volume_shock']:.2f} AggBuy={current_features['aggressor_buy_vol']:.0f} AggSell={current_features['aggressor_sell_vol']:.0f}")
        
        # Prepare Input for QuantumNet (Sequence)
        # Replacing DF logic with pre-computed input_seq from worker
        input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(self.device) # (1, Seq, 33)
        
        # 2. Parallel Inference
        loop = asyncio.get_running_loop()
        
        # A. QuantumNet
        async def run_quantum():
            with torch.no_grad():
                policy, val = self.quantum(input_tensor)
                # Policy: [Wait, Buy, Sell]
                p = policy.cpu().numpy()[0]
                return p, val.item()

        # B. LightGBM
        async def run_lgbm():
            # Defined feature list MATCHING training_filter.py EXACTLY
            # Missing features will be defaulted to 0
            lgbm_features = [
                'open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread', 'tick_count', 
                'aggressor_buy_vol', 'aggressor_sell_vol', 'avg_dom_imbalance', 'volatility_skew_proxy', 
                'dom_imbalance', 'volatility_skew', 'ema_short', 'ema_long', 'atr', 'adx', 'rsi', 'center', 
                'price_vs_center', 'cloud_width', 'ema_spread', 'atr_percent', 'candle_size', 'wick_upper', 
                'wick_lower', 'log_return', 'volatility_5', 'spread_to_atr', 'volume_density', 
                'cloud_dist_atr', 'wick_ratio', 'volume_shock', 'volatility_shock', 'order_imbalance_proxy'
            ]
            
            # Construct dict for DataFrame
            lgbm_input_dict = {k: [current_features.get(k, 0.0)] for k in lgbm_features}
            lgbm_df = pd.DataFrame(lgbm_input_dict)
            
            return self.lgbm.predict(lgbm_df, symbol)[0]

        # C. ARIMA-GARCH (Update & Predict)
        async def run_ts():
            ts_model = self._get_ts_model(symbol)
            # Update with latest return
            # Assuming log_return is in df
            ret = latest_row.get('log_return', 0.0)
            ts_model.update_return(ret)
            pred_r, pred_vol = ts_model.predict_next()
            return pred_r, pred_vol

        # D. DQN
        async def run_dqn():
            # Optimized Feature Selection for 1-5m Scalping (Target 50 dim)
            # Prioritizing Microstructure, Momentum, and Volatility
            dqn_features = [
                # 1. Microstructure (High Priority for Scalping)
                'volume_shock', 'order_imbalance_proxy', 'volume_density', 'dom_imbalance',
                'wick_ratio', 'wick_upper', 'wick_lower', 
                
                # 2. Momentum & Trend
                'rsi', 'adx', 'log_return', 'price_vs_center', 'ema_spread', 'trend_direction',
                'center', 'cloud_width', 'cloud_dist_atr',
                
                # 3. Volatility & Risk
                'atr', 'atr_percent', 'spread_to_atr', 'candle_size', 'spread',
                
                # 4. State Flags
                'distance_ok', 'volatility_ok', 'spread_ok', 'trend_filter_ok', 'chop_ok',
                'reclaim_state', 'is_reclaim_signal', 'bars_since_last',
                
                # 5. Raw & Context
                'close', 'tick_volume', 'ema_short', 'ema_long',
                'hour', 'day_of_week',
                'sl', 'tp'
            ]
            
            state_values = []
            for k in dqn_features:
                # Try current_features first (has derived + context), then latest_row (has raw dataframe cols)
                val = current_features.get(k)
                if val is None:
                    val = latest_row.get(k, 0.0)
                state_values.append(float(val))
            
            # Pad to 50 dimensions
            remaining = 50 - len(state_values)
            if remaining > 0:
                state_values.extend([0.0] * remaining)
            
            # Ensure strictly 50
            state = np.array(state_values[:50], dtype=np.float32)
            
            # Debug: Check shape
            if state.shape != (50,):
                logger.error(f"❌ DQN State Shape Mismatch: {state.shape}")
                state = np.zeros(50, dtype=np.float32) # Fallback
                
            max_q, action_idx, _ = self.dqn.predict(state)
            return max_q, action_idx, state

        # Execute Ensemble
        results = await asyncio.gather(
            run_quantum(),
            run_lgbm(),
            run_ts(),
            run_dqn(),
            return_exceptions=True
        )
        
        q_res, l_res, ts_res, d_res = results
        
        # Unpack & Handle Errors
        q_policy, q_val = (np.array([0.33, 0.33, 0.33]), 0.0) if isinstance(q_res, Exception) else q_res
        l_mfe = 0.0 if isinstance(l_res, Exception) else l_res
        pred_ret, pred_vol = (0.0, 0.001) if isinstance(ts_res, Exception) else ts_res
        
        dqn_conf = 0.0
        dqn_action_idx = 0
        dqn_state = None
        if isinstance(d_res, Exception):
            pass # Keep defaults
        else:
            dqn_conf, dqn_action_idx, dqn_state = d_res
        
        
        # 3. Dynamic Regime Switching & Model Averaging
        # ---------------------------------------------
        # Determine Market Regime
        adx = current_features.get('adx', 20.0)
        atr_pct = current_features.get('atr_percent', 0.1)
        
        # Thresholds
        TREND_ADX_THRESHOLD = 25.0
        RANGE_ADX_THRESHOLD = 20.0
        HIGH_VOL_THRESHOLD = 0.2 # 0.2% per bar
        
        regime = "NEUTRAL"
        
        # Default Weights (Balanced)
        w_q = 0.40 # QuantumNet (Trend/Sequence)
        w_l = 0.30 # LightGBM (Pattern/Level)
        w_d = 0.20 # DQN (Scalping/Reactive)
        w_t = 0.10 # TimeSeries (Mean Rev)
        
        if atr_pct > HIGH_VOL_THRESHOLD:
            regime = "VOLATILE"
            # In high vol, trust reactive models (DQN) and strict levels (LGBM) more than Sequence (Quantum)
            # Reduce confidence overall effectively
            w_q = 0.20
            w_l = 0.30
            w_d = 0.40 # DQN handles noise/reward better
            w_t = 0.10
            
        elif adx > TREND_ADX_THRESHOLD:
            regime = "TREND"
            # Trust Trend Follower (QuantumNet)
            w_q = 0.60
            w_l = 0.20
            w_d = 0.10
            w_t = 0.10
            
        elif adx < RANGE_ADX_THRESHOLD:
            regime = "RANGE"
            # Trust Mean Reversion components
            w_q = 0.20
            w_l = 0.30
            w_d = 0.30
            w_t = 0.20 # Boost TimeSeries (Mean Rev)

        # Normalize signals to [-1, 1] (Sell < 0 < Buy)
        # Quantum: Buy prob - Sell prob
        s_q = q_policy[1] - q_policy[2] 
        
        # LGBM: Use own directional hint (ema_short - ema_long), fallback to price_vs_center
        dir_hint = np.sign(current_features.get('ema_short', 0) - current_features.get('ema_long', 0))
        if dir_hint == 0:
            dir_hint = np.sign(current_features.get('price_vs_center', 0))
        s_l = (l_mfe / 5.0) * dir_hint  # Normalize MFE~5 to ~1
        s_l = float(np.clip(s_l, -1.0, 1.0)) # Clip LGBM
        
        # DQN: Use predicted action for direction (0 wait, 1 buy, 2 sell)
        dqn_dir = 0.0
        if dqn_action_idx == 1:
            dqn_dir = 1.0
        elif dqn_action_idx == 2:
            dqn_dir = -1.0
        s_d = dqn_dir * (dqn_conf / 10.0) # Normalize Q~10 to 1
        s_d = float(np.clip(s_d, -1.0, 1.0)) # Clip DQN
        
        # TS: Pred Return
        # Normalize and Clip to prevent score explosion
        s_t = np.sign(pred_ret) * (abs(pred_ret) / 0.001) 
        s_t = float(np.clip(s_t, -1.0, 1.0))
            
        # Combined Score
        final_score = w_q * s_q + w_l * s_l + w_d * s_d + w_t * s_t
        
        # DEBUG: Check for score explosion
        if abs(final_score) > 10:
            logger.warning(f"⚠️ Score Explosion Detected!")
            logger.warning(f"   Q={s_q:.4f} (w={w_q})")
            logger.warning(f"   L={s_l:.4f} (raw_mfe={l_mfe:.4f}, w={w_l})")
            logger.warning(f"   D={s_d:.4f} (raw_conf={dqn_conf:.4f}, w={w_d})")
            logger.warning(f"   T={s_t:.4f} (raw_ret={pred_ret:.6f}, w={w_t})")
            
        # Log Regime
        logger.info(f"      🌍 Regime: {regime} (ADX={adx:.1f} ATR%={atr_pct:.2f}) -> Weights: Q={w_q} L={w_l} D={w_d} T={w_t}")
        
        # 4. Decision Logic
        threshold = 0.15 # Lowered to allow more trades
        should_execute = abs(final_score) > threshold
        
        # 5. Integrate Reclaim Logic (Priority Override)
        # ---------------------------------------------
        reclaim_action, reclaim_score = self._check_reclaim_signal(current_features)
        
        # Decide Action
        threshold = 0.15 # Minimum confidence
        final_action = "WAIT"
        
        # === RECLAIM OVERRIDE ===
        if reclaim_action and not RISK_OFF:
            final_action = reclaim_action
            final_score = reclaim_score # Force high score
            logger.info(f"🚀 RECLAIM Signal Triggered: {reclaim_action} (Price={current_features.get('close')} vs Center={current_features.get('center')})")
        # ========================
        elif final_score > threshold:
            final_action = "BUY"
        elif final_score < -threshold:
            final_action = "SELL"
            
        should_execute = (final_action != "WAIT")
        
        # 5. Dynamic Risk (VaR based)
        # Use ARIMA predicted vol
        atr = latest_row.get('atr', 0.0001)
        close_price = latest_row['close']
        
        # Volatility Ratio: Predicted Vol (GARCH) / Current Vol (ATR)
        current_vol_pct = atr / close_price
        # Avoid division by zero
        if current_vol_pct < 1e-6: current_vol_pct = 1e-6
        
        # Clamp predicted vol to avoid runaway when ARIMA is unstable
        max_vol_pct = current_vol_pct * 6.0
        pred_vol = float(np.clip(pred_vol, 0.0, max_vol_pct))
        raw_vol_multiplier = pred_vol / current_vol_pct
        
        # Cap multiplier to prevent excessive stops (tighter cap)
        vol_multiplier = max(1.5, min(raw_vol_multiplier, MAX_VOL_MULT))
        
        # Risk Management (SL/TP)
        # Basic: 1.5x ATR
        base_sl_pips = atr * vol_multiplier
        base_tp_pips = atr * vol_multiplier * 1.5
        
        # Adjust TP by confidence (Capped at 3x boost to prevent unrealistic targets)
        # For scalping, rarely exceed 5R. 
        confidence_mult = min(3.0, 1 + abs(final_score))
        adj_tp_pips = base_tp_pips * confidence_mult
        
        # Log Risk Calculation Details
        logger.info(f"      🛡️ Dynamic Risk: BaseSL={base_sl_pips:.1f}p | VolMult={vol_multiplier:.2f} (Raw={raw_vol_multiplier:.2f}) | AdjSL={base_sl_pips:.1f}p TP={adj_tp_pips:.1f}p")
        
        # Protection: tiered rejection to reduce False Positives
        # Skip this check if it's a Reclaim signal (Trust Reclaim Rule)
        if not reclaim_action:
            # 1. Spread Protection (Citadel Risk)
            if current_features.get('liquidity_ok', 1.0) == 0.0:
                 should_execute = False
                 logger.warning(f"🛑 Signal Rejected: Poor Liquidity (Spread > 0.2 ATR)")

            # 2. Stalemate/Chop Protection
            elif current_features.get('is_stalemate', 0.0) == 1.0:
                 should_execute = False
                 logger.warning(f"🛑 Signal Rejected: Market Stalemate (Low ADX & Low Velocity)")

            # 3. Volatility Check (Existing)
            elif raw_vol_multiplier > REJECT_VOL_TIER1 and abs(final_score) < REJECT_VOL_TIER1_SCORE:
                should_execute = False
                logger.warning(f"🛑 Signal Rejected: Extreme Volatility Risk (RawMult={raw_vol_multiplier:.1f} > {REJECT_VOL_TIER1}) with Low Confidence (|Score|={abs(final_score):.2f} < {REJECT_VOL_TIER1_SCORE})")
            elif raw_vol_multiplier > REJECT_VOL_TIER2 and abs(final_score) < REJECT_VOL_TIER2_SCORE:
                should_execute = False
                logger.warning(f"🛑 Signal Rejected: High Volatility (RawMult={raw_vol_multiplier:.1f} > {REJECT_VOL_TIER2}) with Low Score (|Score|={abs(final_score):.2f} < {REJECT_VOL_TIER2_SCORE})")
        
        # Convert to Price
        cp = close_price
        if final_action == "BUY":
            sl_price = cp - base_sl_pips
            tp_price = cp + adj_tp_pips
        else:
            sl_price = cp + base_sl_pips
            tp_price = cp - adj_tp_pips

        latency = (time.time() - start_time) * 1000
        
        # Log
        dom_imb = current_features.get('dom_imbalance', 0.0)
        self.decision_logger.log(symbol, final_action, cp, s_q, s_l, s_d, final_score, latency, dom_imb)
        
        # Update features with decision results for storage
        current_features['sl'] = sl_price
        current_features['tp'] = tp_price
        
        # Store AI features to Supabase for training
        signal_timestamp = int(latest_row.get('time', time.time()))
        self._store_ai_features(symbol, signal_timestamp, current_features, final_action, final_score)
        
        logger.info(f"🤖 {symbol} {final_action} | Score: {final_score:.2f} | DOM: {dom_imb:.2f} | SL: {sl_price:.5f} TP: {tp_price:.5f} | Latency: {latency:.1f}ms")
        
        # Record Pending Experience for DQN Training
        if should_execute and dqn_state is not None:
             # Action mapping: WAIT=0, BUY=1, SELL=2
             action_idx = 1 if final_action == "BUY" else 2 
             if final_action == "SELL": action_idx = 2
             
             # Store context for training
             atr = latest_row.get('atr', 1.0)
             self.pending_experiences[request.request_id] = (dqn_state, action_idx, time.time(), symbol, atr)

        return alphaos_pb2.SignalResponse(
            request_id=request.request_id,
            client_id="alphaos-1m",
            should_execute=should_execute,
            action=final_action,
            confidence=float(abs(final_score)),
            adjusted_sl=float(sl_price),
            adjusted_tp=float(tp_price),
            reason=f"Score: {final_score:.2f} (Q={s_q:.2f} L={s_l:.2f})"
        )

if __name__ == "__main__":
    engine = LocalAIEngine()
    asyncio.run(engine.run())
