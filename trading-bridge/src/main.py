from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import time
import os
from supabase import create_client, Client
import json
import glob
import pandas as pd
from datetime import datetime, timezone

# gRPC Imports
from grpc_server import start_grpc_server, AlphaZeroService
# Note: these are generated at runtime in the container
import alphaos_pb2

app = FastAPI(title="MT5 Trading Bridge (HTTP)")

# 简单的内存队列和存储
command_queue = []
last_status = {}
active_symbols = {} # { "EURUSD": { "bid": 1.1, "ask": 1.2, "last_seen": 1234567890 } }
last_trade = None  # Stores the most recent trade execution

# 历史数据请求存储 { request_id: asyncio.Future }
history_requests: Dict[str, asyncio.Future] = {}
# DOM 数据请求存储 { request_id: asyncio.Future }
dom_requests: Dict[str, asyncio.Future] = {}

# 信号文件目录 (根据 MT5 实际安装位置可能需要调整，或通过 ENV 传入)
# Docker 环境中应挂载到 /app/signals
SIGNAL_DIR = os.environ.get("SIGNAL_DIR", "/app/signals")

# Global gRPC Service Instance
grpc_service = AlphaZeroService()

class TradeRequest(BaseModel):
    action: str # BUY, SELL, CLOSE
    symbol: Optional[str] = None
    volume: Optional[float] = 0.01
    sl: Optional[float] = 0.0
    tp: Optional[float] = 0.0
    ticket: Optional[int] = 0
    type: Optional[str] = "TRADE" # TRADE or PENDING
    price: Optional[float] = 0.0 # For Pending Orders

class StatusUpdate(BaseModel):
    account: Optional[dict] = None  # { balance, equity, margin, free_margin }
    positions: Optional[List[dict]] = None # [{ ticket, symbol, type, volume, pnl, sl, tp ... }]
    # Legacy support
    symbol: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    period: Optional[str] = None
    
    # DOM Summary (New)
    dom: Optional[Dict[str, float]] = None # { imbalance, best_bid_vol, best_ask_vol }
    quotes: Optional[List[Dict[str, Any]]] = None # Market Watch Quotes

class TradeReport(BaseModel): 
    ticket: int
    symbol: str
    type: str  # "BUY" or "SELL"
    volume: float
    price: float
    time: str
    entry: Optional[str] = "IN" # "IN", "OUT", "INOUT"
    position_id: Optional[int] = 0
    profit: Optional[float] = 0.0
    commission: Optional[float] = 0.0
    swap: Optional[float] = 0.0
    mae: Optional[float] = 0.0
    mfe: Optional[float] = 0.0

class HistoryData(BaseModel):
    request_id: str
    symbol: str
    timeframe: str
    data: List[Dict[str, Any]] # List of candles
    count: int

class DOMLevel(BaseModel):
    price: float
    volume: int
    volume_real: float

class DOMData(BaseModel):
    request_id: str
    symbol: str
    count: int
    bids: List[DOMLevel]
    asks: List[DOMLevel]

class SignalPayload(BaseModel):
    symbol: str
    action: str # "BUY" or "SELL"
    price: float
    sl: Optional[float] = 0.0
    tp: Optional[float] = 0.0
    source: Optional[str] = "http_api"
    comment: Optional[str] = ""

# --- Inference Models ---
class CandleData(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: int
    # New Microstructure Fields (Optional for backward compatibility)
    rv: Optional[float] = 0.0 # Real Volume
    tc: Optional[int] = 0     # Tick Count
    ab: Optional[float] = 0.0 # Aggressor Buy
    as_: Optional[float] = 0.0 # Aggressor Sell (using as_ because 'as' is reserved)

    class Config:
        fields = {'as_': 'as'} # Map JSON 'as' to 'as_'

class DOMLevelSimple(BaseModel):
    p: float
    v: int

class DOMDataSimple(BaseModel):
    bids: List[DOMLevelSimple]
    asks: List[DOMLevelSimple]

class InferencePayload(BaseModel):
    type: str
    symbol: str
    timeframe: str
    action: str
    candles: List[CandleData]
    dom: DOMDataSimple
    ask: float
    bid: float

# --- Automation Manager ---
class AutomationManager:
    def __init__(self):
        self.rules = {} # { "SYMBOL": rule_dict, "GLOBAL": rule_dict }
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase = None
        self.last_sync = 0
        
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                print(f"Failed to init Supabase in AutomationManager: {e}")

    async def sync_rules(self):
        if not self.supabase: return
        
        # Sync every 10 seconds
        if time.time() - self.last_sync < 10:
            return
        
        try:
            response = self.supabase.table("automation_rules").select("*").eq("is_enabled", True).execute()
            new_rules = {}
            for rule in response.data:
                new_rules[rule['symbol']] = rule
            self.rules = new_rules
            self.last_sync = time.time()
            # print(f"DEBUG: Synced {len(self.rules)} automation rules from Supabase")
        except Exception as e:
            print(f"Failed to sync automation rules: {e}")

    async def get_history_data(self, symbol: str, timeframe: str = "PERIOD_H1", count: int = 100):
        """
        Helper to fetch history data from MT5 via the async command queue.
        """
        request_id = str(uuid.uuid4())
        command = {
            "type": "GET_HISTORY",
            "request_id": request_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count,
            "from": 0,
            "to": 0
        }
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        history_requests[request_id] = future
        command_queue.append(command)
        
        try:
            print(f"DEBUG: Requesting history for {symbol}...")
            data = await asyncio.wait_for(future, timeout=10.0)
            return data
        except Exception as e:
            print(f"❌ Failed to fetch history for {symbol}: {e}")
            if request_id in history_requests:
                del history_requests[request_id]
            return None

    async def get_dom_data(self, symbol: str):
        """
        Helper to fetch DOM data from MT5 via the async command queue.
        """
        request_id = str(uuid.uuid4())
        command = {
            "type": "GET_DOM",
            "request_id": request_id,
            "symbol": symbol
        }
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        dom_requests[request_id] = future
        command_queue.append(command)
        
        try:
            print(f"DEBUG: Requesting DOM for {symbol}...")
            data = await asyncio.wait_for(future, timeout=5.0)
            return data
        except Exception as e:
            print(f"❌ Failed to fetch DOM for {symbol}: {e}")
            if request_id in dom_requests:
                del dom_requests[request_id]
            return None

    async def evaluate_signal(self, signal_data):
        raw_symbol = signal_data.get("symbol")
        # Normalize symbol: 
        # 1. Remove dot suffixes (e.g. EURUSD.r -> EURUSD)
        # 2. Uppercase
        normalized_upper = raw_symbol.upper().split('.')[0]
        
        symbol = normalized_upper
        rule = self.rules.get(symbol)

        # 3. If not found, try fuzzy match against loaded rules
        # e.g. Raw "GBPUSDx" (Norm "GBPUSDX") contains Rule "GBPUSD" ?
        if not rule:
             for rule_symbol in self.rules.keys():
                 if rule_symbol == "GLOBAL": continue
                 # Check if normalized symbol starts with rule symbol (e.g. GBPUSDx starts with GBPUSD)
                 # Or if rule symbol is contained in it
                 if normalized_upper.startswith(rule_symbol):
                     symbol = rule_symbol
                     rule = self.rules.get(symbol)
                     # print(f"DEBUG: Fuzzy matched {raw_symbol} -> {symbol}")
                     break
        
        # 4. Fallback to GLOBAL
        if not rule:
            rule = self.rules.get("GLOBAL")
        
        if not rule:
            return False, "No matching automation rule"
            
        if not rule.get("is_enabled"):
            return False, "Automation disabled for this rule"

        # 5. Check for existing positions (Prevent stacking/over-trading) & Handle Reversal
        # If max_positions is set (e.g. 1), and we already have open positions for this symbol, skip.
        max_pos = rule.get("max_positions", 3) 
        signal_action = signal_data.get("action", "").upper() # "BUY" or "SELL"
        
        current_positions = []
        if last_status and "positions" in last_status and last_status["positions"]:
            for pos in last_status["positions"]:
                if pos.get("symbol") == symbol:
                    current_positions.append(pos)
        
        current_count = len(current_positions)
        same_dir_count = sum(1 for p in current_positions if p.get("type", "").upper() == signal_action)
        
        # Check for Reversal: Signal is different from existing position type
        # Assuming we don't hedge (hold both BUY and SELL), so all positions should be same type.
        # If we find an opposing position, we should CLOSE it first.
        
        has_opposing_position = False
        for pos in current_positions:
            pos_type = pos.get("type", "").upper() # "BUY" or "SELL"
            # Logic: If I want to BUY, and I have SELL positions -> Opposing
            if signal_action == "BUY" and pos_type == "SELL":
                has_opposing_position = True
                self._queue_close_command(pos, f"Auto-Reverse to {signal_action}")
            elif signal_action == "SELL" and pos_type == "BUY":
                has_opposing_position = True
                self._queue_close_command(pos, f"Auto-Reverse to {signal_action}")
        
        if has_opposing_position:
            # If we closed opposing positions, we consider the "slot" freed up for the new trade.
            # So we allow the new trade to proceed (Reverse).
            return True, rule
            
        # Directional limit: DISABLED (was max 2 per direction)
        # if same_dir_count >= 2:
        #     return False, f"Max positions per direction reached ({same_dir_count}/2)"
        
        # If no opposing positions, check normal stacking limit (all positions)
        # DISABLED: Allow unlimited positions
        # if current_count >= max_pos:
        #     return False, f"Max positions reached ({current_count}/{max_pos})"
        
        return True, rule

    async def calculate_kelly_lot_size(self, symbol: str, rule: dict, confidence: float = None) -> float:
        """
        Calculate optimal lot size using Kelly Criterion.
        
        Formula: f* = (p * b - q) / b
        Where:
          p = Win probability (Historical OR AI Confidence)
          b = Win/Loss ratio (avg win ÷ avg loss)
          q = Loss probability (1 - p)
        """
        if not self.supabase:
            return float(rule.get('fixed_lot_size', 0.01))
        
        lookback = rule.get('kelly_lookback_trades', 50)
        fraction = float(rule.get('kelly_fraction', 0.25))
        max_lot = float(rule.get('max_lot_size', 1.0))
        fixed_lot = float(rule.get('fixed_lot_size', 0.01))
        
        # Get account balance from last_status
        account_balance = 10000.0  # Default
        if last_status and 'account' in last_status:
            account_balance = float(last_status['account'].get('balance', 10000))
        
        try:
            # Fetch recent closed trades for this symbol
            response = self.supabase.table('trades') \
                .select('pnl_net') \
                .eq('symbol', symbol) \
                .eq('status', 'closed') \
                .order('created_at', desc=True) \
                .limit(lookback) \
                .execute()
            
            if not response.data or len(response.data) < 10:
                print(f"📊 Kelly: Insufficient data for {symbol} ({len(response.data) if response.data else 0} trades), using fixed lot")
                return fixed_lot
            
            pnls = [float(t['pnl_net']) for t in response.data if t['pnl_net'] is not None]
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            
            if not wins or not losses:
                print(f"📊 Kelly: No win/loss data for {symbol}, using fixed lot")
                return fixed_lot
            
            # Historical Stats
            hist_win_rate = len(wins) / len(pnls)
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            
            # === AI Confidence Integration ===
            if confidence is not None and confidence > 0:
                # Use AI Confidence as 'p' (Win Probability)
                # But dampen it? Let's assume raw confidence is the estimated probability
                # Clamp between 0.1 and 0.9 to avoid extremes
                p = max(0.1, min(confidence, 0.9))
                # print(f"🤖 Kelly AI Override: WinRate {hist_win_rate:.2f} -> {p:.2f} (Conf={confidence:.2f})")
            else:
                p = hist_win_rate
                
            # Kelly calculation
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            q = 1 - p
            kelly_raw = (p * b - q) / b if b > 0 else 0
            
            # Apply fractional Kelly and bounds (cap at 25% of bankroll max)
            kelly = max(0, min(kelly_raw * fraction, 0.25))
            
            # Convert Kelly % to lot size
            # For HIGH-FREQUENCY trading: use conservative sizing
            # Higher margin_per_lot = smaller positions
            margin_per_lot = 5000.0  # $5000 per lot (conservative for HFT)
            min_lot = 0.05  # Minimum 0.05 lots for HFT
            
            risk_amount = account_balance * kelly
            lot_size = risk_amount / margin_per_lot
            lot_size = round(lot_size / 0.05) * 0.05  # Round to 0.05 increments
            
            # Apply hard cap and floor (0.05 - 1.0 for HFT)
            final_lot = min(max(lot_size, min_lot), max_lot)
            
            print(f"📊 Kelly: {symbol} p={p:.2f} (H:{hist_win_rate:.2f}) b={b:.2f} K*={kelly_raw:.3f} Kelly%={kelly*100:.1f}% Risk=${risk_amount:.0f} -> {final_lot} lots")
            return final_lot
            
        except Exception as e:
            print(f"❌ Kelly calculation error: {e}")
            return fixed_lot

    def _queue_close_command(self, pos, comment="Auto-Close"):
        """Helper to queue a CLOSE command"""
        cmd = {
            "type": "TRADE",
            "action": "CLOSE",
            "symbol": pos.get("symbol"),
            "ticket": pos.get("ticket"),
            "volume": pos.get("volume"),
            "comment": comment
        }
        command_queue.append(cmd)
        print(f"🔄 Reversal: Queued CLOSE for Ticket {pos.get('ticket')} ({pos.get('type')})")

    async def execute_auto_trade(self, signal_data, rule):
        symbol = signal_data.get("symbol")
        raw_action = signal_data.get("action", "")
        price = signal_data.get("price")
        sl = signal_data.get("sl")
        tp = signal_data.get("tp")
        
        # Kelly Criterion Position Sizing
        if rule.get("use_kelly_sizing"):
            confidence = signal_data.get("confidence")
            volume = await self.calculate_kelly_lot_size(symbol, rule, confidence)
        else:
            volume = float(rule.get("fixed_lot_size", 0.01))
        
        if "BUY" in raw_action.upper():
            action = "BUY"
        elif "SELL" in raw_action.upper():
            action = "SELL"
        else:
            action = raw_action 
        
        command = {
            "type": "TRADE",
            "action": action,
            "symbol": symbol,
            "volume": volume,
            "sl": sl,
            "tp": tp,
            "ticket": 0, 
            "price": price,
            "comment": "Auto-Exec"
        }
        
        command_queue.append(command)
        print(f"🚀 Auto-Execution: {action} (from {raw_action}) {symbol} {volume} Lots (Rule: {rule.get('id')})")
        return command

automation_manager = AutomationManager()

# --- Signal Watcher ---
async def watch_signal_directory():
    """
    Periodically scan the signal directory for new JSON files from MT5.
    """
    print(f"Starting signal watcher on {SIGNAL_DIR}...")
    if not os.path.exists(SIGNAL_DIR):
        try:
            os.makedirs(SIGNAL_DIR, exist_ok=True)
            print(f"Created signal directory: {SIGNAL_DIR}")
        except Exception as e:
            print(f"Failed to create signal directory: {e}")
            return

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase = None
    
    if supabase_url and supabase_key:
        try:
            supabase = create_client(supabase_url, supabase_key)
        except Exception as e:
            print(f"Failed to init Supabase client in watcher: {e}")

    while True:
        try:
            # Sync automation rules
            await automation_manager.sync_rules()

            # Scan for .json files
            files = glob.glob(os.path.join(SIGNAL_DIR, "*.json"))
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                    
                    print(f"🔔 New Signal Received: {content}")
                    
                    # Check Automation
                    is_auto, rule_or_msg = await automation_manager.evaluate_signal(content)
                    auto_status = "skipped"
                    
                    if is_auto:
                        try:
                            await automation_manager.execute_auto_trade(content, rule_or_msg)
                            auto_status = "executed"
                        except Exception as exec_error:
                            print(f"❌ Auto-Execution Failed: {exec_error}")
                            auto_status = f"failed: {str(exec_error)}"
                    else:
                        print(f"Automation Skipped: {rule_or_msg}")

                    # Save extended features to training_signals table if available
                    if supabase and "ema_short" in content:
                        try:
                            ts_val = content.get("timestamp", int(time.time()))
                            dt_obj = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                            
                            # Generate signal_id if not present
                            signal_id = content.get("signal_id")
                            if not signal_id:
                                signal_id = f"{content.get('symbol')}_{ts_val}"

                            training_data = {
                                "signal_id": signal_id,
                                "symbol": content.get("symbol"),
                                "action": content.get("action"),
                                "timestamp": dt_obj.isoformat(),
                                "signal_price": content.get("price"),
                                "sl": content.get("sl"),
                                "tp": content.get("tp"),
                                "ema_short": content.get("ema_short"),
                                "ema_long": content.get("ema_long"),
                                "atr": content.get("atr"),
                                "adx": content.get("adx"),
                                "center": content.get("center"),
                                "distance_ok": bool(content.get("distance_ok")),
                                "slope_ok": bool(content.get("slope_ok")),
                                "trend_filter_ok": bool(content.get("trend_filter_ok")),
                                "htf_trend_ok": bool(content.get("htf_trend_ok")),
                                "volatility_ok": bool(content.get("volatility_ok")),
                                "chop_ok": bool(content.get("chop_ok")),
                                "spread_ok": bool(content.get("spread_ok")),
                                "bars_since_last": content.get("bars_since_last"),
                                "trend_direction": content.get("trend_direction"),
                                "ema_cross_event": content.get("ema_cross_event"),
                                "ema_spread": content.get("ema_spread"),
                                "atr_percent": content.get("atr_percent"),
                                "reclaim_state": content.get("reclaim_state"),
                                "is_reclaim_signal": bool(content.get("is_reclaim_signal")),
                                "price_vs_center": content.get("price_vs_center"),
                                "cloud_width": content.get("cloud_width"),
                                "executed": is_auto,
                                "execution_time": datetime.now(timezone.utc).isoformat() if is_auto else None
                            }
                            supabase.table("training_signals").upsert(training_data, on_conflict="signal_id").execute()
                            print("💾 Extended training features saved to DB")
                        except Exception as e:
                            print(f"⚠️ Failed to save training features: {e}")

                    # Insert into Supabase (Legacy signals table)
                    if supabase:
                        try:
                            raw_action = content.get("action", "").upper()
                            db_action = "BUY" if "BUY" in raw_action else "SELL"
                            original_comment = content.get("comment", "")
                            final_comment = f"{original_comment} | Original: {raw_action} | Auto: {auto_status}"
                            
                            signal_data = {
                                "symbol": content.get("symbol"),
                                "action": db_action, 
                                "price": content.get("price"),
                                "sl": content.get("sl"),
                                "tp": content.get("tp"),
                                "status": "new",
                                "source": "mt5_indicator",
                                "raw_data": content,
                                "comment": final_comment 
                            }
                            supabase.table("signals").insert(signal_data).execute()
                            print("✅ Signal saved to DB")
                        except Exception as db_error:
                            print(f"❌ Failed to save signal to DB: {db_error}")
                    
                    os.remove(file_path)
                    
                except Exception as e:
                    print(f"Error processing signal file {file_path}: {e}")
                    try:
                        os.rename(file_path, file_path + ".err")
                    except:
                        pass
                        
        except Exception as e:
            print(f"Watcher error: {e}")
            
        await asyncio.sleep(0.5) # Check every 500ms

@app.post("/signals/new")
async def new_signal(signal: SignalPayload):
    """
    Inject a new signal via HTTP (e.g., from external scripts or manual testing).
    Triggers the same AI evaluation logic as MT5 file signals.
    """
    content = signal.dict()
    print(f"🔔 New HTTP Signal Received: {content}")
    
    # Check Automation
    is_auto, rule_or_msg = await automation_manager.evaluate_signal(content)
    auto_status = "skipped"
    
    if is_auto:
        try:
            await automation_manager.execute_auto_trade(content, rule_or_msg)
            auto_status = "executed"
        except Exception as exec_error:
            print(f"❌ Auto-Execution Failed: {exec_error}")
            auto_status = f"failed: {str(exec_error)}"
    else:
        print(f"Automation Skipped: {rule_or_msg}")

    # Insert into Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    print(f"DEBUG: Supabase Config - URL: {supabase_url}, Key: {supabase_key[:10]}... (Length: {len(supabase_key) if supabase_key else 0})")

    if supabase_url and supabase_key:
        try:
            supabase = create_client(supabase_url, supabase_key)
            raw_action = content.get("action", "").upper()
            db_action = "BUY" if "BUY" in raw_action else "SELL"
            original_comment = content.get("comment", "")
            final_comment = f"{original_comment} | Original: {raw_action} | Auto: {auto_status}"
            
            signal_data = {
                "symbol": content.get("symbol"),
                "action": db_action, 
                "price": content.get("price"),
                "sl": content.get("sl"),
                "tp": content.get("tp"),
                "status": "new",
                "source": content.get("source", "http_api"),
                "raw_data": content,
                "comment": final_comment 
            }
            print(f"DEBUG: Attempting to insert signal data: {signal_data}")
            res = supabase.table("signals").insert(signal_data).execute()
            print(f"✅ HTTP Signal saved to DB. Response: {res}")
        except Exception as db_error:
            import traceback
            print(f"❌ Failed to save HTTP signal to DB: {db_error}")
            traceback.print_exc()
            return {"status": "error", "message": str(db_error), "detail": traceback.format_exc()}
    else:
        print("❌ Supabase credentials missing in environment variables!")
        return {"status": "error", "message": "Supabase credentials missing"}
            
    return {"status": "processed", "auto_execution": auto_status, "reason": str(rule_or_msg) if not is_auto else "ok"}

@app.post("/inference")
async def inference_endpoint(payload: InferencePayload):
    """
    Endpoint for 1-5m AI Inference (Push from EA)
    """
    print(f"🧠 Inference Request: {payload.symbol} {payload.timeframe} (Candles: {len(payload.candles)})")
    
    # 1. Convert to gRPC Request
    req_id = str(uuid.uuid4())
    
    grpc_candles = []
    for c in payload.candles:
        grpc_candles.append(alphaos_pb2.Candle(
            time=c.t, open=c.o, high=c.h, low=c.l, close=c.c, volume=float(c.v),
            # Map new fields
            volume_real=c.rv,
            tick_count=c.tc,
            aggressor_buy=c.ab,
            aggressor_sell=c.as_
        ))
        
    grpc_bids = [alphaos_pb2.DOMLevel(price=l.p, volume=float(l.v), volume_real=float(l.v)) for l in payload.dom.bids]
    grpc_asks = [alphaos_pb2.DOMLevel(price=l.p, volume=float(l.v), volume_real=float(l.v)) for l in payload.dom.asks]
    
    # Calc quick filters & DOM for TechContext so AI 侧不再出现 Filters=0
    last_candle = payload.candles[-1]
    close_px = last_candle.c
    high_px = last_candle.h
    low_px = last_candle.l
    spread_val = max(payload.ask - payload.bid, 0.0)
    spread_pct = spread_val / close_px if close_px else 0.0
    range_pct = (high_px - low_px) / close_px if close_px else 0.0

    distance_ok = True  # 默认放行，避免全零；细化可接入EA信号
    trend_filter_ok = True
    volatility_ok = range_pct < 0.02   # 当根波动 <2% 视为可交易
    spread_ok = spread_pct < 0.0005    # 点差 <0.05%

    tech_context = alphaos_pb2.TechnicalContext(
        dom_imbalance = 0.0,
        distance_ok = distance_ok,
        trend_filter_ok = trend_filter_ok,
        volatility_ok = volatility_ok,
        spread_ok = spread_ok,
        bars_since_last = 0,
    )
    
    total_bid = sum([b.volume for b in grpc_bids])
    total_ask = sum([a.volume for a in grpc_asks])
    if total_bid + total_ask > 0:
        tech_context.dom_imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        tech_context.best_bid_vol = grpc_bids[0].volume if grpc_bids else 0.0
        tech_context.best_ask_vol = grpc_asks[0].volume if grpc_asks else 0.0

    print(f"🧮 Filters Preview -> Dist:{distance_ok} Trend:{trend_filter_ok} Vol:{volatility_ok} Spr:{spread_ok} | spread_pct={spread_pct:.6f} range_pct={range_pct:.6f}")

    signal_req = alphaos_pb2.SignalRequest(
        request_id=req_id,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        market_context=grpc_candles,
        dom_bids=grpc_bids,
        dom_asks=grpc_asks,
        signal_source="mt5_inference_push",
        action="SCAN",
        suggested_entry=payload.bid if payload.action == "SELL" else payload.ask,
        suggested_sl=0.0,
        suggested_tp=0.0,
        technical_context=tech_context
    )
    
    # 2. Send to AI Engine
    try:
        # Timeout 2s
        ai_response = await grpc_service.send_signal_and_wait(signal_req)
        
        if ai_response:
            print(f"🤖 AI Decision: {ai_response.should_execute} (Act: {ai_response.action}, Conf: {ai_response.confidence:.2f})")
            
            # === STORE SIGNAL TO SUPABASE ===
            if automation_manager.supabase:
                try:
                    # Use timestamp from last candle
                    ts_val = int(last_candle.t)
                    dt_obj = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                    
                    # Generate deterministic signal_id matching AI Engine's logic
                    signal_id = f"{payload.symbol}_{ts_val}"
                    
                    training_signal_data = {
                        "signal_id": signal_id,
                        "symbol": payload.symbol,
                        "action": ai_response.action, # AI's decided action (BUY/SELL/WAIT)
                        "timestamp": dt_obj.isoformat(),
                        "signal_price": last_candle.c,
                        "sl": ai_response.adjusted_sl,
                        "tp": ai_response.adjusted_tp,
                        "executed": ai_response.should_execute,
                        # Store basic context
                        "spread_ok": spread_ok,
                        "volatility_ok": volatility_ok
                    }
                    
                    # Upsert (AI Engine might update it later with features)
                    automation_manager.supabase.table("training_signals").upsert(training_signal_data, on_conflict="signal_id").execute()
                    
                    # --- ALSO SYNC TO 'signals' TABLE FOR FRONTEND VISIBILITY ---
                    # Format comment to match frontend regex: /AI:\s*(\d+\.?\d*)/
                    # e.g. "AI: 0.85 | Status: Executed"
                    status_desc = "Executed" if ai_response.should_execute else f"Skipped ({ai_response.reason})"
                    frontend_comment = f"AI: {ai_response.confidence:.2f} | Status: {status_desc}"
                    
                    frontend_signal_data = {
                        "symbol": payload.symbol,
                        "action": ai_response.action,
                        "price": last_candle.c,
                        "sl": ai_response.adjusted_sl,
                        "tp": ai_response.adjusted_tp,
                        "status": "processed" if ai_response.should_execute else "skipped",
                        "source": "ai_inference",
                        "comment": frontend_comment
                    }
                    automation_manager.supabase.table("signals").insert(frontend_signal_data).execute()
                    # ------------------------------------------------------------

                except Exception as e:
                    print(f"⚠️ Failed to store inference signal: {e}")
            # ================================

            if ai_response.should_execute:
                # 3. Check Automation Rules
                # Pass action to allow reversal logic
                is_auto, rule = await automation_manager.evaluate_signal({
                    "symbol": payload.symbol, 
                    "action": ai_response.action,
                    "confidence": ai_response.confidence
                })
                
                if is_auto and isinstance(rule, dict): # rule might be bool/dict/msg
                    # 4. Queue Execution
                    # Check for Kelly Criterion
                    if rule.get("use_kelly_sizing"):
                        volume = await automation_manager.calculate_kelly_lot_size(payload.symbol, rule, ai_response.confidence)
                    else:
                        volume = float(rule.get("fixed_lot_size", 0.01))
                    
                    command = {
                        "type": "TRADE",
                        "action": ai_response.action,
                        "symbol": payload.symbol,
                        "volume": volume,
                        "sl": ai_response.adjusted_sl,
                        "tp": ai_response.adjusted_tp,
                        "price": 0.0, # Market
                        "comment": f"AI-{ai_response.confidence:.2f}"
                    }
                    command_queue.append(command)
                    return {"status": "executed", "action": ai_response.action}
                else:
                    return {"status": "ignored", "reason": "Automation disabled"}
            
            return {"status": "wait", "reason": ai_response.reason}
            
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "timeout"}

@app.on_event("startup")
async def startup_event():
    # Start watcher and gRPC server in background
    asyncio.create_task(watch_signal_directory())
    asyncio.create_task(start_grpc_server(grpc_service))

@app.post("/trade/execute")
async def execute_trade(trade: TradeRequest):
    command = {
        "type": trade.type if trade.type else "TRADE",
        "action": trade.action,
        "symbol": trade.symbol,
        "volume": trade.volume,
        "sl": trade.sl,
        "tp": trade.tp,
        "ticket": trade.ticket, 
        "price": trade.price
    }
    command_queue.append(command)
    return {"status": "queued", "queue_length": len(command_queue)}

@app.post("/trade/report")
async def report_trade(report: TradeReport):
    global last_trade
    last_trade = report.dict()
    print(f"Trade Reported: {last_trade}")
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            db_side = report.type.lower()
            
            if report.entry == "IN":
                # 1. Sync to 'trades' table
                existing = supabase.table("trades").select("id").eq("external_order_id", str(report.position_id)).execute()
                if existing.data:
                    print(f"⚠️  Position {report.position_id} already exists")
                else:
                    data = {
                        "symbol": report.symbol,
                        "side": db_side,
                        "quantity": report.volume,
                        "entry_price": report.price,
                        "status": "open",
                        "notes": f"MT5 Deal: {report.ticket} | Position ID: {report.position_id}",
                        "external_order_id": str(report.position_id),
                        "commission": report.commission,
                        "swap": report.swap,
                        "mae": report.mae,
                        "mfe": report.mfe
                    }
                    supabase.table("trades").insert(data).execute()
                    print(f"✅ Trade OPEN synced (Position ID: {report.position_id})")
                
                # 2. Link execution details to 'training_signals'
                # Try to find a matching recent signal
                try:
                    # Matching logic: Same Symbol, executed=True, action contains BUY/SELL
                    action_dir = "BUY" if report.type in [0, "DEAL_TYPE_BUY"] else "SELL"
                    
                    # Look for the latest unlinked signal for this symbol
                    # Use ilike for partial match (e.g., "RECLAIM_SELL" contains "SELL")
                    recent_signals = supabase.table("training_signals").select("signal_id, action") \
                        .eq("symbol", report.symbol) \
                        .ilike("action", f"%{action_dir}%") \
                        .eq("executed", True) \
                        .is_("position_id", "null") \
                        .order("timestamp", desc=True) \
                        .limit(1) \
                        .execute()
                    
                    print(f"🔍 Looking for signal to link: Symbol={report.symbol} Action contains '{action_dir}'")
                        
                    if recent_signals.data:
                        sig_id = recent_signals.data[0]['signal_id']
                        
                        update_data = {
                            "order_id": str(report.ticket),
                            "position_id": str(report.position_id),
                            "execution_price": report.price,
                            # execution_spread is tricky without tick data at moment of execution
                            "broker_time": datetime.fromtimestamp(int(report.time), tz=timezone.utc).isoformat() 
                        }
                        supabase.table("training_signals").update(update_data).eq("signal_id", sig_id).execute()
                        print(f"🔗 Linked execution Position={report.position_id} to signal {sig_id}")
                    else:
                        print(f"⚠️ No unlinked signal found for {report.symbol} {action_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to link execution to training signal: {e}")

            elif report.entry == "OUT":
                # 1. Update 'trades' table
                target_trade = None
                response = supabase.table("trades").select("*").eq("external_order_id", str(report.position_id)).eq("status", "open").execute()
                if response.data:
                    target_trade = response.data[0]
                else:
                    closed_check = supabase.table("trades").select("id").eq("external_order_id", str(report.position_id)).eq("status", "closed").execute()
                    if not closed_check.data:
                         response = supabase.table("trades").select("*").eq("symbol", report.symbol).eq("status", "open").execute()
                         for t in response.data:
                             if str(report.position_id) in t.get('notes', ''):
                                 target_trade = t
                                 break

                if target_trade:
                    pnl_net = report.profit + report.commission + report.swap
                    update_data = {
                        "status": "closed",
                        "exit_price": report.price,
                        "pnl_net": pnl_net,
                        "pnl_gross": report.profit,
                        "commission": (target_trade.get('commission', 0) or 0) + report.commission,
                        "swap": (target_trade.get('swap', 0) or 0) + report.swap,
                        "notes": target_trade['notes'] + f" | Closed: Deal {report.ticket}, PnL: {report.profit:.2f}",
                        "mae": report.mae if report.mae != 0 else target_trade.get('mae', 0),
                        "mfe": report.mfe if report.mfe != 0 else target_trade.get('mfe', 0)
                    }
                    supabase.table("trades").update(update_data).eq("id", target_trade['id']).execute()
                    print(f"✅ Trade CLOSED synced (ID: {target_trade['id']})")
                
                # 2. Update 'training_signals' with outcome
                try:
                    # Find signal by position_id
                    print(f"🔍 Looking for training signal with position_id={report.position_id}")
                    training_sig = supabase.table("training_signals").select("signal_id").eq("position_id", str(report.position_id)).execute()
                    
                    # Fallback: If position_id match fails, try to find by symbol and order_id
                    if not training_sig.data:
                        print(f"⚠️ No match by position_id. Trying order_id={report.ticket}")
                        training_sig = supabase.table("training_signals").select("signal_id").eq("order_id", str(report.ticket)).execute()
                    
                    if training_sig.data:
                        sig_id = training_sig.data[0]['signal_id']
                        
                        outcome_data = {
                            "exit_price": report.price,
                            "exit_time": datetime.fromtimestamp(int(report.time), tz=timezone.utc).isoformat() if report.time else None,
                            "result_profit": report.profit,
                            "result_mae": report.mae,
                            "result_mfe": report.mfe,
                            "result_win": report.profit > 0,
                            "exit_reason": report.comment if hasattr(report, 'comment') else "closed"
                        }
                        supabase.table("training_signals").update(outcome_data).eq("signal_id", sig_id).execute()
                        print(f"🎯 Training signal {sig_id} outcome updated (PnL: {report.profit})")
                    else:
                        print(f"❌ No training signal found for position {report.position_id} or order {report.ticket}")
                        
                        # Write outcome file for Auto-Learner to pick up
                        if os.path.exists(SIGNAL_DIR):
                            try:
                                outcome_file_data = {
                                     "signal_id": sig_id,
                                     "symbol": report.symbol,
                                     "exit_price": report.price,
                                     "exit_time": int(time.time()),
                                     "profit": report.profit,
                                     "mae": report.mae,
                                     "mfe": report.mfe
                                }
                                fname = f"outcomes_{sig_id}_{int(time.time())}.json"
                                with open(os.path.join(SIGNAL_DIR, fname), 'w') as f:
                                     json.dump(outcome_file_data, f)
                                print(f"📝 Wrote outcome file for Auto-Learner: {fname}")
                            except Exception as e:
                                print(f"⚠️ Failed to write outcome file: {e}")

                except Exception as e:
                    print(f"⚠️ Failed to update training signal outcome: {e}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to sync to Supabase: {e}")
            return {"status": "error", "message": str(e), "detail": traceback.format_exc()}
    
    return {"status": "received"}

@app.get("/commands/pop")
async def pop_command():
    if command_queue:
        return command_queue.pop(0)
    return None

@app.post("/status/update")
async def update_status(status: StatusUpdate):
    global last_status
    
    # Merge logic: only update fields that are not None
    new_data = status.dict(exclude_unset=True, exclude_none=True)
    last_status.update(new_data)
    
    if status.symbol:
        active_symbols[status.symbol] = {
            "bid": status.bid,
            "ask": status.ask,
            "last_seen": time.time()
        }
    current_time = time.time()
    to_remove = [s for s, data in active_symbols.items() if current_time - data["last_seen"] > 10]
    for s in to_remove:
        del active_symbols[s]
    return {"status": "received"}

@app.get("/status")
async def get_status():
    return {
        "bridge_status": "connected",
        "last_mt5_update": last_status,
        "active_symbols": list(active_symbols.keys()),
        "symbol_prices": active_symbols,
        "pending_commands": len(command_queue),
        "last_trade": last_trade
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/history")
async def get_history(symbol: str, timeframe: str, count: int = 1000, from_ts: int = 0, to_ts: int = 0):
    request_id = str(uuid.uuid4())
    command = {
        "type": "GET_HISTORY",
        "request_id": request_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "count": count,
        "from": from_ts,
        "to": to_ts
    }
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    history_requests[request_id] = future
    command_queue.append(command)
    try:
        data = await asyncio.wait_for(future, timeout=60.0)
        return data
    except asyncio.TimeoutError:
        if request_id in history_requests:
            del history_requests[request_id]
        raise HTTPException(status_code=504, detail="Timeout waiting for MT5 response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/history")
async def receive_history(history: HistoryData):
    req_id = history.request_id
    if req_id in history_requests:
        future = history_requests[req_id]
        if not future.done():
            future.set_result(history.dict())
        del history_requests[req_id]
        return {"status": "accepted"}
    return {"status": "ignored", "reason": "request_id not found or expired"}

@app.post("/data/dom")
async def receive_dom(dom: DOMData):
    req_id = dom.request_id
    if req_id in dom_requests:
        future = dom_requests[req_id]
        if not future.done():
            future.set_result(dom.dict())
        del dom_requests[req_id]
        return {"status": "accepted"}
    return {"status": "ignored", "reason": "request_id not found or expired"}
