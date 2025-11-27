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
from ml_engine import feature_engineer, predictor
import pandas as pd

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
    account: dict  # { balance, equity, margin, free_margin }
    positions: List[dict] # [{ ticket, symbol, type, volume, pnl, sl, tp ... }]
    # Legacy support
    symbol: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

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

class SignalPayload(BaseModel):
    symbol: str
    action: str # "BUY" or "SELL"
    price: float
    sl: Optional[float] = 0.0
    tp: Optional[float] = 0.0
    source: Optional[str] = "http_api"
    comment: Optional[str] = ""

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
            print(f"DEBUG: Synced {len(self.rules)} automation rules from Supabase")
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

    async def evaluate_signal(self, signal_data):
        symbol = signal_data.get("symbol")
        
        # 1. Check for specific symbol rule first, then GLOBAL
        rule = self.rules.get(symbol) or self.rules.get("GLOBAL")
        
        if not rule:
            print(f"DEBUG: No rule found for {symbol}. Rules loaded: {len(self.rules)}")
            return False, "No matching automation rule"
            
        if not rule.get("is_enabled"):
            print(f"DEBUG: Rule found for {symbol} but disabled.")
            return False, "Automation disabled for this rule"

        # 2. Spread Check (Best Effort)
        max_spread = rule.get("max_spread_points", 50)
        market_data = active_symbols.get(symbol)
        
        if market_data and max_spread > 0:
            bid = market_data.get("bid")
            ask = market_data.get("ask")
            if bid and ask:
                spread = ask - bid
                # Basic estimation
                pass
        
        # 3. AI Mode Handling
        ai_mode = rule.get("ai_mode", "indicator_ai")
        
        if ai_mode == 'legacy':
             print(f"DEBUG: Legacy mode - Auto-Execution Approved for {symbol}. Rule ID: {rule.get('id')}")
             return True, rule

        # Data Collection & Remote Inference
        print(f"DEBUG: Fetching context data for {symbol} (Mode: {ai_mode})...")
        
        history_payload = await self.get_history_data(symbol, "PERIOD_H1", 100)
        
        if history_payload and "data" in history_payload:
            candles = history_payload["data"]
            
            # Construct gRPC Request
            req_id = str(uuid.uuid4())
            grpc_candles = []
            for c in candles:
                # Map candle data to proto message
                grpc_candles.append(alphaos_pb2.Candle(
                    time=int(c.get('time', 0)),
                    open=float(c.get('open', 0)),
                    high=float(c.get('high', 0)),
                    low=float(c.get('low', 0)),
                    close=float(c.get('close', 0)),
                    volume=float(c.get('tick_volume', 0))
                ))
            
            signal_req = alphaos_pb2.SignalRequest(
                request_id=req_id,
                symbol=symbol,
                timeframe="H1",
                market_context=grpc_candles,
                signal_source="mt5_bridge",
                action=signal_data.get("action", ""),
                suggested_entry=float(signal_data.get("price", 0)),
                suggested_sl=float(signal_data.get("sl", 0)),
                suggested_tp=float(signal_data.get("tp", 0))
            )
            
            # Send to AI Engine
            print(f"📡 Sending signal {req_id} to Local AI Engine...")
            ai_response = await grpc_service.send_signal_and_wait(signal_req)
            
            if ai_response:
                print(f"🤖 AI Response received: {ai_response.should_execute} (Conf: {ai_response.confidence:.2f})")
                
                # Save to DB (Snapshot)
                if self.supabase:
                    try:
                        training_record = {
                            "symbol": symbol,
                            "features": {"raw_candles_count": len(candles)}, # Simplified for now
                            "market_context": candles[-5:],
                            "model_version": "remote_v1",
                            "notes": f"Remote AI Decision: {ai_response.should_execute} | Conf: {ai_response.confidence:.2f}"
                        }
                        self.supabase.table("training_datasets").insert(training_record).execute()
                    except Exception as e:
                         print(f"❌ Failed to save training data: {e}")
                
                if ai_response.should_execute:
                    # Override parameters if AI suggested adjustments
                    # For now, just return True
                    return True, rule
                else:
                    return False, f"AI rejected signal: {ai_response.reason}"
            else:
                print("⚠️ No response from AI Engine (Timeout/Offline).")
                # Fallback policy: If AI is required but offline, skip trade?
                # Or fall back to legacy?
                # For safety, let's skip if mode is pure_ai, but maybe allow if indicator_ai?
                # Let's be safe and skip.
                return False, "AI Engine Offline/Timeout"

        else:
            print("⚠️ Could not fetch history, skipping AI check.")
            return False, "AI check failed (no data)"

        return True, rule

    def execute_auto_trade(self, signal_data, rule):
        symbol = signal_data.get("symbol")
        raw_action = signal_data.get("action", "")
        price = signal_data.get("price")
        sl = signal_data.get("sl")
        tp = signal_data.get("tp")
        
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
                            automation_manager.execute_auto_trade(content, rule_or_msg)
                            auto_status = "executed"
                        except Exception as exec_error:
                            print(f"❌ Auto-Execution Failed: {exec_error}")
                            auto_status = f"failed: {str(exec_error)}"
                    else:
                        print(f"Automation Skipped: {rule_or_msg}")

                    # Insert into Supabase
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
            automation_manager.execute_auto_trade(content, rule_or_msg)
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
                
            elif report.entry == "OUT":
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
                    update_data = {
                        "status": "closed",
                        "exit_price": report.price,
                        "pnl_net": report.profit + report.commission + report.swap,
                        "pnl_gross": report.profit,
                        "commission": (target_trade.get('commission', 0) or 0) + report.commission,
                        "swap": (target_trade.get('swap', 0) or 0) + report.swap,
                        "notes": target_trade['notes'] + f" | Closed: Deal {report.ticket}, PnL: {report.profit:.2f}",
                        "mae": report.mae if report.mae != 0 else target_trade.get('mae', 0),
                        "mfe": report.mfe if report.mfe != 0 else target_trade.get('mfe', 0)
                    }
                    supabase.table("trades").update(update_data).eq("id", target_trade['id']).execute()
                    print(f"✅ Trade CLOSED synced (ID: {target_trade['id']})")

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
    last_status = status.dict()
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
        data = await asyncio.wait_for(future, timeout=30.0)
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
