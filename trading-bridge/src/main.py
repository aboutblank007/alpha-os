from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import time
import os
from supabase import create_client, Client

app = FastAPI(title="MT5 Trading Bridge (HTTP)")

# 简单的内存队列和存储
command_queue = []
last_status = {}
active_symbols = {} # { "EURUSD": { "bid": 1.1, "ask": 1.2, "last_seen": 1234567890 } }
last_trade = None  # Stores the most recent trade execution

# 历史数据请求存储 { request_id: asyncio.Future }
history_requests: Dict[str, asyncio.Future] = {}

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

# ...

@app.post("/trade/execute")
async def execute_trade(trade: TradeRequest):
    command = {
        "type": trade.type if trade.type else "TRADE",
        "action": trade.action,
        "symbol": trade.symbol,
        "volume": trade.volume,
        "sl": trade.sl,
        "tp": trade.tp,
        "ticket": trade.ticket, # Pass as number, EA will parse it
        "price": trade.price
    }
    command_queue.append(command)
    return {"status": "queued", "queue_length": len(command_queue)}

@app.post("/trade/report")
async def report_trade(report: TradeReport):
    global last_trade
    last_trade = report.dict()
    print(f"Trade Reported: {last_trade}")
    
    # Auto-Journaling to Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # Map MT5 data to Supabase schema
            db_side = report.type.lower() # "BUY" -> "buy"
            
            if report.entry == "IN":
                # Check if this position is already recorded (防重复插入)
                existing = supabase.table("trades").select("id").eq("external_order_id", str(report.position_id)).execute()
                
                if existing.data:
                    print(f"⚠️  Position {report.position_id} already exists in DB, skipping duplicate insert")
                else:
                    # Insert new trade
            data = {
                "symbol": report.symbol,
                "side": db_side,
                "quantity": report.volume,
                "entry_price": report.price,
                "status": "open",
                        "notes": f"MT5 Deal: {report.ticket} | Position ID: {report.position_id}",
                        "external_order_id": str(report.position_id),  # Store position_id here for easy matching
                "pnl_net": 0,
                "pnl_gross": 0,
                        "commission": report.commission,
                        "swap": report.swap,
                        "mae": report.mae,
                        "mfe": report.mfe
                    }
                    supabase.table("trades").insert(data).execute()
                    print(f"✅ Trade OPEN synced to Supabase (Position ID: {report.position_id}, Deal: {report.ticket})")
                
            elif report.entry == "OUT":
                # Update existing trade (Close)
                # Strategy 1: Match by external_order_id (Position ID)
                target_trade = None
                
                # Try to find by external_order_id first (must be "open" status)
                response = supabase.table("trades").select("*").eq("external_order_id", str(report.position_id)).eq("status", "open").execute()
                if response.data:
                    target_trade = response.data[0]
                    print(f"Found trade by position_id: {report.position_id}")
                else:
                    # Check if this position was already closed (防止重复关闭)
                    closed_check = supabase.table("trades").select("id, status").eq("external_order_id", str(report.position_id)).eq("status", "closed").execute()
                    if closed_check.data:
                        print(f"⚠️  Position {report.position_id} already closed (Deal {report.ticket}), skipping duplicate close update")
                        # Don't fall back to FIFO matching - just skip
                        target_trade = None
                    else:
                        # Strategy 2: Match by symbol + status=open + notes containing position_id
                        response = supabase.table("trades").select("*").eq("symbol", report.symbol).eq("status", "open").execute()
                        open_trades = response.data
                        
                        for t in open_trades:
                            if str(report.position_id) in t.get('notes', ''):
                                target_trade = t
                                print(f"Found trade by notes search: {report.position_id}")
                                break
                        
                        # Strategy 3: FIFO matching - DISABLED to prevent accidental closes
                        # if not target_trade and open_trades:
                        #     target_trade = open_trades[0] 
                        #     print(f"Warning: Using FIFO matching for symbol {report.symbol}")

                if target_trade:
                    update_data = {
                        "status": "closed",
                        "exit_price": report.price,
                        "pnl_net": report.profit + report.commission + report.swap, # Total Net PnL
                        "pnl_gross": report.profit,
                        "commission": (target_trade.get('commission', 0) or 0) + report.commission,
                        "swap": (target_trade.get('swap', 0) or 0) + report.swap,
                        "notes": target_trade['notes'] + f" | Closed: Deal {report.ticket}, PnL: {report.profit:.2f}",
                        "mae": report.mae if report.mae != 0 else target_trade.get('mae', 0),
                        "mfe": report.mfe if report.mfe != 0 else target_trade.get('mfe', 0)
                    }
                    
                    supabase.table("trades").update(update_data).eq("id", target_trade['id']).execute()
                    print(f"✅ Trade CLOSED synced to Supabase (ID: {target_trade['id']}, PnL: {report.profit:.2f})")
                elif not closed_check.data if 'closed_check' in locals() else True:
                    # Only log error if position wasn't already closed
                    print(f"❌ Error: Could not find open trade for closing deal {report.ticket} (Position ID: {report.position_id})")

        except Exception as e:
            print(f"Failed to sync to Supabase: {e}")
    
    return {"status": "received"}

@app.get("/commands/pop")
async def pop_command():
    if command_queue:
        return command_queue.pop(0)
    return None

@app.post("/status/update")
async def update_status(status: StatusUpdate):
    global last_status
    
    # Debug: Print positions count if any
    if status.positions:
        print(f"Received {len(status.positions)} positions from EA")
        # Sample log to verify data structure
        if len(status.positions) > 0:
            print(f"First position sample: {status.positions[0]}")
    else:
        print("Received status update with NO positions")
    
    last_status = status.dict()
    
    # Update active symbols list
    if status.symbol:
        active_symbols[status.symbol] = {
            "bid": status.bid,
            "ask": status.ask,
            "last_seen": time.time()
        }
    
    # Prune old symbols (older than 10s)
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
        "last_trade": last_trade  # Expose last trade to frontend
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- History Data Endpoints ---

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
    
    # Create a future to wait for the response
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    history_requests[request_id] = future
    
    # Queue the command for EA
    command_queue.append(command)
    
    try:
        # Wait for data with timeout (e.g., 30 seconds)
        data = await asyncio.wait_for(future, timeout=30.0)
        return data
    except asyncio.TimeoutError:
        # Clean up if timed out
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
