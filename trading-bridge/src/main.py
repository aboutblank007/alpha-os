from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="MT5 Trading Bridge (HTTP)")

# 简单的内存队列
command_queue = []
last_status = {}
last_trade = None  # Stores the most recent trade execution

class TradeRequest(BaseModel):
    action: str # BUY, SELL
    symbol: str
    volume: float = 0.01

class StatusUpdate(BaseModel):
    account: dict  # { balance, equity, margin, free_margin }
    positions: List[dict] # [{ ticket, symbol, type, volume, pnl, ... }]
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
    time: str  # ISO timestamp or similar

@app.post("/trade/execute")
async def execute_trade(trade: TradeRequest):
    command = {
        "type": "TRADE",
        "action": trade.action,
        "symbol": trade.symbol,
        "volume": trade.volume
    }
    command_queue.append(command)
    return {"status": "queued", "queue_length": len(command_queue)}

@app.post("/trade/report")
async def report_trade(report: TradeReport):
    global last_trade
    last_trade = report.dict()
    print(f"Trade Reported: {last_trade}")
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
    return {"status": "received"}

@app.get("/status")
async def get_status():
    return {
        "bridge_status": "connected",
        "last_mt5_update": last_status,
        "pending_commands": len(command_queue),
        "last_trade": last_trade  # Expose last trade to frontend
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
