from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
import time

class RuntimeSnapshot(BaseModel):
    """
    AlphaOS v4 Runtime State Snapshot (SSOT for UI/Audit).
    Stored in TimescaleDB 'runtime_state' hypertable.
    """
    timestamp: float = Field(default_factory=time.time)
    symbol: str
    
    # Critical States
    warmup_progress: float = 0.0
    ticks_total: int = 0
    open_positions: int = 0
    guardian_halt: bool = False
    exit_v21_enabled: bool = True
    
    # Additional Context (Optional)
    market_phase: str = "UNKNOWN"
    temperature: float = 0.0
    entropy: float = 0.0
    db_snapshot_count: int = 0  # Sequence number
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1706540000.0,
                "symbol": "XAUUSD",
                "warmup_progress": 1.0,
                "ticks_total": 12345,
                "open_positions": 1,
                "guardian_halt": False,
                "exit_v21_enabled": True
            }
        }
