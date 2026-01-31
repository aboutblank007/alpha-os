from __future__ import annotations
import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from alphaos.core.config import DatabaseConfig
from alphaos.monitoring.runtime_state import RuntimeSnapshot

logger = logging.getLogger(__name__)

class RuntimeStore:
    """
    Persists RuntimeSnapshot to TimescaleDB (SSOT).
    """
    def __init__(self, config: DatabaseConfig, symbol: str):
        self.config = config
        self.symbol = symbol
        self.engine = create_async_engine(config.connection_string, echo=False)
        self._snapshot_count = 0
        
    async def initialize(self):
        """Create table and hypertable if not exists."""
        async with self.engine.begin() as conn:
            # Create table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS runtime_state (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    warmup_progress DOUBLE PRECISION,
                    ticks_total BIGINT,
                    open_positions INTEGER,
                    guardian_halt BOOLEAN,
                    exit_v21_enabled BOOLEAN,
                    market_phase TEXT,
                    temperature DOUBLE PRECISION,
                    entropy DOUBLE PRECISION,
                    db_snapshot_count BIGINT
                );
            """))

            # Backward-compat migration: older schema may have `timestamp` column instead of `time`.
            try:
                cols = await conn.execute(
                    text(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = 'runtime_state'
                        """
                    )
                )
                col_map = {row[0]: row[1] for row in cols.fetchall()}
                if "time" not in col_map:
                    await conn.execute(text("ALTER TABLE runtime_state ADD COLUMN IF NOT EXISTS time TIMESTAMPTZ;"))
                    if "timestamp" in col_map:
                        dt = (col_map.get("timestamp") or "").lower()
                        if "timestamp" in dt:
                            await conn.execute(text("UPDATE runtime_state SET time = timestamp WHERE time IS NULL;"))
                        else:
                            await conn.execute(
                                text("UPDATE runtime_state SET time = to_timestamp(timestamp) WHERE time IS NULL;")
                            )
            except Exception as e:
                logger.debug(f"RuntimeStore schema inspection/migration skipped: {e}")
            
            # Convert to hypertable (ignore if already exists)
            try:
                await conn.execute(text("""
                    SELECT create_hypertable('runtime_state', 'time', if_not_exists => TRUE);
                """))
            except Exception as e:
                logger.debug(f"Hypertable creation info: {e}")
                
            logger.info("RuntimeStore initialized (TimescaleDB)")

    async def write_snapshot(self, snapshot: RuntimeSnapshot):
        """Write snapshot to DB."""
        self._snapshot_count += 1
        snapshot.db_snapshot_count = self._snapshot_count
        
        # Async write (fire and forget pattern optional, here we await for safety)
        try:
            async with self.engine.begin() as conn:
                params = {
                    "ts": snapshot.timestamp,
                    "symbol": snapshot.symbol,
                    "warmup": snapshot.warmup_progress,
                    "ticks": snapshot.ticks_total,
                    "pos": snapshot.open_positions,
                    "halt": snapshot.guardian_halt,
                    "exit_v21": snapshot.exit_v21_enabled,
                    "phase": snapshot.market_phase,
                    "temp": snapshot.temperature,
                    "entropy": snapshot.entropy,
                    "count": snapshot.db_snapshot_count,
                }

                # Preferred schema: `time` column (TIMESTAMPTZ)
                try:
                    await conn.execute(
                        text(
                            """
                            INSERT INTO runtime_state (
                                time, symbol, warmup_progress, ticks_total,
                                open_positions, guardian_halt, exit_v21_enabled,
                                market_phase, temperature, entropy, db_snapshot_count
                            ) VALUES (
                                to_timestamp(:ts), :symbol, :warmup, :ticks,
                                :pos, :halt, :exit_v21,
                                :phase, :temp, :entropy, :count
                            )
                            """
                        ),
                        params,
                    )
                except Exception:
                    # Backward compat: try `timestamp` column (float seconds) if `time` doesn't exist
                    await conn.execute(
                        text(
                            """
                            INSERT INTO runtime_state (
                                timestamp, symbol, warmup_progress, ticks_total,
                                open_positions, guardian_halt, exit_v21_enabled,
                                market_phase, temperature, entropy, db_snapshot_count
                            ) VALUES (
                                :ts, :symbol, :warmup, :ticks,
                                :pos, :halt, :exit_v21,
                                :phase, :temp, :entropy, :count
                            )
                            """
                        ),
                        params,
                    )
        except Exception as e:
            logger.error(f"Failed to write runtime snapshot: {e}")

    async def close(self):
        await self.engine.dispose()
