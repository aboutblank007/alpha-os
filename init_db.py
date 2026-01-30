
import asyncio
from alphaos.core.config import AlphaOSConfig
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def init_db():
    config = AlphaOSConfig()
    db_url = config.database.connection_string
    
    print(f"Connecting to {db_url}...")
    engine = create_async_engine(db_url)
    
    # Transaction 1: Create Table
    async with engine.begin() as conn:
        print("Creating runtime_state table...")
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS runtime_state (
                timestamp FLOAT PRIMARY KEY,
                symbol VARCHAR(20),
                warmup_progress FLOAT,
                ticks_total INTEGER,
                open_positions INTEGER,
                guardian_halt BOOLEAN,
                market_phase VARCHAR(50),
                temperature FLOAT,
                entropy FLOAT,
                exit_v21_enabled BOOLEAN,
                db_snapshot_count INTEGER,
                extra_data JSONB
            );
        """))
        print("Table created successfully.")

    # Transaction 2: TimescaleDB (Optional)
    # We do this in a separate connection/transaction so failure doesn't rollback the table creation
    try:
        async with engine.begin() as conn:
             # Check if TimescaleDB extension exists
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            # Note: create_hypertable requires TIMESTAMP/INTEGER type for time column, FLOAT might fail
            # We skip this for now to ensure stability unless we change schema
            # await conn.execute(text("SELECT create_hypertable('runtime_state', 'timestamp', if_not_exists => TRUE);"))
            # print("Converted to Hypertable.")
            pass
    except Exception as e:
        print(f"TimescaleDB setup skipped: {e}")

    await engine.dispose()
    print("Database initialization complete.")

if __name__ == "__main__":
    asyncio.run(init_db())
