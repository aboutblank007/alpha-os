import requests
import pandas as pd
import time
import os
from src.features import FeatureEngineer
import logging

# Config
BRIDGE_URL = "http://49.235.153.73:8000"
SYMBOL = "BTCUSD"
TIMEFRAME = "M1"
TOTAL_CANDLES = 10000
BATCH_SIZE = 1000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCollector")

def fetch_history(symbol, timeframe, count, to_time=0):
    # Use the GET endpoint which triggers the Bridge->EA flow
    url = f"{BRIDGE_URL}/history"
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "count": count,
        "to_ts": to_time, # Note: Parameter name in main.py is to_ts
        "from_ts": 0
    }
    
    try:
        # Increase timeout because Bridge needs to wait for EA
        resp = requests.get(url, params=params, timeout=35) 
        if resp.status_code == 200:
            json_data = resp.json()
            # The GET /history endpoint returns the HistoryData object dict directly
            # structure: { request_id, symbol, timeframe, data: [...], count }
            return json_data.get("data", [])
        else:
            logger.error(f"Failed to fetch: {resp.status_code} - {resp.text}")
            return []
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []

def collect_data():
    logger.info(f"Starting collection for {SYMBOL} {TIMEFRAME}...")
    
    all_candles = []
    last_time = 0 # 0 means latest
    
    collected = 0
    while collected < TOTAL_CANDLES:
        logger.info(f"Fetching batch ending at {last_time if last_time > 0 else 'NOW'}...")
        batch = fetch_history(SYMBOL, TIMEFRAME, BATCH_SIZE, last_time)
        
        if not batch:
            logger.warning("No data received, stopping.")
            break
            
        # Filter duplicates/overlap if necessary, but Bridge usually handles 'to' correctly (exclusive or inclusive?)
        # MT5 CopyRates 'to' is inclusive usually.
        
        # Convert to DF for easy handling
        df_batch = pd.DataFrame(batch)
        
        # If we got the same last candle again, stop to avoid infinite loop
        if all_candles and df_batch.iloc[-1]['time'] == all_candles[-1]['time']:
             logger.info("Reached end of available history.")
             break

        # Append
        # Note: Bridge returns oldest first? Or newest first?
        # Usually MT5 CopyRates returns oldest first (index 0 is oldest).
        # Let's check timestamps.
        if df_batch.iloc[0]['time'] > df_batch.iloc[-1]['time']:
            # Descending?
            pass
        
        # Add to list
        # If we are paginating backwards, we need the time of the *first* candle of this batch (oldest) to be the 'to' of next batch?
        # Actually CopyRates(to_date) gets candles *before* that date.
        
        # Let's assume we fetch latest first, then use oldest time as 'to' for next batch.
        # But batch comes as list.
        
        # Strategy:
        # 1. Fetch latest 1000.
        # 2. Take the time of the first element (oldest in batch).
        # 3. Use that as 'to' for next request.
        
        if not all_candles:
             all_candles = batch
        else:
             # Prepend or Append?
             # If we are going backwards in time, we should PREPEND.
             all_candles = batch + all_candles 
        
        last_time = batch[0]['time'] - 1 # One second before the oldest candle
        collected += len(batch)
        logger.info(f"Collected {len(batch)} candles. Total: {len(all_candles)}")
        
        time.sleep(0.5) # Be nice to the server

    # Create DataFrame
    df = pd.DataFrame(all_candles)
    
    if df.empty:
        logger.error("No data collected. Exiting.")
        return

    # Sort by time just in case
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop_duplicates(subset=['time'])
    else:
        logger.error("Data collected but missing 'time' column.")
        return
    
    logger.info(f"Final dataset size: {len(df)} rows")
    
    # Calculate Features
    logger.info("Calculating features...")
    fe = FeatureEngineer()
    df_features = fe.add_technical_features(df)
    
    # Save
    filename = "training_data_continuous.csv"
    df_features.to_csv(filename, index=False)
    logger.info(f"Saved to {filename}")

if __name__ == "__main__":
    collect_data()

