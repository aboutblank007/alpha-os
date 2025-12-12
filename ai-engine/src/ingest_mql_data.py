import os
import json
import pandas as pd
import glob
import logging
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataIngestion")

# Configuration
# Default to standard MT5 Wine path or a local 'data/signals' folder for testing
DEFAULT_MQL5_DIR = os.path.expanduser("~/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals")
MQL5_FILES_DIR = os.environ.get("SIGNAL_DIR", DEFAULT_MQL5_DIR)
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_data.csv")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def ingest_data(source_dir=MQL5_FILES_DIR):
    logger.info(f"🔍 Looking for data in: {source_dir}")
    
    merged_data = []
    
    # --- 1. Fetch from Supabase (Real-Time Source) ---
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            logger.info("📡 Connecting to Supabase...")
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Fetch TWO types of training examples:
            # 1. Completed Trades (Positive + Negative Outcomes)
            # 2. AI Decisions to WAIT/SCAN (Negative Samples - "Don't Trade")
            
            # Strategy: Pull completed trades + sample of WAIT/SCAN decisions
            # To avoid overwhelming with WAIT samples (13k+), limit to recent or sample
            
            # 1. All Completed Trades (from training_signals)
            logger.info("   📊 Fetching completed trades from 'training_signals'...")
            response_completed = supabase.table("training_signals") \
                .select("*") \
                .not_.is_("result_profit", "null") \
                .execute()
            
            completed_trades = response_completed.data if response_completed.data else []
            logger.info(f"   ✅ Found {len(completed_trades)} completed trades")
            
            # 2. Market Scans aka Negative Samples (from market_scans)
            # These are WAIT/SCAN signals that we will treat as potential trades for Counterfactual Sim
            logger.info("   🚫 Fetching market scans (from 'market_scans')...")
            response_scans = supabase.table("market_scans") \
                .select("*") \
                .order("timestamp", desc=True) \
                .limit(20000) \
                .execute()
            
            market_scans = response_scans.data if response_scans.data else []
            logger.info(f"   ✅ Found {len(market_scans)} market scans (negative samples)")

            # 3. Legacy Negative Samples (from training_signals, before split)
            # We want to use the 10k+ historical WAIT signals for simulation too!
            logger.info("   🕰️ Fetching legacy negative samples (from 'training_signals')...")
            response_legacy = supabase.table("training_signals") \
                .select("*") \
                .is_("result_profit", "null") \
                .not_.is_("ai_features", "null") \
                .order("timestamp", desc=True) \
                .limit(15000) \
                .execute()
                
            legacy_negatives = response_legacy.data if response_legacy.data else []
            logger.info(f"   ✅ Found {len(legacy_negatives)} legacy negative samples")

            # Merge all datasets
            # Notes: 'completed_trades' have real PnL. 'market_scans' and 'legacy_negatives' will be simulated.
            db_data = completed_trades + market_scans + legacy_negatives
            
            if db_data:
                logger.info(f"✅ Fetched {len(db_data)} total records from Supabase "
                            f"({len(completed_trades)} trades + {len(market_scans)} scans + {len(legacy_negatives)} legacy).")
                for row in db_data:
                    # Flatten the row
                    flat_row = row.copy()
                    
                    # For negative samples (WAIT/SCAN), create synthetic labels
                    # These represent "correctly avoided a trade"
                    if row.get('result_profit') is None:
                        # Negative sample: AI decided not to trade
                        # Label: result_profit = 0 (neutral), result_mfe = 0 (no move)
                        # This teaches model "it's OK to wait"
                        flat_row['result_profit'] = 0.0
                        flat_row['result_mfe'] = 0.0  # Target for regression
                        flat_row['result_mae'] = 0.0
                        flat_row['result_win'] = False
                        flat_row['is_negative_sample'] = True
                    else:
                        flat_row['is_negative_sample'] = False
                    
                    # Extract AI-computed features from JSONB column if present
                    if row.get('ai_features'):
                        import json
                        try:
                            ai_feats = row['ai_features']
                            if isinstance(ai_feats, str):
                                ai_feats = json.loads(ai_feats)
                            if isinstance(ai_feats, dict):
                                flat_row.update(ai_feats)
                                logger.debug(f"Merged {len(ai_feats)} AI features for {row.get('signal_id')}")
                        except Exception as e:
                            logger.warning(f"Failed to parse ai_features: {e}")
                    
                    flat_row["source"] = "supabase"
                    merged_data.append(flat_row)
            else:
                logger.info("ℹ️ No training data found in Supabase yet.")
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch from Supabase: {e}")
            
    else:
        logger.warning("⚠️ Supabase credentials not found. Skipping DB ingestion.")

    # --- 2. Fetch from Local Files (Legacy/Offline Source) ---
    # DISABLED: User requested Online-Only Data
    # if os.path.exists(source_dir):
    #     # Patterns for new data collector format
    #     signal_files = glob.glob(os.path.join(source_dir, "signals_*.json"))
    #     outcome_files = glob.glob(os.path.join(source_dir, "outcomes_*.json"))
        
    #     if signal_files:
    #         signals_map = {}
    #         outcomes_map = {}

    #         # Read Signals
    #         for file_path in signal_files:
    #             try:
    #                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    #                     for line in f:
    #                         if not line.strip(): continue
    #                         try:
    #                             data = json.loads(line)
    #                             if data.get("type") == "SIGNAL":
    #                                 signals_map[data["signal_id"]] = data
    #                         except json.JSONDecodeError:
    #                             continue
    #             except Exception as e:
    #                 logger.error(f"Error reading {file_path}: {e}")

    #         # Read Outcomes
    #         for file_path in outcome_files:
    #             try:
    #                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    #                     for line in f:
    #                         if not line.strip(): continue
    #                         try:
    #                             data = json.loads(line)
    #                             if data.get("type") == "OUTCOME":
    #                                 outcomes_map[data["signal_id"]] = data
    #                         except json.JSONDecodeError:
    #                             continue
    #             except Exception as e:
    #                 logger.error(f"Error reading {file_path}: {e}")

    #         logger.info(f"Found {len(signals_map)} signals and {len(outcomes_map)} outcomes locally.")

    #         # Merge Local Data
    #         for signal_id, signal in signals_map.items():
    #             outcome = outcomes_map.get(signal_id)
    #             if outcome:
    #                 row = signal.copy()
    #                 row["outcome"] = outcome.get("outcome")
    #                 row["mfe"] = outcome.get("mfe")
    #                 row["mae"] = outcome.get("mae")
    #                 row["exit_price"] = outcome.get("exit_price")
    #                 row["close_time"] = outcome.get("close_time")
    #                 row["has_outcome"] = True
    #                 row["source"] = "local_file"
    #                 if "type" in row: del row["type"]
    #                 merged_data.append(row)

    # --- 3. Final Processing ---
    if not merged_data:
        logger.info("⚠️ No training data found (DB or Local).")
        return

    new_df = pd.DataFrame(merged_data)
    
    # Determine columns to keep? Or just keep all for feature engineering
    # Cleanups
    if "created_at" in new_df.columns:
        new_df["timestamp"] = pd.to_datetime(new_df["created_at"]).astype(int) / 10**9
        
    # Deduplication
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            # Merge and deduplicate based on signal_id
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=["signal_id"], keep="last")
        except Exception as e:
            logger.error(f"Error reading existing CSV: {e}. Overwriting.")
            combined_df = new_df
    else:
        combined_df = new_df

    # Save
    try:
        combined_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"✅ Saved {len(combined_df)} records to {OUTPUT_CSV}")
        
        # Stats
        if "outcome" in combined_df.columns:
            logger.info(f"\n📊 Class Balance:\n{combined_df['outcome'].value_counts(normalize=True)}")
            
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")

if __name__ == "__main__":
    ingest_data()

