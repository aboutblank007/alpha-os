import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime, timedelta

# Add src to path to import FeatureEngineer
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from features import FeatureEngineer

# Supabase for counterfactual MFE calculation
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhanceFeatures")

def calculate_counterfactual_mfe(df, lookforward_bars=240):
    """
    Simulation of trade outcomes for unexecuted signals (Counterfactual Analysis).
    
    Logic:
    1. Sort data by symbol and timestamp.
    2. For each 'WAIT'/'SCAN' signal (negative sample) that has valid SL and TP:
       - Simulate a trade entry at proper price.
       - Iterate forward through subsequent records (acting as price history).
       - Check if SL or TP is hit.
       - Calculate virtual profit, MFE, MAE.
    
    Returns:
        DataFrame with updated 'result_profit', 'result_mfe', and 'is_simulated' columns.
    """
    if 'is_simulated' not in df.columns:
        df['is_simulated'] = False
        
    # Validation using SUPABASE credentials if needed (can be optional here as we use in-memory data for simulation)
    # But current logic relies on in-memory iteration of the dataframe itself, assuming it contains price history.
    # The previous logic queried Supabase.
    # If df is dense (M1), in-memory is better. If sparse, we might need Supabase.
    # User confirmed we want full simulation. Let's assume input df (from training_signals) has enough density or we accept approximation.
    # Note: training_signals contains ALL signals (WAIT/SCAN/TRADE). If AI runs every minute, it's dense.
    
    logger.info("🔮 Running FULL In-Memory Counterfactual Trade Simulation...")
    
    # Ensure sorted
    df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
    
    updated_count = 0
    MAX_HOLD_BARS = lookforward_bars
    
    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_df = df[symbol_mask].copy()
        
        # We simulate negative samples that have SL/TP
        neg_indices = symbol_df[
            (symbol_df['is_negative_sample'] == True) & 
            (symbol_df['sl'].notna()) & 
            (symbol_df['tp'].notna())
        ].index
        
        if len(neg_indices) == 0:
            continue
            
        logger.info(f"   Simulating {len(neg_indices)} virtual trades for {symbol}...")
        
        # Arrays for speed
        timestamps = symbol_df['timestamp'].values
        opens = symbol_df['price'].values
        # Using open/price as proxy for granular price action
        
        for idx in neg_indices:
            row = symbol_df.loc[idx]
            entry_price = float(row.get('price', 0))
            if entry_price == 0: continue
                
            sl = float(row.get('sl'))
            tp = float(row.get('tp'))
            
            # Determine direction
            direction = 0
            # Try to infer from SL/TP first as it's definitive for the intended trade
            if tp > entry_price and sl < entry_price:
                direction = 1 # BUY
            elif tp < entry_price and sl > entry_price:
                direction = -1 # SELL
            else:
                # Fallback to action string
                action_str = str(row.get('ai_action', '')).upper()
                if 'BUY' in action_str: direction = 1
                elif 'SELL' in action_str: direction = -1
            
            if direction == 0: continue
            
            # Simulation
            local_idx = symbol_df.index.get_loc(idx)
            outcome = None
            exit_price = entry_price
            mfe_price = entry_price
            mae_price = entry_price
            
            end_pos = min(len(timestamps), local_idx + MAX_HOLD_BARS)
            
            # Scan forward
            # Optim: could use numpy vectorization, but loop is readable and acceptable for <20k rows
            # We start from next bar
            for i in range(local_idx + 1, end_pos):
                curr_p = opens[i]
                
                if direction == 1:
                    if curr_p > mfe_price: mfe_price = curr_p
                    if curr_p < mae_price: mae_price = curr_p
                    if curr_p <= sl:
                        outcome = 'SL'
                        exit_price = sl
                        break
                    if curr_p >= tp:
                        outcome = 'TP'
                        exit_price = tp
                        break
                else:
                    if curr_p < mfe_price: mfe_price = curr_p
                    if curr_p > mae_price: mae_price = curr_p
                    if curr_p >= sl:
                        outcome = 'SL'
                        exit_price = sl
                        break
                    if curr_p <= tp:
                        outcome = 'TP'
                        exit_price = tp
                        break
            
            # Check if timed out
            if outcome is None:
                # Close at last available price
                if end_pos > local_idx + 1:
                     exit_price = opens[end_pos - 1]
                     outcome = 'TIMEOUT'
            
            # Calculate metrics
            pip_scale = float(row.get('point', 0.00001)) or 0.00001
            atr = float(row.get('atr', 0.0001)) or 0.0001
            
            if direction == 1:
                profit_raw = exit_price - entry_price
                mfe_raw = mfe_price - entry_price
            else:
                profit_raw = entry_price - exit_price
                mfe_raw = entry_price - mfe_price
                
            norm_mfe = np.clip(mfe_raw / atr, -2, 10)
            
            # Update
            df.loc[idx, 'result_profit'] = profit_raw
            df.loc[idx, 'result_mfe'] = norm_mfe
            df.loc[idx, 'is_simulated'] = True
            
            updated_count += 1

    logger.info(f"   ✅ Simulated {updated_count:,}/{len(df[df['is_negative_sample']==True]):,} negative samples.")
    return df

INPUT_FILE = 'training_data.csv'
OUTPUT_FILE = 'training_data_enhanced.csv'

def enhance_features():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Please place the CSV from DataCollector here.")
        return

    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"   Original shape: {df.shape}")
    
    # Initialize Feature Engineer
    fe = FeatureEngineer()
    
    # --- PRE-PROCESSING FOR SUPABASE DATA ---
    # 1. Map 'price' to 'close' if missing
    if 'close' not in df.columns and 'price' in df.columns:
        print("ℹ️  Mapping 'price' to 'close'...")
        df['close'] = df['price']

    # 2. Map EA-specific column names
    if 'avg_dom_imbalance' in df.columns:
        df['dom_imbalance'] = df['avg_dom_imbalance']
    if 'volatility_skew_proxy' in df.columns:
        df['volatility_skew'] = df['volatility_skew_proxy']

    # 3. Check if Technical Features already exist (from Supabase/EA)
    tech_cols = ['ema_short', 'ema_long', 'atr', 'adx', 'rsi']
    has_tech = all(col in df.columns for col in tech_cols)
    
    if has_tech:
        print("✅ Technical features present. Skipping recalculation.")
        # Ensure derived columns like 'pip_scale' are set for subsequent steps
        if 'pip_scale' not in df.columns:
             df['pip_scale'] = df['close'].apply(fe._infer_pip_scale)
    else:
        # 2. Add Technical Features (Only if missing)
        print("⚙️  Calculating Technical Features...")
        try:
            df = fe.add_technical_features(df)
        except KeyError as e:
            print(f"⚠️  Skipping Technical Features due to missing columns: {e}")
    
    # 4. Add Derived Features
    print("⚙️  Calculating Derived Features...")
    try:
        df = fe.add_derived_features(df)
    except Exception as e:
        print(f"⚠️  Skipping Derived Features: {e}")
    
    # 5. Add Microstructure Features
    print("⚙️  Calculating Microstructure Features...")
    try:
        # Create dummy high/low if missing, to allow some calculations
        if 'high' not in df.columns and 'candle_size' in df.columns:
            # Approximation: High = Close + WickUpper? No, depends on candle type.
            # We only really need high/low for CLV.
            # If we lack high/low, add_microstructure might fail or produce garbage.
            # Let's try, and catch error.
            pass
        
        df = fe.add_microstructure_features(df)
    except Exception as e:
        print(f"⚠️  Skipping Microstructure Features: {e}")
    
    # 6. Generate Labels (MFE Target) if missing
    if 'result_mfe' in df.columns:
        print("✅ Found 'result_mfe' from Supabase. Using as target.")
        df['mfe'] = df['result_mfe']
        # Also clean up string "null" or NaNs
        df['mfe'] = pd.to_numeric(df['mfe'], errors='coerce')
    elif 'mfe' not in df.columns:
        print("🏷️  Generating synthetic training labels (MFE)...")
        # ... (original logic) ...
        # Logic: Assume Trend Following strategy (EMA Crossover)
        # Look ahead 12 bars to find max potential profit (R-multiple)
        
        if 'high' in df.columns and 'low' in df.columns:
            LOOKAHEAD = 12
            
            # Calculate Forward High/Low over N bars
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD)
            df['fwd_high'] = df['high'].rolling(window=indexer).max()
            df['fwd_low'] = df['low'].rolling(window=indexer).min()
            
            # Determine Trend Direction (Proxy for Signal)
            # 1 = Buy (EMA12 > EMA26), -1 = Sell
            if 'ema_short' in df.columns:
                df['trend_dir'] = np.where(df['ema_short'] > df['ema_long'], 1, -1)
            else:
                df['trend_dir'] = 0
            
            # Calculate MFE (R-Multiple)
            # Buy MFE: (MaxHigh - Entry) / ATR
            # Sell MFE: (Entry - MinLow) / ATR
            # Entry assumed to be Close price of signal bar
            
            mfe_buy = (df['fwd_high'] - df['close']) / (df['atr'] + 1e-8)
            mfe_sell = (df['close'] - df['fwd_low']) / (df['atr'] + 1e-8)
            
            df['mfe'] = np.where(df['trend_dir'] == 1, mfe_buy, mfe_sell)
            
            # Clip extreme values (e.g., flash crashes or bad data)
            df['mfe'] = df['mfe'].clip(lower=-2, upper=10)
            
            print("   ✅ Generated 'mfe' target (ATR-normalized R-multiple)")
        else:
            print("⚠️  Cannot generate MFE labels: Missing 'high'/'low' columns.")

    # 7. Handle NaNs
    original_len = len(df)
    
    # Check NaN status
    nan_counts = df.isna().sum()
    print("   NaN counts per column (Top 10):")
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))
    
    # Only drop rows where TARGET is missing
    if 'mfe' in df.columns:
        df.dropna(subset=['mfe'], inplace=True)
    
    # For other features, fill with 0 or mean, or let LightGBM handle it.
    # We'll fill critical technicals if missing, but LightGBM is robust.
    # But let's drop rows where 'close' is missing as it's fundamental.
    if 'close' in df.columns:
        df.dropna(subset=['close'], inplace=True)
        
    dropped = original_len - len(df)
    print(f"   Dropped {dropped} rows due to missing Target/Close.")
    
    # 8. Calculate Counterfactual MFE for negative samples
    if 'is_negative_sample' in df.columns:
        neg_count = (df['is_negative_sample'] == True).sum()
        if neg_count > 0:
            try:
                df = calculate_counterfactual_mfe(df, lookforward_bars=12)
            except Exception as e:
                logger.error(f"⚠️ Counterfactual MFE failed: {e}")
                import traceback
                traceback.print_exc()
    
    # 9. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved enhanced training data to {OUTPUT_FILE} (Shape: {df.shape})")
    print("   Ready for training with train_filter.py!")

if __name__ == "__main__":
    enhance_features()
