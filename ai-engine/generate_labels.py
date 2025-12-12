import pandas as pd
import numpy as np
import glob
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LabelGen")

def generate_labels():
    input_pattern = "training_data_*.csv"
    output_file = "training_data_enhanced.csv" # Direct to enhanced for training
    
    files = glob.glob(input_pattern)
    if not files:
        logger.error("No input files found!")
        return

    all_data = []

    for f in files:
        logger.info(f"Processing {f}...")
        try:
            df = pd.read_csv(f)
            
            # 1. Normalize Symbol Names
            # 'bgp' -> 'GBPUSD' (heuristic based on filename if symbol col is weird)
            filename = os.path.basename(f).lower()
            if 'bgp' in filename:
                df['symbol'] = 'GBPUSD'
            elif 'btc' in filename:
                df['symbol'] = 'BTCUSD'
            elif 'nas100' in filename:
                df['symbol'] = 'NAS100'
            elif 'us30' in filename:
                df['symbol'] = 'US30'
            
            # Ensure sort by time
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 2. Generate MFE Labels (Simulate perfect foresight)
            # Look ahead 60 bars (approx 1 hour on 1m chart)
            LOOKAHEAD = 60
            
            # Vectorized Lookahead
            # We want: Max(High[i+1:i+60]) - Close[i]  (Long Potential)
            #          Close[i] - Min(Low[i+1:i+60])  (Short Potential)
            
            # Prepare Rolling Windows (using pandas rolling backward, so we reverse)
            reversed_high = df['high'][::-1]
            reversed_low = df['low'][::-1]
            
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD)
            fwd_max_high = df['high'].rolling(window=indexer, min_periods=1).max()
            fwd_min_low = df['low'].rolling(window=indexer, min_periods=1).min()
            
            # Calculate Potentials
            long_profit = fwd_max_high - df['close']
            short_profit = df['close'] - fwd_min_low
            
            # Assign Label: We want the model to predict the *Max Potential* regardless of direction?
            # Or specifically direction? 
            # Current `train_filter.py` treats `mfe` as a regression target for quality.
            # Let's assign `mfe` as the MAXIMUM of either direction.
            # And `outcome` as direction? (1 for Long, -1 for Short)
            
            df['mfe'] = np.maximum(long_profit, short_profit)
            
            # Filter Noise: If MFE is tiny, it's virtually 0
            # Calculate ATR for normalization (simple rolling)
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(14).mean().bfill()
            
            # Normalize MFE by ATR (MFE Ratio)
            # This makes the target "Likelihood of >1R move"
            # Overwrite mfe absolute with mfe_ratio??
            # train_filter uses raw MFE currently. Let's stick to raw but add a ratio column.
            
            # Add synthetic columns expected by train_filter
            df['mae'] = 0.0 # Perfect entry simulation
            df['pnl'] = df['mfe'] # Optimistic
            df['outcome'] = np.where(long_profit > short_profit, 1, -1) # "perfect direction"
            
            # Drop the last LOOKAHEAD rows (cannot know future)
            df = df.iloc[:-LOOKAHEAD].copy()
            
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Failed to process {f}: {e}")

    if not all_data:
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    logger.info(f"✅ Generated labels for {len(final_df)} rows. Saved to {output_file}")

if __name__ == "__main__":
    generate_labels()
