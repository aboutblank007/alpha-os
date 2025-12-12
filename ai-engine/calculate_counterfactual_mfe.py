"""
反事实 MFE 计算工具

为负样本（WAIT/SCAN决策）计算"如果交易会怎样"的MFE标签。
使用 training_signals 表的时间序列数据进行回测模拟。
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CounterfactualMFE")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
LOOKFORWARD_BARS = 12  # 向后看12根K线（12分钟）

def calculate_counterfactual_mfe_from_signals(df):
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
    logger.info("🔮 Running FULL Counterfactual Trade Simulation...")
    
    # Ensure sorted for simulation
    df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
    
    # New columns
    if 'is_simulated' not in df.columns:
        df['is_simulated'] = False
        
    updated_count = 0
    
    # Constants
    MAX_HOLD_BARS = 240  # 4 hours max hold time simulation
    
    # Iterate by symbol to speed up and isolate context
    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_df = df[symbol_mask].copy()
        
        # Get indices of negative samples to simulate
        # Must have SL and TP to simulate
        neg_indices = symbol_df[
            (symbol_df['is_negative_sample'] == True) & 
            (symbol_df['sl'].notna()) & 
            (symbol_df['tp'].notna())
        ].index
        
        if len(neg_indices) == 0:
            continue
            
        logger.info(f"   Simulating {len(neg_indices)} trades for {symbol}...")
        
        # Pre-convert columns to numpy for speed
        timestamps = symbol_df['timestamp'].values
        opens = symbol_df['price'].values # Assuming 'price' is entry/close
        highs = symbol_df['price'].values # Approximate if High not avail, else parse ai_features
        lows = symbol_df['price'].values  # Approximate
        # Note: If we have candles, use them. If only 'price', simulation is approximate (Close-to-Close)
        # Better: Try to parse High/Low from ai_features if available? similar to previous script
        
        # Let's try to get more granular data if possible, otherwise use 'price' column as [Open, High, Low, Close] approximation
        # Ideally we fetch candles. For now, we use the density of signals.
        
        # Loop strictly logic
        final_profits = []
        final_mfes = []
        
        for idx in neg_indices:
            # Current Row
            row = symbol_df.loc[idx]
            entry_price = float(row.get('price', 0))
            sl = float(row.get('sl'))
            tp = float(row.get('tp'))
            
            # Determine direction based on SL/TP if action is ambiguous
            direction = 0 # 1 for Buy, -1 for Sell
            
            # Use 'action' if available/reliable, else infer
            action_str = str(row.get('ai_action', '')).upper()
            if 'BUY' in action_str:
                direction = 1
            elif 'SELL' in action_str:
                direction = -1
            else:
                # Infer from SL/TP
                if tp > entry_price and sl < entry_price:
                    direction = 1
                elif tp < entry_price and sl > entry_price:
                    direction = -1
            
            if direction == 0 or entry_price == 0:
                continue
                
            # Find integer index in the numpy arrays
            # current_pos is the index within symbol_df, need to match with numpy array index
            # symbol_df is a slice, so we need local integer index
            local_idx = symbol_df.index.get_loc(idx)
            
            # Simulation loop
            # Start from next bar
            outcome = None # 'TP', 'SL', 'TIMEOUT'
            exit_price = entry_price
            
            mfe_price = entry_price
            mae_price = entry_price
            
            # Max lookahead
            end_search = min(len(timestamps), local_idx + MAX_HOLD_BARS)
            
            for i in range(local_idx + 1, end_search):
                curr_p = opens[i] # Approximate with available price
                
                # Check metrics (assuming curr_p is the "tick" or candle close)
                # In a real candle, we'd check Low vs SL and High vs TP.
                # Here we only have discrete points.
                
                # Update MAE/MFE
                if direction == 1: # BUY
                    if curr_p > mfe_price: mfe_price = curr_p
                    if curr_p < mae_price: mae_price = curr_p
                    
                    # Check Exit
                    if curr_p <= sl:
                        outcome = 'SL'
                        exit_price = sl # Slippage ignored
                        break
                    if curr_p >= tp:
                        outcome = 'TP'
                        exit_price = tp
                        break
                        
                else: # SELL
                    if curr_p < mfe_price: mfe_price = curr_p
                    if curr_p > mae_price: mae_price = curr_p
                    
                    # Check Exit
                    if curr_p >= sl:
                        outcome = 'SL'
                        exit_price = sl
                        break
                    if curr_p <= tp:
                        outcome = 'TP'
                        exit_price = tp
                        break
            
            # If loop finished without outcome
            if outcome is None:
                outcome = 'TIMEOUT'
                # Close at last seen price
                if local_idx + 1 < len(opens):
                    exit_price = opens[min(len(opens)-1, end_search-1)]
                
            # Calculate Profit (PIP scale?)
            # Simplified raw price diff
            if direction == 1:
                profit_raw = exit_price - entry_price
                mfe_raw = mfe_price - entry_price
            else:
                profit_raw = entry_price - exit_price
                mfe_raw = entry_price - mfe_price
            
            # Normalize to some standard if needed, or store raw.
            # Existing data 'result_profit' seems to be currency or pips?
            # User sample: profit 4.2. XAUUSD price ~2600. 4.2 profit on 0.01 lot? 
            # Let's try to match existing format.
            # If we don't know pip value, maybe just store raw price diff for now?
            # Or use 'pip_scale' column if exists.
            
            pip_scale = float(row.get('point', 0.00001))
            if pip_scale == 0: pip_scale = 0.00001
                
            # Existing result_mfe is "ratio to ATR" in previous code?
            # Previous logic: mfe = (max_price - entry_price) / atr
            atr = float(row.get('atr', 0.0001))
            if atr == 0: atr = 0.0001
            
            norm_mfe = mfe_raw / atr
            norm_mfe = np.clip(norm_mfe, -2, 10)
            
            # Update DataFrame
            df.loc[idx, 'result_profit'] = profit_raw # Estimated
            df.loc[idx, 'result_mfe'] = norm_mfe
            df.loc[idx, 'is_simulated'] = True
            
            updated_count += 1
            
    logger.info(f"   ✅ Simulated trades for {updated_count} samples.")
    return df


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.append(os.path.dirname(__file__))
    
    # 加载训练数据
    df = pd.read_csv("training_data_enhanced.csv")
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Negative samples: {(df['is_negative_sample'] == True).sum()}")
    
    # 计算反事实MFE
    df = calculate_counterfactual_mfe_from_signals(df)
    
    # 保存
    df.to_csv("training_data_enhanced_counterfactual.csv", index=False)
    logger.info(f"✅ Saved to training_data_enhanced_counterfactual.csv")
