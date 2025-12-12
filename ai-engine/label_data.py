import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Labeler")

def apply_triple_barrier(df, tp_mult=1.5, sl_mult=1.0, time_limit=24):
    """
    Apply Triple Barrier Method to label data for classification.
    
    Args:
        df: DataFrame with OHLCV and 'atr' column
        tp_mult: Take Profit multiplier (x ATR)
        sl_mult: Stop Loss multiplier (x ATR)
        time_limit: Max bars to hold position
        
    Returns:
        df with 'label' column: 0=Wait, 1=Buy, 2=Sell
    """
    logger.info("Starting Triple Barrier Labeling...")
    
    labels = []
    
    # We need future data to label current bar
    # Iterate up to len-time_limit
    for i in tqdm(range(len(df) - time_limit)):
        current_close = df.iloc[i]['close']
        current_atr = df.iloc[i]['atr']
        
        # Define Barriers
        # 1. Buy Scenario barriers
        buy_tp = current_close + (current_atr * tp_mult)
        buy_sl = current_close - (current_atr * sl_mult)
        
        # 2. Sell Scenario barriers
        sell_tp = current_close - (current_atr * tp_mult)
        sell_sl = current_close + (current_atr * sl_mult)
        
        label = 0 # Default: WAIT
        
        # Look forward
        future_window = df.iloc[i+1 : i+1+time_limit]
        
        # Check for Buy Win
        # Using High for TP and Low for SL
        # Find first index where price hits barrier
        
        # Buy Logic:
        # Win if High >= buy_tp
        # Loss if Low <= buy_sl
        
        buy_win_idx = -1
        buy_loss_idx = -1
        
        # Vectorized check within window? Or simple loop for correctness
        # Simple loop is safer for "which happened first"
        
        # Check Buy
        for j in range(len(future_window)):
            bar = future_window.iloc[j]
            if bar['low'] <= buy_sl:
                buy_loss_idx = j
                break # Hit SL first
            if bar['high'] >= buy_tp:
                buy_win_idx = j
                break # Hit TP first
        
        # Check Sell (inverse)
        sell_win_idx = -1
        sell_loss_idx = -1
        
        for j in range(len(future_window)):
            bar = future_window.iloc[j]
            if bar['high'] >= sell_sl:
                sell_loss_idx = j
                break # Hit SL first
            if bar['low'] <= sell_tp:
                sell_win_idx = j
                break # Hit TP first
                
        # Determine Label
        # Priority: If Buy wins, Label 1. If Sell wins, Label 2.
        # Conflict? Rarely happens that both win in same window without hitting SL? 
        # Actually, if volatility is huge, could hit both TPs? Unlikely with SL check.
        
        if buy_win_idx != -1:
            label = 1
        elif sell_win_idx != -1:
            label = 2
        else:
            label = 0 # Neither hit TP (Time exit or Chop)
            
        labels.append(label)
        
    # Fill remaining
    labels.extend([0] * time_limit)
    
    df['label'] = labels
    
    # Stats
    vc = df['label'].value_counts()
    logger.info(f"Label Distribution:\n{vc}")
    
    return df

if __name__ == "__main__":
    try:
        logger.info("Loading training_data_continuous.csv...")
        df = pd.read_csv("training_data_continuous.csv")
        
        # Ensure ATR exists (it should from FeatureEngineer)
        if 'atr' not in df.columns:
            logger.error("ATR column missing. Please ensure FeatureEngineer ran correctly.")
            exit(1)
            
        df_labeled = apply_triple_barrier(df)
        
        df_labeled.to_csv("labeled_data.csv", index=False)
        logger.info("✅ Saved to labeled_data.csv")
        
    except FileNotFoundError:
        logger.error("training_data_continuous.csv not found. Run collect_continuous_data.py first.")
