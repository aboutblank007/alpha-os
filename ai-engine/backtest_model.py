import pandas as pd
import numpy as np
import lightgbm as lgb
# import matplotlib.pyplot as plt
import os

# Settings
MODEL_PATH = 'ai-engine/models/lgbm_scalping_v2.txt'
DATA_FILE = 'training_data_enhanced.csv'
OUTPUT_DIR = 'analysis_results'
MFE_THRESHOLD = 2.5  # From your analysis (Top 20% was 3.96, so 2.5 is a conservative start)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def run_backtest():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Data & Model
    df = load_data(DATA_FILE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=MODEL_PATH)

    # 2. Prepare Features (Must match training exactly)
    feature_names = model.feature_name()
    print(f"Model expects features: {feature_names}")
    
    # Check for missing columns
    missing_cols = [f for f in feature_names if f not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in CSV: {missing_cols}")
        # Fill with 0 for now just to run
        for c in missing_cols:
            df[c] = 0.0

    X = df[feature_names]

    # 3. Predict
    print("Running inference...")
    df['predicted_mfe'] = model.predict(X)

    # 4. Simulation Logic
    print(f"Simulating trades with MFE Threshold > {MFE_THRESHOLD}...")
    
    # --- Realistic Simulation ---
    def simulate_trade(row, prediction):
        # Dynamic TP based on confidence
        # If prediction is high, aim higher. If low (but passed filter), aim conservative.
        
        # Base Params
        entry_price = row['price']
        atr = row['atr'] if row['atr'] > 0 else 1.0 # Fallback
        
        # AI Dynamic TP
        # Strategy: Aim for 80% of predicted MFE
        # But respect a minimum R:R of 1:1 based on ATR
        target_dist = max(prediction * 0.8, atr * 1.0)
        stop_dist = atr * 1.0 # Fixed 1ATR risk
        
        # Actual Market Data
        actual_mfe = row.get('mfe', 0)
        actual_mae = row.get('mae', 0)
        
        # Outcome Logic (Conservative)
        # 1. Did we hit SL?
        if actual_mae >= stop_dist:
            return -stop_dist # Loss
            
        # 2. Did we hit TP?
        if actual_mfe >= target_dist:
            return target_dist # Win
            
        # 3. Time out / Close at end
        # If exit_price exists, use it. Else estimate.
        if 'exit_price' in row and row['exit_price'] > 0:
            if 'BUY' in row['action']:
                return row['exit_price'] - entry_price
            else:
                return entry_price - row['exit_price']
        
        return 0.0 # Break even if no data

    # Vectorize is hard with complex logic, use apply
    df['pnl_ai'] = df.apply(lambda x: simulate_trade(x, x['predicted_mfe']) if x['predicted_mfe'] > MFE_THRESHOLD else 0.0, axis=1)
    
    # Baseline PnL (Fixed 1.5 TP / 1.0 SL)
    def simulate_baseline(row):
        atr = row['atr'] if row['atr'] > 0 else 1.0
        tp = atr * 1.5
        sl = atr * 1.0
        mfe = row.get('mfe', 0)
        mae = row.get('mae', 0)
        
        if mae >= sl: return -sl
        if mfe >= tp: return tp
        return 0.0 # Simplify
        
    df['pnl_raw'] = df.apply(simulate_baseline, axis=1)

    df['executed'] = df['predicted_mfe'] > MFE_THRESHOLD
    
    # 5. Metrics
    total_trades = len(df)
    executed_trades = df['executed'].sum()
    
    # Stats for Baseline
    base_wins = (df['pnl_raw'] > 0).sum()
    base_wr = base_wins / total_trades * 100 if total_trades > 0 else 0
    base_avg = df['pnl_raw'].mean()
    
    # Stats for AI
    ai_trades_df = df[df['executed']]
    ai_wins = (ai_trades_df['pnl_ai'] > 0).sum()
    ai_wr = ai_wins / executed_trades * 100 if executed_trades > 0 else 0
    ai_avg = ai_trades_df['pnl_ai'].mean()
    
    # Sharpe (approx)
    risk_free = 0
    ai_std = ai_trades_df['pnl_ai'].std()
    ai_sharpe = (ai_avg - risk_free) / ai_std * np.sqrt(len(ai_trades_df)) if ai_std > 0 else 0
    
    raw_cum_pnl = df['pnl_raw'].cumsum()
    ai_cum_pnl = df['pnl_ai'].cumsum()
    
    print(f"\n=== Backtest Results (Dynamic TP) ===")
    print(f"Baseline: {total_trades} trades | Win Rate: {base_wr:.1f}% | Avg PnL: {base_avg:.2f} | Total: {raw_cum_pnl.iloc[-1]:.2f}")
    print(f"AI Model: {executed_trades} trades | Win Rate: {ai_wr:.1f}% | Avg PnL: {ai_avg:.2f} | Total: {ai_cum_pnl.iloc[-1]:.2f}")
    print(f"AI Sharpe Ratio: {ai_sharpe:.2f}")
    print(f"Improvement: {(ai_avg - base_avg) / base_avg * 100:.1f}% per trade")
    
    # 6. Plot - Skipped for headless exec
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['timestamp'], raw_cum_pnl, label='Baseline Strategy', alpha=0.6)
    # plt.plot(df['timestamp'], ai_cum_pnl, label='AI Filtered (>2.5 MFE)', linewidth=2)
    # plt.title(f'Strategy Comparison: Baseline vs AI Scalper (Thresh={MFE_THRESHOLD})')
    # plt.xlabel('Date')
    # plt.ylabel('Cumulative PnL (Points)')
    # plt.legend()
    # plt.grid(True)
    # 
    # out_file = os.path.join(OUTPUT_DIR, 'backtest_comparison.png')
    # plt.savefig(out_file)
    # print(f"Chart saved to {out_file}")

if __name__ == "__main__":
    run_backtest()

