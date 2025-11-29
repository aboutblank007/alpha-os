import pandas as pd
import os

file_path = 'training_data.csv'

if not os.path.exists(file_path):
    # Try to find it in ai-engine if not in root
    file_path = 'ai-engine/training_data.csv'

if os.path.exists(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    cols_to_check = ['volatility_ok', 'chop_ok', 'trend_filter_ok', 'slope_ok', 'distance_ok']
    
    for col in cols_to_check:
        if col in df.columns:
            print(f"\n=== {col} Distribution ===")
            print(df[col].value_counts(normalize=True))
            
            if 'outcome' in df.columns:
                print(f"Win Rate by {col}:")
                print(df.groupby(col)['outcome'].mean())
        else:
            print(f"\n⚠️ Column {col} not found in CSV.")
else:
    print("❌ training_data.csv not found.")

