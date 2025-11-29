import json
import glob
import pandas as pd
import os
import tarfile
import io

def process_data(tar_path='backtest_data.tar.gz', output_file='training_data.csv'):
    if not os.path.exists(tar_path):
        print(f"Error: {tar_path} not found.")
        return

    signals = {}
    outcomes = {}

    print(f"Processing {tar_path}...")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
                
            # Read file content
            f = tar.extractfile(member)
            if f:
                try:
                    content = f.read().decode('utf-8', errors='ignore') # Ignore decode errors
                    # Handle multiple JSON objects (JSONL format)
                    for line in content.splitlines():
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            
                            if data.get('type') == 'SIGNAL':
                                signals[data['signal_id']] = data
                            elif data.get('type') == 'OUTCOME':
                                outcomes[data['signal_id']] = data
                                
                        except json.JSONDecodeError:
                            continue
                except Exception as e:
                    print(f"Error reading file {member.name}: {e}")
                    continue

    print(f"Found {len(signals)} signals and {len(outcomes)} outcomes.")

    # Merge Data
    merged_data = []
    for sig_id, sig_data in signals.items():
        row = sig_data.copy()
        
        # Merge outcome
        if sig_id in outcomes:
            outcome_data = outcomes[sig_id]
            row['outcome'] = outcome_data.get('outcome') # 0 or 1
            row['exit_price'] = outcome_data.get('exit_price')
            row['close_time'] = outcome_data.get('close_time')
            
            # New Metrics
            row['mfe'] = outcome_data.get('mfe', 0.0)
            row['mae'] = outcome_data.get('mae', 0.0)
            
            row['has_outcome'] = True
        else:
            row['outcome'] = None
            row['has_outcome'] = False
            row['mfe'] = None
            row['mae'] = None
            
        merged_data.append(row)

    if not merged_data:
        print("No data found.")
        return

    df = pd.DataFrame(merged_data)
    
    # Feature Validation: Ensure new columns exist even if old data is mixed
    new_cols = ['rsi', 'tick_volume', 'spread', 'candle_size', 'wick_upper', 'wick_lower']
    for col in new_cols:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' missing (Old Data?). Filling with 0.")
            df[col] = 0.0

    # Save
    df.to_csv(output_file, index=False)
    print(f"✅ Successfully exported {len(df)} records to {output_file}")
    
    # Preview new columns if available
    preview_cols = ['signal_id', 'outcome', 'mfe', 'mae']
    preview_cols = [c for c in preview_cols if c in df.columns]
    print("Sample data (Outcomes):")
    print(df[preview_cols].head())

if __name__ == "__main__":
    process_data()
