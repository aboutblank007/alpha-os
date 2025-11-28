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
                content = f.read().decode('utf-8')
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

    print(f"Found {len(signals)} signals and {len(outcomes)} outcomes.")

    # Merge Data
    merged_data = []
    for sig_id, sig_data in signals.items():
        row = sig_data.copy()
        
        # Remove raw feature object if flattened, or keep if needed. 
        # The JSON usually has flat fields from the EA structure.
        
        # Merge outcome
        if sig_id in outcomes:
            outcome_data = outcomes[sig_id]
            row['outcome'] = outcome_data.get('outcome') # 0 or 1
            row['exit_price'] = outcome_data.get('exit_price')
            row['close_time'] = outcome_data.get('close_time')
            row['has_outcome'] = True
        else:
            row['outcome'] = None
            row['has_outcome'] = False
            
        merged_data.append(row)

    if not merged_data:
        print("No data found.")
        return

    df = pd.DataFrame(merged_data)
    
    # Cleanup columns
    cols_to_drop = ['type', 'action', 'symbol'] # Optional cleanup
    # Keep strictly numeric + outcome for training, but keep ID for reference
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"✅ Successfully exported {len(df)} records to {output_file}")
    print("Sample data:")
    print(df[['signal_id', 'outcome', 'has_outcome']].head())

if __name__ == "__main__":
    process_data()

