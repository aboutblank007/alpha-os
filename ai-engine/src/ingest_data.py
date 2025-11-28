import os
import json
import pandas as pd
import glob

# Configuration
MQL5_FILES_DIR = os.path.expanduser("~/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals")
# Note: Adjust MQL5_FILES_DIR to your actual MT5 Data Folder path if different. 
# On macOS with Crossover/Wine, it's deep in the Library folder.
# If you are copying files manually, set this to your local folder.

OUTPUT_CSV = "ai-engine/data/training_dataset.csv"

def ingest_data(source_dir=MQL5_FILES_DIR):
    print(f"🔍 Looking for data in: {source_dir}")
    
    json_files = glob.glob(os.path.join(source_dir, "training_data_*.json"))
    
    if not json_files:
        print("❌ No training data found. Run PivotTrend_DataCollector in MT5 Strategy Tester first.")
        # Fallback for testing/dev if files are locally in project
        local_fallback = "data/signals"
        if os.path.exists(local_fallback):
            json_files = glob.glob(os.path.join(local_fallback, "*.json"))
        
        if not json_files:
            return

    all_data = []
    
    for file_path in json_files:
        print(f"📄 Processing {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # The file contains line-separated JSON objects (NDJSON-like)
            # Or sometimes it might be a single array if we changed logic, but current logic is append line-by-line.
            content = f.read()
            
            # Fix: MQL5 might write messy JSON if appended without newlines properly, 
            # but our code adds \n. Let's split by newline.
            lines = content.strip().split('\n')
            
            for line in lines:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error decoding line: {e}")
                    continue

    if not all_data:
        print("⚠️ No valid data rows found.")
        return

    df = pd.DataFrame(all_data)
    
    # Basic cleaning
    df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='s')
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Successfully saved {len(df)} samples to {OUTPUT_CSV}")
    print(df.head())
    print("\n📊 Class Balance:")
    print(df['outcome'].value_counts(normalize=True))

if __name__ == "__main__":
    # You can override the path via command line or here
    # For now, let's assume user copies files to a local 'data' folder if direct access fails
    ingest_data("data/raw_signals") 

