import requests
import json
import os
import sys

# Configuration (Injected or Defaults)
# Cloud Credentials (Source)
CLOUD_URL = "https://wbhnblpjzrjwyqxullog.supabase.co"
CLOUD_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndiaG5ibHBqenJqd3lxeHVsbG9nIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzYyNTU1NCwiZXhwIjoyMDc5MjAxNTU0fQ.-R8kJSx7fXQ-1nmI1qMjmfKR0xxeo5Mi_hDxYVnfrb8"

# Local Credentials (Target) -> Assumes running on the same machine as the target Supabase
LOCAL_URL = "http://localhost:54321" 
# We need the LOCAL SERVICE KEY. 
# Attempt to read from env or argument. If typically standard Supabase Docker, it might be in .env
# For now, we will try to find it in the standard location or ask user to provide/hardcode if known default.
# Default Supabase Docker Service Key is often the one in .env.example
# Let's try to match the one from .env.local if it matches the 'service_role' pattern.

# List of tables to migrate
TABLES = [
    'training_datasets', 
    'user_preferences', 
    'accounts', 
    'automation_rules', 
    'journal_notes', 
    'training_signals', 
    'signals', 
    'trades'
]

def get_headers(key):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal" # Don't return all inserted rows to save bandwidth
    }

def migrate_table(table_name, target_key):
    print(f"\n📦 Migrating table: {table_name}...")
    
    # 1. Fetch from Cloud
    try:
        # Fetching all provided row limits. For larger datasets, pagination is needed.
        # Here we fetch up to 10000 rows.
        url = f"{CLOUD_URL}/rest/v1/{table_name}?select=*"
        resp = requests.get(url, headers=get_headers(CLOUD_KEY))
        if resp.status_code != 200:
            print(f"   ❌ Failed to fetch from Cloud: {resp.text}")
            return
        
        data = resp.json()
        count = len(data)
        print(f"   ⬇️  Fetched {count} rows from Cloud.")
        
        if count == 0:
            return

        # 2. Upsert to Local
        # Use UPSERT (merge-duplicates)
        headers = get_headers(target_key)
        headers["Prefer"] = "resolution=merge-duplicates"
        
        local_api_url = f"{LOCAL_URL}/rest/v1/{table_name}"
        
        # Batch insert if needed? Supabase handles bulk inserts usually well.
        # Let's try chunks of 1000
        chunk_size = 1000
        for i in range(0, count, chunk_size):
            chunk = data[i:i+chunk_size]
            r = requests.post(local_api_url, json=chunk, headers=headers)
            if r.status_code in [200, 201, 204]:
                 print(f"   ⬆️  Upserted rows {i} to {i+len(chunk)}")
            else:
                 print(f"   ❌ Failed to upsert batch {i}: {r.status_code} - {r.text}")

    except Exception as e:
        print(f"   ❌ Error migrating {table_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 migrate_from_cloud.py <LOCAL_SERVICE_ROLE_KEY>")
        # Try to read from env or file
        local_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    else:
        local_key = sys.argv[1]

    if not local_key:
        print("❌ Error: Local Service Role Key not provided.")
        sys.exit(1)
        
    print(f"🚀 Starting Migration from {CLOUD_URL} to {LOCAL_URL}")
    
    for table in TABLES:
        migrate_table(table, local_key)
        
    print("\n✅ Migration Finished.")
