import requests
import json
import os

URL = "https://wbhnblpjzrjwyqxullog.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndiaG5ibHBqenJqd3lxeHVsbG9nIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzYyNTU1NCwiZXhwIjoyMDc5MjAxNTU0fQ.-R8kJSx7fXQ-1nmI1qMjmfKR0xxeo5Mi_hDxYVnfrb8"

headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json"
}

# 1. Try to fetch OpenAPI spec (usually lists all tables)
try:
    print("Checking OpenAPI spec...")
    resp = requests.get(f"{URL}/rest/v1/?apikey={KEY}")
    if resp.status_code == 200:
        spec = resp.json()
        print("\n✅ Successfully connected to Cloud Supabase API!")
        print("\n📊 Available Tables (from Definitions):")
        # PostgREST root endpoint returns OpenAPI-like definitions
        if 'definitions' in spec:
            for table_name in spec['definitions']:
                print(f" - {table_name}")
        else:
             print("No table definitions found in root response.")
             print("Raw keys:", spec.keys())
    else:
        print(f"❌ Failed to fetch tables. Status: {resp.status_code}")
        print(resp.text)

except Exception as e:
    print(f"❌ Error: {e}")
