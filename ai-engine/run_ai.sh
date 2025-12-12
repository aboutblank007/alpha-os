#!/bin/bash

# AI Engine Startup Script

# 1. Navigate to directory
cd "$(dirname "$0")"

# 2. Activate Virtual Environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run ./setup_local.sh first."
    exit 1
fi

# 3. Set Configuration (Modify IP if needed)
# gRPC address format: IP:PORT (no http://)
export CLOUD_BRIDGE_URL="100.91.208.22:50051"

# 4. Run Client
echo "🚀 Starting AI Engine..."
echo "Target Bridge: $CLOUD_BRIDGE_URL"
python3 src/client.py
