#!/bin/bash

# Setup Local AI Engine

echo "🔧 Setting up Local AI Engine..."

# 1. Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 2. Install deps
echo "Installing dependencies..."
pip install -r requirements.txt

# 3. Generate Proto
echo "Generating gRPC code..."
# Copy proto from main repo if needed, or assume it's in src/proto (user must copy or symlink)
# For this script, we assume the user copies ../src/proto/alphaos.proto to proto/alphaos.proto
mkdir -p src/proto
cp ../src/proto/alphaos.proto src/proto/alphaos.proto

python -m grpc_tools.protoc \
    -Isrc/proto \
    --python_out=src \
    --grpc_python_out=src \
    src/proto/alphaos.proto

echo "✅ Setup Complete. Run 'source venv/bin/activate && python src/client.py' to start."

