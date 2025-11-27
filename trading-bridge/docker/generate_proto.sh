#!/bin/bash
# Script to generate gRPC code inside the container

echo "🛠️ Generating gRPC code..."

# Ensure target directory exists
mkdir -p /app/src

# Run protoc
# We assume the proto file is copied to /app/src/proto/alphaos.proto
python -m grpc_tools.protoc \
    -I/app/src/proto \
    --python_out=/app/src \
    --grpc_python_out=/app/src \
    /app/src/proto/alphaos.proto

if [ $? -eq 0 ]; then
    echo "✅ gRPC code generated successfully."
else
    echo "❌ Failed to generate gRPC code."
    exit 1
fi

