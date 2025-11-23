#!/bin/bash

# Server configuration
SERVER="alphaos"
REMOTE_DIR="~/alpha-os"

echo "🚀 Deploying AlphaOS Frontend to $SERVER..."

# 1. Sync files
echo "📂 Syncing files..."
# Ensure remote directory exists
echo "  - Creating remote directory: $REMOTE_DIR"
ssh $SERVER "mkdir -p $REMOTE_DIR" || { echo "❌ Failed to create remote directory. Check SSH connection."; exit 1; }

rsync -avz --progress \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude '.git' \
    --exclude '.idea' \
    --exclude '.vscode' \
    --exclude 'terminals' \
    --exclude '文档' \
    ./ $SERVER:$REMOTE_DIR/

# 2. Copy .env if it exists
if [ -f .env.local ]; then
    echo "🔑 Copying .env.local to remote as .env..."
    scp .env.local $SERVER:$REMOTE_DIR/.env
elif [ -f .env ]; then
    echo "🔑 Copying .env to remote..."
    scp .env $SERVER:$REMOTE_DIR/.env
else
    echo "⚠️ No .env or .env.local found! Deployment might fail."
fi

# 3. Build and Restart
echo "🏗️ Building and Restarting container..."
# We run docker-compose in the root directory now
ssh $SERVER "cd $REMOTE_DIR && \
    docker-compose down && \
    # Source .env file explicitly to ensure variables are loaded for build args
    export \$(cat .env | xargs) && \
    docker-compose up -d --build"

echo "✅ Deployment complete! Frontend should be available at http://49.235.153.73:3001"
