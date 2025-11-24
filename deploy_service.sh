#!/bin/bash

# Server configuration
SERVER="alphaos"
REMOTE_DIR="~/alpha-os"

# Check for arguments
if [ $# -eq 0 ]; then
    echo "❌ Please provide a service name (e.g., 'bridge-api', 'frontend', 'web')"
    echo "Usage: ./deploy_service.sh <service_name>"
    exit 1
fi

SERVICE=$1

echo "🚀 Deploying Service: $SERVICE to $SERVER..."

# 1. Sync files (Sync everything as code changes might affect any service)
echo "📂 Syncing codebase..."
# Ensure remote directory exists
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

# 2. Copy .env if it exists (Always good to ensure env is fresh)
if [ -f .env.local ]; then
    echo "🔑 Copying .env.local to remote as .env..."
    scp .env.local $SERVER:$REMOTE_DIR/.env
elif [ -f .env ]; then
    echo "🔑 Copying .env to remote..."
    scp .env $SERVER:$REMOTE_DIR/.env
else
    echo "⚠️ No .env or .env.local found! Check if your service needs it."
fi

# 3. Rebuild and Restart Specific Service
echo "🏗️ Rebuilding and Restarting service: $SERVICE..."

ssh $SERVER "cd $REMOTE_DIR && \
    # Load env vars from root .env
    if [ -f .env ]; then export \$(cat .env | grep -v '^#' | xargs); fi && \
    
    if [ \"$SERVICE\" == \"bridge-api\" ] || [ \"$SERVICE\" == \"mt5\" ]; then
        # Backend services in trading-bridge/docker/docker-compose.yml
        echo 'Navigate to trading-bridge/docker for backend services...' && \
        if [ -f .env ]; then cp .env trading-bridge/docker/.env; fi && \
        cd trading-bridge/docker && \
        docker-compose up -d --build $SERVICE
    else
        # Frontend service in root docker-compose.yml
        # Service name is likely 'web' based on your docker-compose.yml
        echo 'Updating frontend service in root...' && \
        if [ \"$SERVICE\" == \"frontend\" ] || [ \"$SERVICE\" == \"alphaos\" ]; then
            TARGET_SERVICE=\"web\"
        else
            TARGET_SERVICE=\"$SERVICE\"
        fi && \
        echo \"Targeting docker-compose service: \$TARGET_SERVICE\" && \
        docker-compose up -d --build \$TARGET_SERVICE
    fi"

echo "✅ Deployment of $SERVICE complete!"
