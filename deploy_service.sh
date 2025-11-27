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

rsync -avz --progress --delete \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude '.git' \
    --exclude '.idea' \
    --exclude '.vscode' \
    --exclude 'terminals' \
    --exclude '文档' \
    --exclude 'trading-bridge/mql5' \
    ./ $SERVER:$REMOTE_DIR/

# Special handling for MQL5 files
if [ "$SERVICE" == "mt5" ] || [ "$SERVICE" == "all" ]; then
    echo "📂 Syncing MQL5 files..."
    rsync -avz --progress \
        trading-bridge/mql5/ $SERVER:$REMOTE_DIR/trading-bridge/mql5/
    
    # Fix permissions for MQL5 files
    ssh $SERVER "chmod -R 777 $REMOTE_DIR/trading-bridge/mql5"
fi

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
        echo '🚀 Using accelerated build with Chinese mirrors...' && \
        # 清理旧容器和镜像
        docker-compose down $SERVICE 2>/dev/null || true && \
        docker system prune -f && \
        # 使用国内镜像加速构建
        DOCKER_BUILDKIT=1 docker-compose build \
            --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
            --build-arg PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
            --build-arg APT_MIRROR=aliyun \
            $SERVICE && \
        docker-compose up -d $SERVICE && \
        echo '📋 Checking service status...' && \
        sleep 5 && \
        docker-compose logs --tail=20 $SERVICE
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
        # 强制清理旧的构建缓存，防止硬盘占满
        docker system prune -f && \
        # 限制构建并发数，防止内存爆炸 (适用于 Next.js)
        # 注意：需要在 docker-compose.yml 或 Dockerfile 中配合使用，或者依赖 Swap
        docker-compose down \$TARGET_SERVICE 2>/dev/null || true && \
        DOCKER_BUILDKIT=1 docker-compose build \$TARGET_SERVICE && \
        docker-compose up -d \$TARGET_SERVICE && \
        sleep 3 && \
        docker-compose logs --tail=15 \$TARGET_SERVICE
    fi"

echo "✅ Deployment of $SERVICE complete!"