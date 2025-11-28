#!/bin/bash

# Local Colima configuration
LOCAL_DIR="$PWD"
PROJECT_NAME="alphaos"

# Check for arguments
if [ $# -eq 0 ]; then
    echo "❌ Please provide one or more service names."
    echo "Supported services: API, MT5, WEB"
    exit 1
fi

# Check if Colima is running
if ! colima status >/dev/null 2>&1; then
    echo "🚀 Starting Colima..."
    colima start
fi

# Check if Docker is available
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not available. Please check Colima status."
    exit 1
fi

# Normalize service names
TARGET_SERVICES=()
for RAW_SERVICE in "$@"; do
    SERVICE_LOWER=$(echo "$RAW_SERVICE" | tr '[:upper:]' '[:lower:]')
    
    case "$SERVICE_LOWER" in
        "api"|"bridge-api")
            TARGET_SERVICES+=("bridge-api")
            ;;
        "mt5")
            TARGET_SERVICES+=("mt5")
            ;;
        "web"|"frontend"|"alphaos")
            TARGET_SERVICES+=("web")
            ;;
        "all")
            TARGET_SERVICES=("bridge-api" "mt5" "web")
            break
            ;;
        *)
            TARGET_SERVICES+=("$RAW_SERVICE")
            ;;
    esac
done

echo "🚀 Deploying Services to Local Colima: ${TARGET_SERVICES[*]}..."

# 禁用 BuildKit 使用传统构建
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Function to deploy backend services
deploy_backend() {
    local SERVICE=$1
    echo "🏗️ Deploying backend service: $SERVICE"
    
    cd "$LOCAL_DIR/trading-bridge/docker" || { echo "❌ trading-bridge/docker directory not found"; return 1; }
    
    # Copy .env if exists
    if [ -f "$LOCAL_DIR/.env.local" ]; then
        cp "$LOCAL_DIR/.env.local" .env
    elif [ -f "$LOCAL_DIR/.env" ]; then
        cp "$LOCAL_DIR/.env" .env
    fi
    
    # Stop service
    echo "🛑 Stopping existing service..."
    docker-compose down $SERVICE 2>/dev/null || true
    
    # Cleanup
    echo "🧹 Cleaning up..."
    docker system prune -f
    
    # Build using traditional method (no BuildKit)
    echo "🚀 Building $SERVICE (traditional build)..."
    
    # 直接使用 docker build 而不是 docker-compose build
    case "$SERVICE" in
        "bridge-api")
            docker build -f Dockerfile.api -t ${PROJECT_NAME}-bridge-api .
            ;;
        "mt5")
            docker build -f Dockerfile.mt5 -t ${PROJECT_NAME}-mt5 . 2>/dev/null || \
            echo "⚠️ MT5 Dockerfile might not exist, using compose directly"
            ;;
    esac
    
    # Start service
    echo "🎯 Starting $SERVICE..."
    docker-compose up -d $SERVICE
    
    # Check status
    echo "📋 Checking service status..."
    sleep 8
    docker-compose logs --tail=10 $SERVICE
    
    cd "$LOCAL_DIR"
}

# Function to deploy frontend service
deploy_frontend() {
    local SERVICE=$1
    echo "🏗️ Deploying frontend service: $SERVICE"
    
    # Copy .env if exists
    if [ -f ".env.local" ]; then
        cp .env.local .env 2>/dev/null || true
    fi
    
    # Stop service
    echo "🛑 Stopping existing service..."
    docker-compose down $SERVICE 2>/dev/null || true
    
    # Cleanup
    echo "🧹 Cleaning up..."
    docker system prune -f
    
    # Build frontend using traditional method
    echo "🚀 Building frontend (traditional build)..."
    docker build -t ${PROJECT_NAME}-web .
    
    # Start service
    echo "🎯 Starting $SERVICE..."
    docker-compose up -d $SERVICE
    
    # Check status
    echo "📋 Checking service status..."
    sleep 5
    docker-compose logs --tail=10 $SERVICE
}

# Special handling for MQL5 files
if [[ " ${TARGET_SERVICES[@]} " =~ "mt5" ]]; then
    echo "🔧 Setting MQL5 files permissions..."
    chmod -R 777 "$LOCAL_DIR/trading-bridge/mql5" 2>/dev/null || true
fi

# Deploy each service
for SERVICE in "${TARGET_SERVICES[@]}"; do
    echo "========================================="
    echo "🚀 Processing: $SERVICE"
    echo "========================================="
    
    case "$SERVICE" in
        "bridge-api"|"mt5")
            deploy_backend "$SERVICE"
            ;;
        "web")
            deploy_frontend "$SERVICE"
            ;;
        *)
            echo "⚠️ Unknown service: $SERVICE, attempting generic deployment..."
            deploy_frontend "$SERVICE"
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "✅ $SERVICE deployed successfully!"
    else
        echo "❌ $SERVICE deployment failed!"
    fi
    echo ""
done

# Show final status
echo "========================================="
echo "📊 Deployment Summary"
echo "========================================="

docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "✅ Local deployment complete!"