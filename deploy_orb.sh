#!/bin/bash

# ==============================================================================
# AlphaOS Deployment Script (OrbStack/macOS Remote)
# Adheres to Antigravity Global Agent Rules v2.1
# ==============================================================================

set -e

# --- Configuration [Rule 2: Remote Infrastructure] ---
REMOTE_HOST="macOS"
REMOTE_DIR="~/alpha-os"
REMOTE_CMD_PREFIX="source ~/.zprofile 2>/dev/null || true; source ~/.zshrc 2>/dev/null || true; export PATH=\$PATH:/usr/local/bin:/opt/homebrew/bin"

# Docker Mirror [Rule 7: China Network Optimization]
MIRROR_PREFIX="docker.m.daocloud.io"

# Logging Helpers [Rule 7: No Silent Failures]
log_info() { echo -e "🚀 \033[1;34m[INFO]\033[0m $1"; }
log_warn() { echo -e "⚠️  \033[1;33m[WARN]\033[0m $1"; }
log_error() { echo -e "❌ \033[1;31m[ERROR]\033[0m $1"; }

# ==============================================================================
# 0. Pre-Flight Checks & Protocol Consistency [Rule 0: Anti-Entropy]
# ==============================================================================
log_info "Protocol Consistency Check..."
if [ -f "./scripts/sync_proto.sh" ]; then
    log_info "Running sync_proto.sh to ensure local assets are pristine..."
    chmod +x ./scripts/sync_proto.sh
    ./scripts/sync_proto.sh || { log_error "Protocol Sync Failed"; exit 1; }
else
    log_warn "scripts/sync_proto.sh not found. Skipping local protocol generation."
fi

# SSH Connection Check
log_info "Checking connection to $REMOTE_HOST..."
if ! ssh -q $REMOTE_HOST exit; then
    log_error "Cannot connect to $REMOTE_HOST. Check SSH config."
    exit 1
fi

# ==============================================================================
# 1. File Synchronization
# ==============================================================================
ARGS="$@"
SYNC_WEB=false
SYNC_BRIDGE=false
SYNC_MT5=false
SYNC_AI=false
DEPLOY_SUPABASE=false
MIGRATE_DATA=false

# Simple argument parsing
if [[ "$ARGS" == *"--web"* ]]; then SYNC_WEB=true; fi
if [[ "$ARGS" == *"--bridge"* ]]; then SYNC_BRIDGE=true; fi
if [[ "$ARGS" == *"--mt5"* ]]; then SYNC_MT5=true; fi
if [[ "$ARGS" == *"--ai"* ]]; then SYNC_AI=true; fi
if [[ "$ARGS" == *"--supabase"* ]]; then DEPLOY_SUPABASE=true; fi
if [[ "$ARGS" == *"--migrate"* ]]; then MIGRATE_DATA=true; fi

# Default to ALL if no specific service selected (ignoring --supabase/--migrate flag presence if no others)
if [ "$SYNC_WEB" = false ] && [ "$SYNC_BRIDGE" = false ] && [ "$SYNC_MT5" = false ] && [ "$SYNC_AI" = false ]; then
    if [ "$DEPLOY_SUPABASE" = false ] && [ "$MIGRATE_DATA" = false ]; then
        SYNC_WEB=true; SYNC_BRIDGE=true; SYNC_MT5=true; SYNC_AI=true; 
        log_info "No specific service flags. Syncing ALL."
    fi
fi

log_info "Syncing files to Remote..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

EXCLUDE_OPTS="--exclude 'node_modules' --exclude '.next' --exclude '.git' --exclude 'terminals' --exclude 'trading-bridge/mql5'"

sync_files() {
    log_info "Syncing Common Configs..."
    # Always sync env files and critical configs
    for file in .env .env.local .dockerignore; do
        if [ -f "$file" ]; then scp "$file" "$REMOTE_HOST:$REMOTE_DIR/"; fi
    done

    if [ "$SYNC_WEB" = true ]; then
        log_info "Syncing Web Assets..."
        rsync -avz $EXCLUDE_OPTS --delete \
            src public package.json package-lock.json next.config.ts postcss.config.mjs tailwind.config.ts tsconfig.json Dockerfile \
            $REMOTE_HOST:$REMOTE_DIR/
        
        # Verify src
        ssh $REMOTE_HOST "ls -F $REMOTE_DIR/src/ >/dev/null 2>&1 || echo '❌ Remote src dir missing'"
    fi

    if [ "$SYNC_BRIDGE" = true ]; then
        log_info "Syncing Trading Bridge..."
        # Rule 0: Sync whole directory to capture new core/ managers/ etc.
        rsync -avz $EXCLUDE_OPTS \
            trading-bridge src/proto \
            $REMOTE_HOST:$REMOTE_DIR/
    fi

    if [ "$SYNC_AI" = true ]; then
        log_info "Syncing AI Engine..."
        # Use --delete to remove old models/artifacts on remote, ensuring 1:1 mirror
        # Exclude ai_decisions.csv to prevent deleting production logs generated on remote
        rsync -avz $EXCLUDE_OPTS --delete --exclude 'ai_decisions.csv' \
            ai-engine \
            $REMOTE_HOST:$REMOTE_DIR/
    fi

    if [ "$SYNC_MT5" = true ]; then
        log_info "Syncing MT5 Installer..."
        if [ -f mt5setup.exe ]; then scp mt5setup.exe $REMOTE_HOST:$REMOTE_DIR/; fi
    fi

    # MQL5: Sync if Bridge or MT5 is involved, as Logic resides in MQL5 too
    if [ "$SYNC_BRIDGE" = true ] || [ "$SYNC_MT5" = true ]; then
        log_info "Syncing MQL5..."
        rsync -avz trading-bridge/mql5/ $REMOTE_HOST:$REMOTE_DIR/trading-bridge/mql5/
        ssh $REMOTE_HOST "chmod -R 777 $REMOTE_DIR/trading-bridge/mql5"
    fi
}

sync_files

# ==============================================================================
# 2. Remote Deployment Generation
# ==============================================================================
log_info "Generating Remote Deployment Configurations..."

# Load local env for interpolation
if [ -f .env.local ]; then export $(grep -v '^#' .env.local | xargs); fi
if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi

# Generate the remote execution script
cat > .deploy_script.sh <<EOF
set -e
cd $REMOTE_DIR
if [ -f .env ]; then export \$(grep -v '^#' .env | xargs); fi

# [Rule 7] Mirror Config
MIRROR_PREFIX="$MIRROR_PREFIX"

# Protocol Artifacts Preparation
mkdir -p trading-bridge/src/proto
cp src/proto/*.proto trading-bridge/src/proto/ 2>/dev/null || true

# ------------------------------------------------------------------------------
# Docker Infrastructure Setup
# ------------------------------------------------------------------------------
echo "🛠️  Setting up Docker Infrastructure..."

# Apply Mirrors to Dockerfiles
sed -i.bak "s|FROM node:|FROM \${MIRROR_PREFIX}/library/node:|g" Dockerfile
sed -i.bak "s|RUN npm ci|RUN npm config set registry https://registry.npmmirror.com \\&\\& npm ci|g" Dockerfile

# Fix for Bridge Dockerfile (check if exists)
if [ -f trading-bridge/docker/Dockerfile.api ]; then
    sed -i.bak "s|FROM python:|FROM \${MIRROR_PREFIX}/library/python:|g" trading-bridge/docker/Dockerfile.api
fi

if [ -f ai-engine/Dockerfile ]; then
    sed -i.bak "s|FROM python:|FROM \${MIRROR_PREFIX}/library/python:|g" ai-engine/Dockerfile
fi

# Networks & Volumes
docker network create --subnet=192.168.97.0/24 alphaos-net 2>/dev/null || true
docker volume create alphaos_signals 2>/dev/null || true
# AI 模型改为宿主机目录 bind mount（避免 OrbStack 容器内部 NFS 路径导致的超时/不可读）
# 因此不再依赖 ai_models volume

# Cleanup Deploy Dirs
rm -rf deploy/bridge deploy/mt5 deploy/web deploy/ai
mkdir -p deploy/bridge deploy/mt5 deploy/web deploy/ai

# Ensure AI decision log target is a file on remote bind mount path.
# If it is accidentally a directory (common after some rsync/docker mount combos), keep its contents and recreate as file.
if [ -d "ai-engine/ai_decisions.csv" ]; then
  echo "⚠️  ai-engine/ai_decisions.csv is a directory. Backing up and recreating as a file..."
  mv "ai-engine/ai_decisions.csv" "ai-engine/ai_decisions.csv.bak_\$(date +%Y%m%d_%H%M%S)" || true
  : > "ai-engine/ai_decisions.csv"
elif [ ! -e "ai-engine/ai_decisions.csv" ]; then
  : > "ai-engine/ai_decisions.csv"
fi

# ------------------------------------------------------------------------------
# Service Definitions
# ------------------------------------------------------------------------------

# --- 1. Trading Bridge (Refactored) ---
cat > deploy/bridge/docker-compose.yml <<INNEREOF
version: '3.8'
services:
  bridge-api:
    build:
      context: ../../trading-bridge
      dockerfile: docker/Dockerfile.api
      args:
        PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple
        PIP_TRUSTED_HOST: pypi.tuna.tsinghua.edu.cn
    container_name: bridge-api
    ports:
      - "8000:8000"   # HTTP
      - "50050:50051" # gRPC External
    environment:
      - SUPABASE_URL=\${NEXT_PUBLIC_SUPABASE_URL}
      - SUPABASE_KEY=\${NEXT_PUBLIC_SUPABASE_ANON_KEY}
      - SIGNAL_DIR=/app/signals
      - AI_ENGINE_HOST=ai-engine
      - AI_ENGINE_PORT=50051
    volumes:
      - alphaos_signals:/app/signals
      - ../../src/proto:/app/src/proto
    networks:
      - alphaos-net
    restart: unless-stopped
    labels:
      description: "AlphaOS Trading Bridge (Thin Entry)"

networks:
  alphaos-net:
    external: true
volumes:
  alphaos_signals:
    external: true
INNEREOF

# --- 2. AI Engine ---
cat > deploy/ai/docker-compose.yml <<INNEREOF
version: '3.8'
services:
  ai-engine:
    build:
      context: ../..
      dockerfile: ai-engine/Dockerfile
      args:
        PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple
        PIP_TRUSTED_HOST: pypi.tuna.tsinghua.edu.cn
    container_name: ai-engine
    ports:
      - "50051:50051"
    environment:
      - PYTHONUNBUFFERED=1
      - CLOUD_BRIDGE_URL=bridge-api:50051
      - SIGNAL_DIR=/app/signals
      - SUPABASE_URL=\${NEXT_PUBLIC_SUPABASE_URL}
      - SUPABASE_KEY=\${NEXT_PUBLIC_SUPABASE_ANON_KEY}
    volumes:
      # Models: bind mount to host path so files are always accessible (no OrbStack NFS paths)
      - ../../ai-engine/models:/app/models:ro
      # [Rule 4] Log persistence
      - ../../ai-engine/ai_decisions.csv:/app/ai_decisions.csv
      - ../../ai-engine/src:/app/src
      - alphaos_signals:/app/signals
    networks:
      - alphaos-net
    restart: unless-stopped

networks:
  alphaos-net:
    external: true
volumes:
  alphaos_signals:
    external: true
INNEREOF

# --- 3. MT5 (VNC) ---
# Prepare Persistent Config
mkdir -p ~/alpha-os-data/mt5_config

cat > deploy/mt5/docker-compose.yml <<INNEREOF
version: '3.8'
services:
  mt5:
    image: gmag11/metatrader5_vnc:latest
    platform: linux/amd64
    container_name: mt5-vnc
    ports:
      - "3000:3000"
      - "8001:8001"
    environment:
      - CUSTOM_USER=root
      - PASSWORD=trading
    volumes:
      - ~/alpha-os-data/mt5_config:/config
      - ../../trading-bridge/mql5:/mnt/alphaos_mql5
      - alphaos_signals:/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals
INNEREOF

# Add Installer Volume if matches
if [ -f mt5setup.exe ]; then
    echo "      - ../../mt5setup.exe:/tmp/mt5setup.exe" >> deploy/mt5/docker-compose.yml
fi

# Finish MT5 Compose
cat >> deploy/mt5/docker-compose.yml <<INNEREOF
    networks:
      - alphaos-net
    restart: unless-stopped
    entrypoint: 
      - /bin/sh
      - -c
      - |
        echo "🛠️ Init AlphaOS MT5..."
        mkdir -p /config/.wine/drive_c
        if [ ! -d "/config/.wine/drive_c/Program Files/MetaTrader 5" ] && [ -f /tmp/mt5setup.exe ]; then
             cp /tmp/mt5setup.exe /config/.wine/drive_c/mt5setup.exe
        fi
        
        # MQL5 Sync Background Task
        (
          for i in \\\$(seq 1 60); do
            if [ -d "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts" ]; then
              echo "📂 Installing MQL5..."
              mkdir -p "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/AlphaOS"
              mkdir -p "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Indicators/AlphaOS"
              cp -r /mnt/alphaos_mql5/* "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/AlphaOS/"
              cp -r /mnt/alphaos_mql5/* "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Indicators/AlphaOS/"
              echo "✅ MQL5 Updated."
              break
            fi
            sleep 5
          done
        ) &
        exec /init

networks:
  alphaos-net:
    external: true
volumes:
  alphaos_signals:
    external: true
INNEREOF

# --- 4. Web Dashboard ---
cat > deploy/web/docker-compose.yml <<INNEREOF
version: '3.8'
services:
  web:
    build:
      context: ../..
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_SUPABASE_URL: \${NEXT_PUBLIC_SUPABASE_URL}
        NEXT_PUBLIC_SUPABASE_ANON_KEY: \${NEXT_PUBLIC_SUPABASE_ANON_KEY}
    container_name: alpha-os-web
    ports:
      - "3001:3000"
    environment:
      - TRADING_BRIDGE_API_URL=http://bridge-api:8000
      - NEXT_PUBLIC_SUPABASE_URL=\${NEXT_PUBLIC_SUPABASE_URL}
      - NEXT_PUBLIC_SUPABASE_ANON_KEY=\${NEXT_PUBLIC_SUPABASE_ANON_KEY}
    networks:
      - alphaos-net
    restart: unless-stopped

networks:
  alphaos-net:
    external: true
INNEREOF

# ------------------------------------------------------------------------------
# Supabase Deployment (Optional)
# ------------------------------------------------------------------------------
if [ "$DEPLOY_SUPABASE" = "true" ]; then
    echo "🚀 Deploying Supabase Stack..."
    mkdir -p ~/alpha-os-data/supabase
    if [ ! -d ~/alpha-os-data/supabase/.git ]; then
         git clone --depth 1 https://github.com/supabase/supabase ~/alpha-os-data/supabase
    fi
    
    cd ~/alpha-os-data/supabase/docker
    git checkout . 2>/dev/null || true
    
    sed -i.bak "s|image: |image: \${MIRROR_PREFIX}/|g" docker-compose.yml
    
    if [ ! -f .env ]; then cp .env.example .env; fi
    
    # Port Configuration (User Rule: Avoid Conflicts)
    sed -i.bak 's/API_PORT=.*/API_PORT=54321/' .env
    sed -i.bak 's/POSTGRES_PORT=.*/POSTGRES_PORT=54322/' .env
    sed -i.bak 's/STUDIO_PORT=.*/STUDIO_PORT=54323/' .env
    # Ensure standard port variable compliance
    if ! grep -q "API_PORT=" .env; then echo "API_PORT=54321" >> .env; fi
    
    # Network Override
    cat > docker-compose.override.yml <<OVERRIDE
version: '3.8'
networks:
  alphaos-net:
    external: true
services:
  studio:
    networks: [default, alphaos-net]
  kong:
    networks: [default, alphaos-net]
  auth:
    networks: [default, alphaos-net]
  rest:
    networks: [default, alphaos-net]
  realtime:
    networks: [default, alphaos-net]
  storage:
    networks: [default, alphaos-net]
  imgproxy:
    networks: [default, alphaos-net]
  meta:
    networks: [default, alphaos-net]
  db:
    networks: [default, alphaos-net]
  analytics:
    networks: [default, alphaos-net]
  functions:
    networks: [default, alphaos-net]
OVERRIDE

    docker compose up -d
    cd $REMOTE_DIR
fi

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------
if [ "$SYNC_BRIDGE" = "true" ]; then (cd deploy/bridge && docker compose up -d --build); fi
if [ "$SYNC_AI" = "true" ]; then (cd deploy/ai && docker compose up -d --build); fi
if [ "$SYNC_MT5" = "true" ]; then (cd deploy/mt5 && docker compose up -d); fi
if [ "$SYNC_WEB" = "true" ]; then (cd deploy/web && docker compose up -d --build); fi

EOF

# ==============================================================================
# 3. Execute Remote Script
# ==============================================================================
log_info "Executing Remote Deployment..."

chmod +x .deploy_script.sh
scp .deploy_script.sh $REMOTE_HOST:$REMOTE_DIR/
ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; bash $REMOTE_DIR/.deploy_script.sh"

log_info "Cleanup..."
rm .deploy_script.sh
ssh $REMOTE_HOST "rm $REMOTE_DIR/.deploy_script.sh"

log_info "✅ Deployment Complete"
