#!/bin/bash

set -e

# Server configuration
REMOTE_HOST="macOS"
REMOTE_DIR="~/alpha-os"

# Command prefix to ensure PATH is correct for non-interactive SSH
REMOTE_CMD_PREFIX="source ~/.zprofile 2>/dev/null || true; source ~/.zshrc 2>/dev/null || true; export PATH=\$PATH:/usr/local/bin:/opt/homebrew/bin"

echo "🚀 Deploying AlphaOS to Remote Host ($REMOTE_HOST)..."

# ==========================================
# 0. Check Connection
# ==========================================
echo "🔍 Checking connection to $REMOTE_HOST..."
if ! ssh -q $REMOTE_HOST exit; then
    echo "❌ Cannot connect to $REMOTE_HOST. Please check your SSH config."
    exit 1
fi

# Check for Docker
if ! ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; command -v docker >/dev/null 2>&1"; then
    echo "❌ Docker not found on $REMOTE_HOST. Please ensure OrbStack or Docker is running."
    exit 1
fi

# ==========================================
# 1. Parse Arguments & Sync Files
# ==========================================
ARGS="$@"
SYNC_ALL=true
SYNC_WEB=false
SYNC_BRIDGE=false
SYNC_MT5=false
SYNC_AI=false

if [ -n "$ARGS" ]; then
    SYNC_ALL=false
    for arg in $ARGS; do
        case $arg in
            "--web") SYNC_WEB=true ;;
            "--bridge") SYNC_BRIDGE=true ;;
            "--mt5") SYNC_MT5=true ;;
            "--ai") SYNC_AI=true ;;
            "--migrate") MIGRATE_DATA=true ;;
        esac
    done
fi

echo "📂 Syncing files to Remote..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Common exclusions
EXCLUDE_OPTS="--exclude 'node_modules' --exclude '.next' --exclude '.git' --exclude 'terminals' --exclude '文档' --exclude 'trading-bridge/mql5'"

# Function to sync specific paths
sync_path() {
    local PATHS="$1"
    echo "   👉 Syncing: $PATHS"
    rsync -avz --progress $EXCLUDE_OPTS --relative $PATHS $REMOTE_HOST:$REMOTE_DIR/
}

if [ "$SYNC_ALL" = true ]; then
    echo "   📦 Syncing ALL files..."
    rsync -avz --progress --delete $EXCLUDE_OPTS ./ $REMOTE_HOST:$REMOTE_DIR/
else
    # Always sync common files
    echo "   📦 Syncing Common files..."
    if [ -f .env.local ]; then scp .env.local $REMOTE_HOST:$REMOTE_DIR/.env; fi
    if [ -f .env ] && [ ! -f .env.local ]; then scp .env $REMOTE_HOST:$REMOTE_DIR/.env; fi
    # Always sync .dockerignore to ensure correct build context
    if [ -f .dockerignore ]; then scp .dockerignore $REMOTE_HOST:$REMOTE_DIR/.dockerignore; fi
    
    if [ "$SYNC_WEB" = true ]; then
        echo "   📦 Syncing Web files..."
        sync_path "src public package.json package-lock.json .dockerignore next.config.ts postcss.config.mjs tailwind.config.ts tsconfig.json Dockerfile"
    fi
    
    # DEBUG: Check remote file structure
    echo "🔍 Debug: Checking remote src directory..."
    ssh $REMOTE_HOST "ls -F $REMOTE_DIR/src/ || echo '❌ src directory missing'"
    
    if [ "$SYNC_BRIDGE" = true ]; then
        echo "   📦 Syncing Bridge files..."
        sync_path "trading-bridge/ src/proto/"
    fi
    
    if [ "$SYNC_AI" = true ]; then
        echo "   📦 Syncing AI files..."
        sync_path "ai-engine/"
    fi
    
    if [ "$SYNC_MT5" = true ]; then
        echo "   📦 Syncing MT5 files..."
        if [ -f mt5setup.exe ]; then scp mt5setup.exe $REMOTE_HOST:$REMOTE_DIR/; fi
    fi
fi

# Sync MQL5 (Always sync if MT5 or Bridge is involved, or just always? Let's sync if MT5 or Bridge or All)
if [ "$SYNC_ALL" = true ] || [ "$SYNC_MT5" = true ] || [ "$SYNC_BRIDGE" = true ]; then
    echo "📂 Syncing MQL5 files..."
    rsync -avz --progress trading-bridge/mql5/ $REMOTE_HOST:$REMOTE_DIR/trading-bridge/mql5/
    ssh $REMOTE_HOST "chmod -R 777 $REMOTE_DIR/trading-bridge/mql5"
fi

# ==========================================
# 2. Remote Deployment
# ==========================================
echo "🔧 Configuring Remote Environment..."

# Get IP (using hostname -I or similar, but for Mac/OrbStack localhost might be fine for ports, 
# but for inter-container comms we use the docker network. 
# For external access, we'll print the remote host's IP).
REMOTE_IP=$(ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; ipconfig getifaddr en0 || ipconfig getifaddr en1 || ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print \$2}' | head -n 1")
if [ -z "$REMOTE_IP" ]; then REMOTE_IP="localhost"; fi

echo "🌐 Remote IP: $REMOTE_IP"

# Check for installer locally to determine mount string
INSTALLER_VOL_STR=""
if [ -f mt5setup.exe ]; then
    INSTALLER_VOL_STR="      - ../../mt5setup.exe:/tmp/mt5setup.exe"
fi

# Load local .env to export variables for script generation
if [ -f .env.local ]; then
    export $(cat .env.local | grep -v '^#' | grep -v '^$' | xargs)
elif [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

# Prepare Docker Compose files on Remote
echo "📦 Generating deployment configurations..."

cat > .deploy_script.sh <<EOF
set -e  # Exit immediately if a command exits with a non-zero status

cd $REMOTE_DIR

# Pass arguments to the script
ARGS="\$@"

# Load env
if [ -f .env ]; then export \$(cat .env | grep -v '^#' | xargs); fi

# Pre-build setup: Copy proto files to trading-bridge so they are in build context
mkdir -p trading-bridge/src/proto
cp src/proto/*.proto trading-bridge/src/proto/ 2>/dev/null || true

# ---------------------------------------------------------
# 🔧 Fix for China Network: Use Docker Mirror
# ---------------------------------------------------------
MIRROR_PREFIX="docker.m.daocloud.io"
echo "🔄 Replacing Docker base images with mirror: \$MIRROR_PREFIX..."

# Modify Dockerfile (Web)
sed -i.bak "s|FROM node:|FROM \${MIRROR_PREFIX}/library/node:|g" Dockerfile
sed -i.bak "s|RUN npm ci|RUN npm config set registry https://registry.npmmirror.com \\&\\& npm ci|g" Dockerfile

# Modify Dockerfile.api (Bridge)
sed -i.bak "s|FROM python:|FROM \${MIRROR_PREFIX}/library/python:|g" trading-bridge/docker/Dockerfile.api

# Modify Dockerfile (AI Engine)
sed -i.bak "s|FROM python:|FROM \${MIRROR_PREFIX}/library/python:|g" ai-engine/Dockerfile

# ---------------------------------------------------------

# Docker Network
docker network create --subnet=192.168.97.0/24 alphaos-net 2>/dev/null || true
# Create shared signal volume
docker volume create alphaos_signals 2>/dev/null || true
# Create AI models persistent volume
docker volume create ai_models 2>/dev/null || true

# Only delete deploy dirs for targeted services or if all
if [ -z "\$1" ]; then
    rm -rf deploy/bridge deploy/mt5 deploy/web deploy/ai
else
    for arg in \$ARGS; do
        if [ "\$arg" == "--bridge" ]; then rm -rf deploy/bridge; fi
        if [ "\$arg" == "--mt5" ]; then rm -rf deploy/mt5; fi
        if [ "\$arg" == "--web" ]; then rm -rf deploy/web; fi
        if [ "\$arg" == "--ai" ]; then rm -rf deploy/ai; fi
    done
fi

mkdir -p deploy/bridge deploy/mt5 deploy/web deploy/ai

# 1. Bridge API
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
        APT_MIRROR: aliyun
    container_name: bridge-api
    ports:
      - "8000:8000"
      # Expose gRPC port if needed for external access, but internal comms use docker network
      - "50050:50051" 
    environment:
      - ZMQ_BIND_ADDRESS=tcp://0.0.0.0:5555
      - ZMQ_SUB_ADDRESS=tcp://0.0.0.0:5556
      - SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL}
      - SUPABASE_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY}
      - SIGNAL_DIR=/app/signals
      # Point Bridge to AI Engine Container
      - AI_ENGINE_HOST=ai-engine
      - AI_ENGINE_PORT=50051
    volumes:
      - alphaos_signals:/app/signals
      - ../../src/proto:/app/src/proto
    networks:
      - alphaos-net
    restart: unless-stopped
    labels:
      icon: https://cdn-icons-png.flaticon.com/512/8099/8099466.png
      description: "AlphaOS Trading Bridge API"

networks:
  alphaos-net:
    external: true

volumes:
  alphaos_signals:
    external: true
INNEREOF

# 2. AI Engine
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
      - SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL}
      - SUPABASE_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY}
    volumes:
      - ai_models:/app/models
      - ../../ai-engine/ai_decisions.csv:/app/ai_decisions.csv
      - ../../ai-engine/src:/app/src
      - alphaos_signals:/app/signals
    networks:
      - alphaos-net
    restart: unless-stopped
    labels:
      icon: https://cdn-icons-png.flaticon.com/512/2103/2103633.png
      description: "AlphaOS AI Engine"

networks:
  alphaos-net:
    external: true
INNEREOF
    cat >> deploy/ai/docker-compose.yml <<INNEREOF

volumes:
  alphaos_signals:
    external: true
  ai_models:
    external: true
INNEREOF


# Check for installer on remote
INSTALLER_VOL=""
if [ -f mt5setup.exe ]; then
    # IMPORTANT: Correct indentation (6 spaces) for YAML injection
    INSTALLER_VOL="      - ../../mt5setup.exe:/tmp/mt5setup.exe"
fi


# Create persistent data directory on remote (IMPORTANT: Outside of deploy folder)
mkdir -p ~/alpha-os-data/mt5_config

# 3. MT5
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
      - PUID=0
      - PGID=0
    volumes:
      # Use persistent path outside of deploy folder
      - ~/alpha-os-data/mt5_config:/config
      # Mount MQL5 source to a temporary location
      - ../../trading-bridge/mql5:/mnt/alphaos_mql5
      - alphaos_signals:/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals
\${INSTALLER_VOL}
    networks:
      - alphaos-net
    restart: unless-stopped
    # Override entrypoint to copy installer and manage MQL5 files
    entrypoint: 
      - /bin/sh
      - -c
      - |
        echo "🛠️ Checking for pre-downloaded installer..."
        mkdir -p /config/.wine/drive_c
        
        # Check if MT5 is already installed
        if [ -d "/config/.wine/drive_c/Program Files/MetaTrader 5" ]; then
            echo "✅ MT5 already installed. Skipping installer copy."
        else
            if [ -f /tmp/mt5setup.exe ]; then
               echo "🚀 MT5 not found. Copying installer to Wine drive_c..."
               cp /tmp/mt5setup.exe /config/.wine/drive_c/mt5setup.exe
            fi
        fi
        
        # Background task to install MQL5 files once MT5 directory is ready
        (
          echo "⏳ Waiting for MT5 directory structure..."
          # Wait up to 5 minutes for MT5 to be installed
          for i in \\\$(seq 1 60); do
            if [ -d "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts" ]; then
              echo "📂 MT5 directory found, installing AlphaOS MQL5..."
              mkdir -p "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/AlphaOS"
              mkdir -p "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Indicators/AlphaOS"
              
              # Copy files (using cp to avoid symlink issues in Wine)
              cp -r /mnt/alphaos_mql5/* "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/AlphaOS/"
              cp -r /mnt/alphaos_mql5/* "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Indicators/AlphaOS/"
              
              echo "✅ AlphaOS MQL5 installed/updated."
              break
            fi
            sleep 5
          done
        ) &

        echo "🚀 Starting original entrypoint..."
        exec /init
    labels:
      icon: https://cdn-icons-png.flaticon.com/512/5968/5968260.png
      description: "MetaTrader 5 VNC"

networks:
  alphaos-net:
    external: true

volumes:
  alphaos_signals:
    external: true
INNEREOF

# Inject installer volume if available
# (Removed complex sed injection, handled via variable expansion above)
if [ -n "\$INSTALLER_VOL" ]; then
    : # No-op
fi

# 4. Web
cat > deploy/web/docker-compose.yml <<INNEREOF
version: '3.8'
services:
  web:
    build:
      context: ../..
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_SUPABASE_URL: ${NEXT_PUBLIC_SUPABASE_URL}
        NEXT_PUBLIC_SUPABASE_ANON_KEY: ${NEXT_PUBLIC_SUPABASE_ANON_KEY}
    container_name: alpha-os-web
    ports:
      - "3001:3000"
    environment:
      - TRADING_BRIDGE_API_URL=http://bridge-api:8000
      - NEXT_PUBLIC_SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL}
      - NEXT_PUBLIC_SUPABASE_ANON_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY}
    networks:
      - alphaos-net
    restart: unless-stopped
    labels:
      icon: https://cdn-icons-png.flaticon.com/512/10832/10832132.png
      description: "AlphaOS Web Dashboard"

networks:
  alphaos-net:
    external: true
INNEREOF

# Execute Deployments
TARGET_SERVICES=()
REBUILD_SERVICES=()
DEPLOY_SUPABASE=false

if [ -z "\$ARGS" ]; then
    TARGET_SERVICES=("ai" "bridge" "mt5" "web")
    REBUILD_SERVICES=("ai" "bridge" "web")
else
    for arg in \$ARGS; do
        case \$arg in
            "--ai")
                TARGET_SERVICES+=("ai")
                REBUILD_SERVICES+=("ai")
                ;;
            "--bridge")
                TARGET_SERVICES+=("bridge")
                REBUILD_SERVICES+=("bridge")
                ;;
            "--mt5")
                TARGET_SERVICES+=("mt5")
                ;;
            "--web")
                TARGET_SERVICES+=("web")
                REBUILD_SERVICES+=("web")
                ;;
            "--supabase")
                DEPLOY_SUPABASE=true
                ;;
        esac
    done
fi

# ---------------------------------------------------------
# Deploy Supabase (Independent Stack)
# ---------------------------------------------------------
if [ "\$DEPLOY_SUPABASE" = true ]; then
    echo "🚀 Deploying Supabase..."
    
    # 1. Prepare Directory
    mkdir -p ~/alpha-os-data/supabase
    
    # 2. Clone if not exists
    if [ ! -d ~/alpha-os-data/supabase/docker ]; then
        echo "📥 Cloning Supabase Docker repo..."
        # We clone into a temp dir and move docker folder or just clone the whole repo
        if [ ! -d ~/alpha-os-data/supabase/.git ]; then
             git clone --depth 1 https://github.com/supabase/supabase ~/alpha-os-data/supabase
        fi
    fi

    # 3. Configure & Start
    cd ~/alpha-os-data/supabase/docker
    
    # Ensure clean state for docker-compose.yml before modifying
    echo "🧹 Resetting Supabase config to clean state..."
    git checkout . 2>/dev/null || true
    
    # Apply Docker Mirror
    echo "🔄 Applying Docker Mirror (\${MIRROR_PREFIX}) to Supabase images..."
    # Replace 'image: ' with 'image: MIRROR_PREFIX/'
    # Note: simple replacement handles most 'supabase/xxx' and 'postgres' cases
    sed -i.bak "s|image: |image: \${MIRROR_PREFIX}/|g" docker-compose.yml

    # Set default env if missing
    if [ ! -f .env ]; then
        echo "⚙️ Configuring default Supabase environment..."
        cp .env.example .env
    fi

    echo "⚙️ Tuning Ports to avoid conflicts..."
    # Force Ports: API=54321 (was 8000), DB=54322 (was 5432), Studio=54323 (was 3000)
    # Using sed to replace or append if not exists would be safer, but simple replace works for default .env
    
    # 1. API Port (Kong)
    if grep -q "API_PORT=" .env; then
        sed -i.bak 's/API_PORT=.*/API_PORT=54321/' .env
    else
        echo "API_PORT=54321" >> .env
    fi

    # 2. DB Port
    if grep -q "POSTGRES_PORT=" .env; then
        sed -i.bak 's/POSTGRES_PORT=.*/POSTGRES_PORT=54322/' .env
    else
        echo "POSTGRES_PORT=54322" >> .env
    fi

    # 3. Studio Port
    if grep -q "STUDIO_PORT=" .env; then
        sed -i.bak 's/STUDIO_PORT=.*/STUDIO_PORT=54323/' .env
    else
        echo "STUDIO_PORT=54323" >> .env
    fi

    # Fix: Some versions use KONG_HTTP_PORT or similar, but API_PORT is standard in recent versions.
    # Also ensure port 8000 is not left in KONG_EXTERNAL_PORT if it exists
    sed -i.bak 's/=8000/=54321/g' .env
    
    echo "🔌 Injecting Network Unification Override..."
    # Create override to force Supabase services into alphaos-net
    cat > docker-compose.override.yml <<OVERRIDE
version: '3.8'
networks:
  alphaos-net:
    external: true

services:
  studio:
    ports:
      - "54323:3000"
    networks:
      - default
      - alphaos-net
  kong:
    networks:
      - default
      - alphaos-net
  auth:
    networks:
      - default
      - alphaos-net
  rest:
    networks:
      - default
      - alphaos-net
  realtime:
    networks:
      - default
      - alphaos-net
  storage:
    networks:
      - default
      - alphaos-net
  imgproxy:
    networks:
      - default
      - alphaos-net
  meta:
    networks:
      - default
      - alphaos-net
  db:
    networks:
      - default
      - alphaos-net
  analytics:
    networks:
      - default
      - alphaos-net
  functions:
    networks:
      - default
      - alphaos-net
OVERRIDE

    echo "🐳 Starting Supabase Services (this may take a while first time)..."
    docker compose up -d
    
    # Return to project dir
    cd \$REMOTE_DIR
fi

echo "🎯 Targeting services: \${TARGET_SERVICES[*]}"

for SERVICE in "\${TARGET_SERVICES[@]}"; do
    echo "🚀 Deploying \$SERVICE..."
    if [[ " \${REBUILD_SERVICES[@]} " =~ " \${SERVICE} " ]]; then
        if [ "\$SERVICE" == "bridge" ] || [ "\$SERVICE" == "ai" ]; then
             (cd deploy/\$SERVICE && docker compose up -d --build)
        else
             (cd deploy/\$SERVICE && docker compose up -d --build)
        fi
    else
        (cd deploy/\$SERVICE && docker compose up -d)
    fi
done
EOF

# Write script to file locally, then copy, then run.
# echo "$DEPLOY_SCRIPT" > .deploy_script.sh (Removed, now using cat > .deploy_script.sh <<EOF above)
chmod +x .deploy_script.sh

# Copy script to Remote
scp .deploy_script.sh $REMOTE_HOST:$REMOTE_DIR/

# Run script with arguments
ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; bash $REMOTE_DIR/.deploy_script.sh $@"

# Clean up
rm .deploy_script.sh
ssh $REMOTE_HOST "rm $REMOTE_DIR/.deploy_script.sh"

echo "✅ Deployment Complete!"
echo "Web: http://$REMOTE_IP:3001"
echo "MT5 VNC: http://$REMOTE_IP:3000"
echo "Bridge API: http://$REMOTE_IP:8000"
if [[ "$ARGS" == *"--supabase"* ]]; then
    echo "Supabase Studio: http://$REMOTE_IP:54323"
    echo "Supabase API: http://$REMOTE_IP:54321"
    echo "Supabase DB: postgres://postgres:postgres@$REMOTE_IP:54322/postgres"
fi

# Optional: Debug Info
echo ""
echo "🔍 Checking Service Status..."
ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; docker ps"
echo ""
if [[ "$ARGS" != *"--supabase"* ]]; then
    # Only show bridge logs if not just deploying supabase, or show generic logs
    ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; cd $REMOTE_DIR/deploy/bridge && docker compose logs --tail=20 bridge-api"
fi

# ==========================================
# Data Migration
# ==========================================
if [ "$MIGRATE_DATA" = true ]; then
    echo ""
    echo "📦 Starting Cloud Data Migration..."
    
    # 1. Sync Migration Script
    echo "   👉 Syncing migration script..."
    scp scripts/migrate_from_cloud.py $REMOTE_HOST:$REMOTE_DIR/
    
    # 2. Extract Service Role Key from Remote Supabase .env
    # The Supabase stack on remote uses ~/alpha-os-data/supabase/docker/.env
    echo "   🔑 Fetching Service Key from Remote Supabase Config..."
    SERVICE_KEY=$(ssh $REMOTE_HOST "grep 'SERVICE_ROLE_KEY=' ~/alpha-os-data/supabase/docker/.env 2>/dev/null | cut -d '=' -f2")
    
    # Clean up key (remove quotes and carriage returns)
    SERVICE_KEY=$(echo $SERVICE_KEY | tr -d '"' | tr -d "'" | tr -d '\r')
    
    if [ -n "$SERVICE_KEY" ]; then
         echo "      Found Key: ${SERVICE_KEY:0:10}..."
    else
         echo "   ⚠️  Could not find SERVICE_ROLE_KEY in remote ~/alpha-os-data/supabase/docker/.env"
         echo "   ⚠️  Trying standard default key..."
         # Fallback or fail. Let's try to proceed, maybe the script defaults or user passed it.
    fi

    echo "   🏃 Running migration on Remote Host (using local Python)..."
    # Ensure requests is installed
    ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; pip3 install requests >/dev/null 2>&1 || true"
    
    ssh $REMOTE_HOST "$REMOTE_CMD_PREFIX; python3 $REMOTE_DIR/migrate_from_cloud.py $SERVICE_KEY"
    
    echo "✅ Migration Step Complete."
fi
