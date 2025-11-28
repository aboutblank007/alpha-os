#!/bin/bash

# 🟢 本地 Bridge 启动脚本 (用于收集本地 MT5 回测数据)

# 1. 设置环境变量
# ⚠️ 请修改下面的路径为您本地 MT5 的实际路径
# 通常在: ~/Library/Application Support/com.neu.crossover/Bottles/MetaTrader 5/...
# 或者: ~/Library/Application Support/MetaTrader 5/...
# 您可以在 MT5 中点击 "File" -> "Open Data Folder" 来找到它
export SIGNAL_DIR="/Users/hanjianglin/Library/Application Support/net.metaquotes.wine.gecko/..." 

# Supabase 配置 (从 .env.local 读取)
if [ -f .env.local ]; then
    export $(grep -v '^#' .env.local | xargs)
fi

# 设置 Python 路径以包含 gRPC 生成代码
export PYTHONPATH=$PYTHONPATH:$(pwd)/ai-engine/src:$(pwd)/trading-bridge/src

# 2. 激活虚拟环境
source ai-engine/venv/bin/activate

# 3. 启动 Bridge
echo "🚀 正在启动本地 Trading Bridge..."
echo "📂 监听信号目录: $SIGNAL_DIR"
echo "📡 连接数据库: $NEXT_PUBLIC_SUPABASE_URL"

python trading-bridge/src/main.py

