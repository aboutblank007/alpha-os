#!/bin/bash

# [Ref: 交易系统前端功能设计.MD] 8.0 启动脚本
# 启动前端开发服务器和 Python API Gateway (支持 Mock 模式)

echo "🚀 Starting Alpha-OS Development Environment..."

# 1. 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    exit 1
fi

# 2. 启动 Python Gateway (Mock Mode)
echo "🐍 Starting Python Gateway (Mock Mode)..."
export MOCK_MODE=true
export PYTHONPATH=$PYTHONPATH:$(pwd)/quantum-engine
# 使用 nohup 后台运行，日志输出到 gateway.log
nohup python3 quantum-engine/qlink/api_gateway.py > gateway.log 2>&1 &
GATEWAY_PID=$!
echo "✅ Gateway running at http://127.0.0.1:8000 (PID: $GATEWAY_PID)"

# 3. 启动 Next.js Frontend
echo "⚛️  Starting Next.js Frontend..."
npm run dev &
NEXT_PID=$!

# 4. 优雅退出处理
cleanup() {
    echo "🛑 Shutting down..."
    kill $GATEWAY_PID
    kill $NEXT_PID
    exit
}

trap cleanup SIGINT

# 等待子进程
wait
