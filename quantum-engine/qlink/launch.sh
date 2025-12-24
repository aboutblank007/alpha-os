#!/bin/bash
# =========================================================
# Q-Link 量子交易系统启动脚本 (多品种版)
# 
# 用法：
#   ./launch.sh                    # 本地模式，所有品种
#   ./launch.sh 192.168.3.10       # 连接远程 MT5，所有品种
#   ./launch.sh 192.168.3.10 xau   # 仅 XAUUSD
#   ./launch.sh 192.168.3.10 btc   # 仅 BTCUSD
#   ./launch.sh 192.168.3.10 all   # 所有品种
# =========================================================

set -e

PIDS=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANTUM_ENGINE_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_BASE_DIR="${QUANTUM_ENGINE_DIR}/models"
LOG_DIR="${QUANTUM_ENGINE_DIR}/logs"
COMBINED_LOG="${LOG_DIR}/combined.log"

# 远程 MT5 IP
MT5_HOST="${1:-127.0.0.1}"
BIND_ADDR="tcp://${MT5_HOST}"

# 品种选择
SYMBOL_MODE="${2:-all}"

# 端口配置 (每个品种用不同的 Alpha-to-Risk 端口)
XAU_SIGNAL_PORT=5560
BTC_SIGNAL_PORT=5561

echo ""
echo "==========================================="
echo "  Q-Link 量子交易系统 (多品种版)"
echo "==========================================="
echo ""
echo "📡 MT5 主机: ${MT5_HOST}"
echo "📊 品种模式: ${SYMBOL_MODE}"
echo ""

# 创建日志目录
mkdir -p "$LOG_DIR"
> "$COMBINED_LOG"

# ================== 环境配置 (M2 Pro 优化) ==================
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export KMP_BLOCKTIME=0
export PENNYLANE_NUM_THREADS=8
export ZMQ_HOST="${MT5_HOST}"
export PYTHONPATH="${QUANTUM_ENGINE_DIR}:${PYTHONPATH}"

# 激活虚拟环境
VENV_PATH="${QUANTUM_ENGINE_DIR}/.venv"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "✅ 已激活虚拟环境"
else
    echo "⚠️  虚拟环境不存在: $VENV_PATH"
    exit 1
fi

pip install pyzmq fastapi uvicorn -q 2>/dev/null || true

# ================== 启动 API 网关 ==================
echo "🚀 启动 API Gateway..."
python3 -u api_gateway.py --host 0.0.0.0 --port 8000 2>&1 | while read -r line; do echo "[Gateway] $line" >> "$COMBINED_LOG"; done &
PID_GATEWAY=$!
PIDS="$PIDS $PID_GATEWAY"
echo "   Gateway PID: $PID_GATEWAY ✅"
sleep 1

# ================== 模型检查函数 ==================
check_model() {
    local model_dir=$1
    local required_files=("artifacts.json" "feature_transformer.pkl" "target_scaler.pkl" "quantum_regressor_best.pt")
    
    for f in "${required_files[@]}"; do
        if [ ! -s "${model_dir}/${f}" ]; then
            return 1
        fi
    done
    return 0
}

# ================== 确定启动品种 ==================
cd "$SCRIPT_DIR"

start_engines() {
    local symbol_key=$1
    local symbol_name=$2
    local model_dir=$3
    local signal_port=$4
    local command_port=$5
    local market_port=$6
    
    # 检查模型完整性
    if ! check_model "$model_dir"; then
        echo "❌ ${symbol_name} 模型文件不完整或损坏: ${model_dir}"
        return 1
    fi
    
    echo ""
    echo "🚀 启动 ${symbol_name} 引擎..."
    echo "   模型目录: ${model_dir}"
    echo "   信号端口: ${signal_port}"
    echo "   命令端口: ${command_port}"
    echo "   市场端口: ${market_port}"
    
    # Alpha Engine
    python3 -u alpha_engine.py --model-dir "$model_dir" --bind "$BIND_ADDR" --signal-port "$signal_port" --market-port "$market_port" 2>&1 | \
        while IFS= read -r line; do
            echo "[${symbol_key}-Alpha] $line" | tee -a "$COMBINED_LOG"
            if [[ "$line" == *"信号:"* ]]; then
                echo "────────────────────────────────────────" | tee -a "$COMBINED_LOG"
            fi
        done &
    PID_ALPHA=$!
    PIDS="$PIDS $PID_ALPHA"
    echo "   Alpha Engine PID: $PID_ALPHA ✅"
    
    sleep 0.5
    
    # Risk Engine
    python3 -u risk_engine.py --model-dir "$model_dir" --bind "$BIND_ADDR" --signal-port "$signal_port" --command-port "$command_port" 2>&1 | \
        while IFS= read -r line; do
            echo "[${symbol_key}-Risk]  $line" | tee -a "$COMBINED_LOG"
            if [[ "$line" == *"生成订单"* ]] || [[ "$line" == *"拒绝"* ]]; then
                echo "════════════════════════════════════════" | tee -a "$COMBINED_LOG"
            fi
        done &
    PID_RISK=$!
    PIDS="$PIDS $PID_RISK"
    echo "   Risk Engine PID: $PID_RISK ✅"
    
    # 尝试应用 macOS taskpolicy 优先级优化 (针对 M2 Pro 异构核心)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # 将 Alpha 引擎设为高优先级 (User Interactive)
        taskpolicy -t 5 -p $PID_ALPHA 2>/dev/null || true
        # 将 Risk 引擎设为正常优先级
        taskpolicy -t 3 -p $PID_RISK 2>/dev/null || true
    fi
    
    return 0
}

# ================== 启动服务 ==================
STARTED=0

# 单品种模式：只启动 XAUUSD
if [ "$SYMBOL_MODE" = "all" ] || [ "$SYMBOL_MODE" = "xau" ]; then
    if start_engines "XAU" "XAUUSD" "${MODEL_BASE_DIR}/xau_v2_alpha101" 5560 5558 5557; then
        STARTED=$((STARTED + 1))
    fi
fi

if [ $STARTED -eq 0 ]; then
    echo "❌ 没有启动任何品种"
    exit 1
fi

# ================== 状态显示 ==================
echo ""
echo "==========================================="
echo "  Q-Link 系统已启动 (${STARTED} 个品种)"
echo "==========================================="
echo ""
echo "🔌 ZMQ 连接到 ${MT5_HOST}:"
echo "   Market Stream : tcp://${MT5_HOST}:5557"
echo "   Command Bus   : tcp://${MT5_HOST}:5558"
echo "   State Sync    : tcp://${MT5_HOST}:5559"
echo ""
echo "📋 合并日志: $COMBINED_LOG"
echo ""
echo "按 Ctrl+C 停止所有服务..."
echo ""
echo "────────────────────────────────────────"
echo ""

# ================== 等待退出 ==================
cleanup() {
    echo ""
    echo "正在停止服务..."
    for pid in $PIDS; do
        kill $pid 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "✅ 服务已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

wait