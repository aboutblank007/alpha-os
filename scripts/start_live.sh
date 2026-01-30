#!/bin/bash
# AlphaOS v4 Live Trading Script
#
# 用法：
#   ./scripts/start_live.sh --model models/v4/run_001
#   ./scripts/start_live.sh --model models/v4/run_001 --dry-run
#
# 必需参数：
#   --model <dir>    模型目录（包含 cfc_encoder.pt + xgb_model.json）
#
# 可选参数：
#   --dry-run        模拟交易（不执行真实订单）
#   --symbol         交易品种（默认 XAUUSD）
#   --volume         订单手数（默认 0.01）
#   --min-conf       最小置信度阈值（默认 0.65）

set -e

# 默认参数
MODEL=""
DRY_RUN=""
SYMBOL="XAUUSD"
VOLUME="0.01"
MIN_CONF="0.65"
ZMQ_TICK_PORT="5555"
ZMQ_ORDER_PORT="5556"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --volume)
            VOLUME="$2"
            shift 2
            ;;
        --min-conf)
            MIN_CONF="$2"
            shift 2
            ;;
        --zmq-tick-port)
            ZMQ_TICK_PORT="$2"
            shift 2
            ;;
        --zmq-order-port)
            ZMQ_ORDER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "AlphaOS v4 Live Trading"
            echo ""
            echo "用法: $0 --model <model_dir> [options]"
            echo ""
            echo "必需参数:"
            echo "  --model <dir>        模型目录（包含 cfc_encoder.pt + xgb_model.json）"
            echo ""
            echo "可选参数:"
            echo "  --dry-run            模拟交易（不执行真实订单）"
            echo "  --symbol <sym>       交易品种（默认: XAUUSD）"
            echo "  --volume <vol>       订单手数（默认: 0.01）"
            echo "  --min-conf <conf>    最小置信度阈值（默认: 0.65）"
            echo "  --zmq-tick-port <p>  ZeroMQ tick 端口（默认: 5555）"
            echo "  --zmq-order-port <p> ZeroMQ order 端口（默认: 5556）"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [[ -z "$MODEL" ]]; then
    echo "错误: --model 参数是必需的"
    echo "用法: $0 --model models/v4/run_001 [--dry-run]"
    exit 1
fi

# 检查模型目录
if [[ ! -d "$MODEL" ]]; then
    echo "错误: 模型目录不存在: $MODEL"
    echo "请先训练模型: ./scripts/start_training.sh --data your_data.csv --output $MODEL"
    exit 1
fi

# 检查必需文件
if [[ ! -f "$MODEL/xgb_model.json" ]] && [[ ! -f "$MODEL/xgb_model.ubj" ]]; then
    echo "警告: 未找到 XGBoost 模型文件（xgb_model.json 或 xgb_model.ubj）"
fi

echo "============================================"
echo "AlphaOS v4 Live Trading"
echo "============================================"
echo "模型目录:   $MODEL"
echo "交易品种:   $SYMBOL"
echo "订单手数:   $VOLUME"
echo "最小置信度: $MIN_CONF"
if [[ -n "$DRY_RUN" ]]; then
    echo "模式:       模拟交易（DRY RUN）"
else
    echo "模式:       实盘交易"
fi
echo "============================================"
echo ""

if [[ -z "$DRY_RUN" ]]; then
    echo "警告: 这将启动实盘交易！"
    echo "请确保 MT5 EA 正在运行并已连接。"
    echo ""
    read -p "按 Enter 继续，或 Ctrl+C 取消..."
fi

# 设置环境变量
export ALPHAOS_MODE="live"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 构建命令
CMD="python -m alphaos.v4.cli serve"
CMD="$CMD --model $MODEL"
CMD="$CMD --symbol $SYMBOL"
CMD="$CMD --volume $VOLUME"
CMD="$CMD --min-confidence $MIN_CONF"
CMD="$CMD --zmq-tick-port $ZMQ_TICK_PORT"
CMD="$CMD --zmq-order-port $ZMQ_ORDER_PORT"

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD --dry-run"
fi

# 运行 v4 推理服务
$CMD
