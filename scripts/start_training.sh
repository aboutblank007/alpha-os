#!/bin/bash
# AlphaOS v4 Training Script
#
# 用法：
#   ./scripts/start_training.sh --data path/to/data.csv
#   ./scripts/start_training.sh --data path/to/data.csv --output models/v4/run_001 --epochs 50
#
# 支持的 CSV 格式：
#   - 标准格式: time_msc, bid, ask, [volume]
#   - MT5 导出格式: Time (EET), Bid, Ask, [AskVolume, BidVolume]
#
# v4 训练流程：
#   1. Tick → Volume Bars 采样
#   2. 降噪预处理（Kalman/Wavelet）
#   3. 特征计算
#   4. Primary 信号生成（PivotSuperTrend + FVG）
#   5. Meta-Labeling（Triple Barrier）
#   6. CfC 编码器训练
#   7. XGBoost Meta Head 训练
#   8. 保存模型（cfc_encoder.pt + xgb_model.json + schema.json）

set -e

# 默认参数
DATA=""
OUTPUT="models/v4"
EPOCHS=""
MAX_TICKS=""
SAMPLING_MODE="volume_bars"
TARGET_VOLUME="100"
DENOISE_MODE="kalman"
CV_SPLITS="5"
USE_CPCV=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max-ticks)
            MAX_TICKS="$2"
            shift 2
            ;;
        --sampling-mode)
            SAMPLING_MODE="$2"
            shift 2
            ;;
        --target-volume)
            TARGET_VOLUME="$2"
            shift 2
            ;;
        --denoise-mode)
            DENOISE_MODE="$2"
            shift 2
            ;;
        --cv-splits)
            CV_SPLITS="$2"
            shift 2
            ;;
        --use-cpcv)
            USE_CPCV="--use-cpcv"
            shift
            ;;
        -h|--help)
            echo "AlphaOS v4 Training Script"
            echo ""
            echo "用法: $0 --data <file> [options]"
            echo ""
            echo "必需参数:"
            echo "  --data <file>           Tick 数据 CSV 文件路径"
            echo ""
            echo "可选参数:"
            echo "  --output <dir>          输出目录（默认: models/v4）"
            echo "  --epochs <n>            训练轮数（默认: 50）"
            echo "  --max-ticks <n>         最大加载 Tick 数（用于测试）"
            echo ""
            echo "采样选项:"
            echo "  --sampling-mode <mode>  采样模式: volume_bars, tick_imbalance（默认: volume_bars）"
            echo "  --target-volume <n>     Volume Bar 目标成交量（默认: 100）"
            echo ""
            echo "降噪选项:"
            echo "  --denoise-mode <mode>   降噪模式: wavelet, kalman, combined, none（默认: kalman）"
            echo ""
            echo "交叉验证选项:"
            echo "  --cv-splits <n>         CV 折数（默认: 5）"
            echo "  --use-cpcv              使用 Combinatorial Purged CV（更慢但更稳健）"
            echo ""
            echo "v4 训练产出:"
            echo "  - cfc_encoder.pt        CfC 编码器权重"
            echo "  - xgb_model.json        XGBoost Meta Head"
            echo "  - schema.json           特征 Schema"
            echo "  - config.json           训练配置"
            echo "  - results.json          训练结果"
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
if [[ -z "$DATA" ]]; then
    echo "错误: --data 参数是必需的"
    echo "用法: $0 --data path/to/data.csv [--output models/v4]"
    echo "使用 --help 查看更多选项"
    exit 1
fi

# 检查数据文件
if [[ ! -f "$DATA" ]]; then
    echo "错误: 数据文件不存在: $DATA"
    exit 1
fi

# 查找 Python
PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 构建命令
CMD="$PYTHON -m alphaos.v4.cli train"
CMD="$CMD --data $DATA"
CMD="$CMD --output $OUTPUT"
CMD="$CMD --sampling-mode $SAMPLING_MODE"
CMD="$CMD --target-volume $TARGET_VOLUME"
CMD="$CMD --denoise-mode $DENOISE_MODE"
CMD="$CMD --cv-splits $CV_SPLITS"

if [[ -n "$EPOCHS" ]]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [[ -n "$MAX_TICKS" ]]; then
    CMD="$CMD --max-ticks $MAX_TICKS"
fi

if [[ -n "$USE_CPCV" ]]; then
    CMD="$CMD $USE_CPCV"
fi

echo "============================================"
echo "AlphaOS v4 Training"
echo "============================================"
echo "数据:          $DATA"
echo "输出:          $OUTPUT"
echo "采样模式:      $SAMPLING_MODE"
echo "目标成交量:    $TARGET_VOLUME"
echo "降噪模式:      $DENOISE_MODE"
echo "CV 折数:       $CV_SPLITS"
if [[ -n "$EPOCHS" ]]; then
    echo "训练轮数:      $EPOCHS"
fi
if [[ -n "$MAX_TICKS" ]]; then
    echo "最大 Ticks:    $MAX_TICKS"
fi
echo "============================================"

# 创建输出目录
mkdir -p "$OUTPUT"

# 设置环境变量
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 运行 v4 训练
$CMD

echo "============================================"
echo "训练完成！"
echo "模型已保存到: $OUTPUT"
echo ""
echo "模型文件:"
echo "  - $OUTPUT/cfc_encoder.pt"
echo "  - $OUTPUT/xgb_model.json"
echo "  - $OUTPUT/schema.json"
echo "============================================"
