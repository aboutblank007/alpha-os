#!/bin/bash

# 遇到错误立即退出
set -e

echo "🚀 启动 AI 训练全流程 (Fetch -> Process -> Train)..."

# 0. 检查环境
echo "🔍 步骤 0: 检查 Python 环境..."
if ! python3 -c "import lightgbm, pandas, sklearn" 2>/dev/null; then
    echo "⚠️  检测到缺少必要的 Python 库。"
    echo "   正在安装依赖..."
    pip install -r ai-engine/requirements.txt
fi

# 1. 获取数据
echo "📥 步骤 1: 从远程服务器获取日志..."
if [ -f "./fetch_training_data.sh" ]; then
    ./fetch_training_data.sh
else
    echo "❌ 错误: 找不到 fetch_training_data.sh"
    exit 1
fi

# 2. 处理数据
echo "🛠️ 步骤 2: 将日志处理为 CSV 格式..."
# 切换到 ai-engine 目录运行，因为 export 脚本假定文件在当前目录
cd ai-engine
if [ -f "backtest_data.tar.gz" ]; then
    python3 export_training_data.py
else
    echo "❌ 错误: ai-engine/backtest_data.tar.gz 不存在，下载可能失败。"
    exit 1
fi
# 返回根目录
cd ..

# 3. 准备训练文件
echo "📋 步骤 3: 准备训练数据..."
# train.py 默认在根目录寻找 training_data.csv
if [ -f "ai-engine/training_data.csv" ]; then
    cp ai-engine/training_data.csv ./training_data.csv
    echo "   已将数据复制到项目根目录。"
else
    echo "❌ 错误: CSV 生成失败。"
    exit 1
fi

# 4. 训练模型
echo "🧠 步骤 4: 训练 AI 模型..."
python3 ai-engine/src/train.py

echo "=================================================="
echo "✅ 全流程完成！"
echo "📂 模型已保存至: ai-engine/models/lgbm_scalping_v1.txt"
echo "📊 训练数据位于: ./training_data.csv"
echo "=================================================="

