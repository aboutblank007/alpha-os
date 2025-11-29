#!/bin/bash

# 配置
SERVER="alphaos"
REMOTE_DIR="~/alpha-os"
CONTAINER_NAME="mt5-vnc" # docker-compose.yml 中定义的容器名
LOCAL_OUTPUT_DIR="./ai-engine"
OUTPUT_FILENAME="backtest_data.tar.gz"

# 容器内路径 (Linux 格式，对应 Windows 的 C:\Program Files\MetaTrader 5\...)
# 注意：Agent-127.0.0.1-3002 是你的具体代理端口
# 这里的路径是基于 gmag11/metatrader5_vnc 镜像的标准 Wine 路径
CONTAINER_PATH="/config/.wine/drive_c/Program Files/MetaTrader 5/Tester/Agent-127.0.0.1-3002/MQL5/Files/AlphaOS/Signals"

echo "🚀 开始从 $SERVER 获取训练数据..."

# 1. 在远程服务器上执行打包操作
echo "📦 正在远程容器中打包日志文件..."
echo "   目标路径: $CONTAINER_PATH"

ssh $SERVER "
    # 确保临时目录存在
    mkdir -p /tmp/alphaos_export
    
    # 检查容器是否运行
    if ! docker ps | grep -q $CONTAINER_NAME; then
        echo '❌ 错误: 容器 $CONTAINER_NAME 未运行'
        exit 1
    fi

    # 使用 docker exec 打包文件
    # 即使路径中有空格，也要小心处理
    echo '   正在查找并打包 JSON 日志...'
    
    docker exec $CONTAINER_NAME bash -c '
        # 检查目录是否存在
        if [ ! -d \"$CONTAINER_PATH\" ]; then
            echo \"❌ 错误: 目录不存在: $CONTAINER_PATH\"
            echo \"   提示: 请检查 Agent 端口号是否正确 (当前: 3002)\"
            ls -d \"/config/.wine/drive_c/Program Files/MetaTrader 5/Tester/\"*
            exit 1
        fi

        cd \"$CONTAINER_PATH\" && \
        # 检查是否有 json 文件
        if ! ls *.json >/dev/null 2>&1; then
            echo \"⚠️ 警告: 该目录下没有找到 .json 文件\"
            exit 1
        fi
        
        tar -czf /tmp/signals_export.tar.gz ./*.json
    '
    
    if [ \$? -eq 0 ]; then
        # 将容器内的包复制到宿主机
        docker cp $CONTAINER_NAME:/tmp/signals_export.tar.gz /tmp/alphaos_export/$OUTPUT_FILENAME
        
        # 清理容器内临时文件
        docker exec $CONTAINER_NAME rm /tmp/signals_export.tar.gz
        exit 0
    else
        exit 1
    fi
"

if [ $? -ne 0 ]; then
    echo "❌ 远程打包失败，请检查以上错误信息。"
    exit 1
fi

# 2. 下载文件到本地
echo "⬇️  正在下载文件到 $LOCAL_OUTPUT_DIR/$OUTPUT_FILENAME ..."
mkdir -p $LOCAL_OUTPUT_DIR
scp $SERVER:/tmp/alphaos_export/$OUTPUT_FILENAME $LOCAL_OUTPUT_DIR/$OUTPUT_FILENAME

# 3. 清理远程临时文件
echo "🧹 清理远程临时文件..."
ssh $SERVER "rm -rf /tmp/alphaos_export"

echo "✅ 数据获取完成！"
echo "👉 你现在可以运行以下命令来处理数据："
echo "   cd ai-engine && python export_training_data.py"

