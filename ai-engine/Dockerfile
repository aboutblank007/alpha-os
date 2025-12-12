# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for LightGBM and gRPC
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ai-engine/requirements.txt .

# Install dependencies
# 1) 安装 CPU 版 torch，强制使用官方 CPU 索引，避免拉取 CUDA 依赖
RUN pip install --default-timeout=1000 --no-cache-dir torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 其他依赖使用国内镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple https://pypi.douban.com/simple" && \
    pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.douban.com"

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy AI Engine source code
COPY ai-engine/src /app/src
COPY ai-engine/models /app/models
COPY ai-engine/ai_decisions.csv /app/ai_decisions.csv

# Copy training scripts and entrypoint
COPY ai-engine/auto_learner.py /app/auto_learner.py
COPY ai-engine/train_filter.py /app/train_filter.py
COPY ai-engine/train_scanner.py /app/train_scanner.py
COPY ai-engine/enhance_features.py /app/enhance_features.py
COPY ai-engine/docker_entrypoint.sh /app/docker_entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/docker_entrypoint.sh

# Copy Protobuf definitions (ensure they are available)
# We'll assume they are in src/proto based on project structure
# If they are symlinked or elsewhere, we might need to adjust
COPY ai-engine/src/proto /app/src/proto

# Set Python path
ENV PYTHONPATH=/app

# Expose gRPC port
EXPOSE 50051

# Run the AI Engine
CMD ["./docker_entrypoint.sh"]
