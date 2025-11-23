FROM scottyhardy/docker-wine:latest

# Install Python, tools, and Supervisor
USER root
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    supervisor \
    wget \
    unzip \
    curl \
    gnupg2 \
    software-properties-common \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --break-system-packages -r requirements.txt

# Create directories
RUN mkdir -p /app/mt5 /app/data /var/log/supervisor

# Copy configuration files
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment variables
ENV DISPLAY=:0
ENV WINEPREFIX=/root/.wine
ENV WINEDLLOVERRIDES="mscoree,mshtml="

# Expose ports
EXPOSE 8000 5900

ENTRYPOINT ["/entrypoint.sh"]
