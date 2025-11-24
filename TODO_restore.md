## 🔧 服务器连接与容器管理

### 服务器连接方式

**连接命令：**
```bash
ssh alphaos
```

### Docker 容器管理

**项目路径：**
```bash
cd ~/trading-bridge/docker
```

**容器列表：**
- `mt5-vnc`: MT5 交易平台 (VNC: http://49.235.153.73:3000)
- `bridge-api`: API 服务 (http://api.lootool.cn:8000)

**前端服务：**
- `alpha-os-web`: 前端页面 (http://49.235.153.73:3001)
  - 部署路径: `~/alpha-os`
  - 独立容器部署，通过 `deploy_service.sh` 管理

**常用命令：**

```bash
# 1. 交易桥接服务 (MT5 + API)
cd ~/trading-bridge/docker
docker-compose ps
docker-compose logs -f
docker-compose restart

# 2. 前端服务
cd ~/alpha-os
docker-compose ps
docker-compose logs -f
docker-compose restart
```

### 文件同步与部署

使用 `deploy_service.sh` 脚本一键部署各服务：

**用法：**
```bash
./deploy_service.sh <服务名称>
```

**示例：**

1. **更新前端 (Frontend):**
   ```bash
   ./deploy_service.sh frontend
   # 或者
   ./deploy_service.sh web
   ```

2. **更新后端 Bridge API:**
   ```bash
   ./deploy_service.sh bridge-api
   ```
   *(脚本会自动处理 `.env` 注入和子目录构建)*

3. **更新 MT5 容器:**
   ```bash
   ./deploy_service.sh mt5
   ```

---

**同步 MQL5 EA (需要手动操作):**
```bash
scp trading-bridge/mql5/BridgeEA.mq5 alphaos:~/trading-bridge/mql5/
# 然后在 VNC 中重新编译 EA
```

---
