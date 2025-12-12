# AlphaOS 部署与迁移指南

## 目录
- [部署架构](#部署架构)
- [首次部署](#首次部署)
- [云端数据迁移](#云端数据迁移)
- [故障排查](#故障排查)

---

## 部署架构

### 网络拓扑

所有服务运行在统一的 `alphaos-net` Docker 网络中：

```
alphaos-net (192.168.97.0/24)
├── Web (alpha-os-web:3001)
├── Bridge API (bridge-api:8000, gRPC:50051)
├── AI Engine (ai-engine:50051)
├── MT5 VNC (mt5-vnc:3000)
└── Supabase 服务组
    ├── Kong API (54321)
    ├── PostgreSQL (54322)
    ├── Studio (54323)
    └── 其他微服务 (Auth, Storage, Realtime...)
```

**优势**：
- 统一网络简化服务发现（可通过容器名直接访问）
- Supabase 双网络模式：内部 `default` + 外部 `alphaos-net`
- 避免端口冲突：使用 54xxx 系列端口

---

## 首次部署

### 前置条件

1. **SSH 配置**：确保本地可 SSH 到远程主机
   ```bash
   # ~/.ssh/config
   Host macOS
       HostName 192.168.3.8
       User lootool
   ```

2. **环境变量配置**：编辑 `.env.local`
   ```bash
   NEXT_PUBLIC_SUPABASE_URL=http://192.168.3.8:54321
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
   ```

### 部署步骤

#### 1. 部署 Supabase（首次必须）

```bash
./deploy_orb.sh --supabase
```

**等待所有容器变为 healthy 状态**（约 1-2 分钟）。

访问 Studio 验证：`http://192.168.3.8:54323`

#### 2. 部署应用服务

```bash
# 一键部署全部
./deploy_orb.sh

# 或分步部署
./deploy_orb.sh --web
./deploy_orb.sh --bridge --ai
```

#### 3. 应用数据库 Schema

```bash
bash scripts/apply_schema.sh
```

此脚本会：
- 上传 `src/db/cloud_schema.sql` 到远程
- 通过 `docker exec` 应用到 Supabase DB
- 创建所有必需的表、索引、RLS 策略

---

## 云端数据迁移

### 场景

将生产数据从云端 Supabase 迁移到本地部署。

### 准备工作

1. **获取云端凭证**
   - 登录 [Supabase Dashboard](https://supabase.com/dashboard/)
   - 进入项目设置 → API
   - 复制 `service_role` key

2. **确认本地 Supabase 已部署且 Schema 已应用**

### 执行迁移

```bash
./deploy_orb.sh --migrate
```

**迁移过程**：
1. 读取本地 `.env.local` 获取云端凭证
2. 通过 Supabase REST API 拉取云端数据
3. 使用本地 Service Role Key 写入本地数据库
4. 支持 upsert（重复执行不会产生重复数据）

**迁移的表**：
- `user_preferences`
- `automation_rules`
- `journal_notes`
- `signals`
- `trades`
-训练数据集`trading_datasets`

> **注意**：`training_signals` 可能因云端 RLS 权限限制无法迁移，这是正常的。

### 验证迁移结果

```bash
# SSH 到远程
ssh macOS

# 查询数据行数
docker exec supabase-db psql -U postgres -d postgres \
  -c "SELECT 'signals' as table, COUNT(*) FROM signals
      UNION ALL
      SELECT 'trades', COUNT(*) FROM trades;"
```

---

## 故障排查

### 问题 1: Web 构建失败 - "supabaseUrl is required"

**症状**：
```
Error: Invalid environment variables
NEXT_PUBLIC_SUPABASE_URL: [ 'Supabase URL must be a valid URL' ]
```

**原因**：Docker 构建时环境变量为空。

**解决方案**：
已在 `deploy_orb.sh` 中修复（Line 127-131）。脚本会在本地加载 `.env.local`，然后变量值直接注入远程 docker-compose 文件。

**验证修复**：
```bash
ssh macOS "cat ~/alpha-os/deploy/web/docker-compose.yml | grep SUPABASE"
# 应显示实际 URL，而非空值
```

---

### 问题 2: Supabase Studio 无法访问（端口冲突）

**症状**：
- 访问 `http://IP:54323` 无响应
- `docker ps` 显示 Studio 端口未映射

**原因**：MT5 已占用 3000 端口，Studio 默认需要映射到宿主机但未配置正确映射。

**解决方案**：
修改 `deploy_orb.sh` (Line 506-509)，添加：
```yaml
studio:
  ports:
    - "54323:3000"
```

然后重新部署 Supabase。

---

### 问题 3: AI 不写入数据

**症状**：
- AI Engine 日志显示处理信号
- 但 Supabase 数据库无新增记录

**原因**：AI Engine 和 Bridge API 缺少 Supabase 环境变量。

**解决方案**：
已修复 `deploy_orb.sh`（Line 252-253, 209-210）。确保：
```yaml
environment:
  - SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL}  # 不要用 \${...}
  - SUPABASE_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY}
```

**验证修复**：
```bash
ssh macOS "docker exec ai-engine env | grep SUPABASE"
# 应显示完整 URL 和 Key
```

重新部署：
```bash
./deploy_orb.sh --ai --bridge
```

---

### 问题 4: 网络碎片化（多个 Docker 网络）

**症状**：
```bash
docker network ls
# 显示多个网络：alphaos-net, alpha-os_default, supabase_default, bridge...
```

**解决方案**：
已通过 `docker-compose.override.yml` 统一。所有服务加入 `alphaos-net`，Supabase 同时保留 `default` 网络以支持内部微服务通信。

**验证网络**：
```bash
ssh macOS "docker network inspect alphaos-net --format '{{range .Containers}}{{.Name}} {{end}}'"
# 应列出所有 AlphaOS 和 Supabase 容器
```

---

## 高级配置

### 自定义部署脚本

编辑 `deploy_orb.sh` 顶部变量：

```bash
REMOTE_HOST="macOS"      # SSH host 名称
REMOTE_DIR="~/alpha-os"  # 远程部署路径
```

### 数据持久化路径

所有数据存储在远程主机：
```
~/alpha-os-data/
├── supabase/          # Supabase 数据与配置
└── mt5_config/        # MT5 配置文件
```

**备份建议**：定期备份此目录。

---

## 常用命令

```bash
# 查看所有服务状态
ssh macOS "docker ps"

# 查看特定服务日志
ssh macOS "docker logs ai-engine --tail 50"
ssh macOS "docker logs bridge-api --tail 50"

# 重启服务
ssh macOS "cd ~/alpha-os/deploy/ai && docker compose restart"

# 进入数据库
ssh macOS "docker exec -it supabase-db psql -U postgres -d postgres"

# 检查网络连通性（从 Bridge API 访问 Supabase）
ssh macOS "docker exec bridge-api curl -s http://kong:8000/rest/v1/"
```

---

## 外部资源

- [deploy_orb.sh](../deploy_orb.sh) - 部署脚本源码
- [apply_schema.sh](../scripts/apply_schema.sh) - Schema 应用脚本
- [migrate_from_cloud.py](../scripts/migrate_from_cloud.py) - 数据迁移脚本
- [cloud_schema.sql](../src/db/cloud_schema.sql) - 完整数据库 Schema
