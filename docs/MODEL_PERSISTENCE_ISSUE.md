# 模型持久化问题与解决方案

**日期**: 2025-12-12  
**问题**: 容器内训练的新模型被 deploy 覆盖  
**严重性**: 🔴 高危

---

## 🐛 问题描述

### 当前工作流程
```bash
# 1. 在容器内训练模型
ssh macOS "docker exec ai-engine python train_filter.py"
# 模型保存到容器内: /app/models/lgbm_*.txt

# 2. 重新部署容器
./deploy_orb.sh --ai
# rsync 将本地旧模型同步到远程，覆盖容器内新训练的模型！
```

### 根本原因
- **deploy_orb.sh** 使用 `rsync` 同步整个 `ai-engine/` 目录
- 包括 `ai-engine/models/` 目录
- 本地的旧模型会覆盖容器内刚刚训练的新模型

### 验证
```bash
# 容器内新模型（训练后）
Dec 12 12:07  # 训练时间

# 本地旧模型
Dec 11 23:28  # 上次同步时间

# 部署后 - 新模型被覆盖！❌
```

---

## ✅ 解决方案

### 方案 1: 训练后从容器复制模型到本地（推荐）

```bash
# 在 train_filter.py 之后立即执行
scp macOS:/path/to/models/* ./ai-engine/models/
# 或
ssh macOS "docker cp ai-engine:/app/models/. /tmp/" && \
  scp macOS:/tmp/lgbm_*.txt ./ai-engine/models/
```

**优点**: 本地始终有最新模型备份  
**缺点**: 需要额外步骤

### 方案 2: 使用 Docker Volume 持久化（最佳）

修改 `deploy_orb.sh`:

```yaml
# AI Engine docker-compose.yml
services:
  ai:
    volumes:
      - ai_models:/app/models  # 模型持久化
      
volumes:
  ai_models:
    external: true  # 创建外部卷
```

部署前创建数据卷:
```bash
ssh macOS "docker volume create ai_models"
```

**优点**: 模型永久保存，不会被覆盖  
**缺点**: 需要修改部署脚本

### 方案 3: rsync 排除 models目录

修改 `deploy_orb.sh` Line ~145:

```bash
# 原来
rsync -avz --delete ai-engine/ macOS:/path/to/ai-engine/

# 改为
rsync -avz --delete --exclude='models/*.txt' ai-engine/ macOS:/path/to/ai-engine/
```

**优点**: 简单，不覆盖现有模型  
**缺点**: 首次部署时没有模型

### 方案 4: 自动训练脚本（推荐组合方案2+4）

创建 `scripts/train_and_deploy.sh`:

```bash
#!/bin/bash
# 联合训练和部署流程

echo "1️⃣ Ingest data and train models..."
ssh macOS "docker exec ai-engine bash -c 'cd /app && \
  python src/ingest_mql_data.py && \
  python enhance_features.py && \
  python train_filter.py'"

echo "2️⃣ Backup models from container..."
ssh macOS "docker cp ai-engine:/app/models/. /tmp/ai_models/"
scp -r macOS:/tmp/ai_models/* ./ai-engine/models/

echo "3️⃣ Deploy AI Engine..."
./deploy_orb.sh --ai

echo "✅ Training and deployment complete!"
```

---

## 🚀 立即修复（临时）

**检查当前模型状态**:
```bash
# 1. 查看容器内模型时间
ssh macOS "docker exec ai-engine ls -lh /app/models/"

# 2. 如果是旧模型（Dec 11），重新训练
ssh macOS "docker exec ai-engine bash -c 'cd /app && python train_filter.py'"

# 3. 验证新模型
ssh macOS "docker exec ai-engine ls -lh /app/models/"
# 应该显示 Dec 12 20:xx

# 4. 备份到本地
mkdir -p ./ai-engine/models_backup/$(date +%Y%m%d)
ssh macOS "docker cp ai-engine:/app/models/lgbm_BTCUSD.txt -" > \
  ./ai-engine/models_backup/$(date +%Y%m%d)/lgbm_BTCUSD.txt
# 重复其他品种...
```

---

## 📋 实施计划

### 短期（立即）
- [x] 验证当前模型是否被覆盖
- [ ] 如被覆盖，重新训练
- [ ] 手动备份新模型到本地

### 中期（本周）
- [ ] 实施方案2（Docker Volume）
- [ ] 修改 deploy_orb.sh 添加 volume 配置
- [ ] 创建 train_and_deploy.sh 统一脚本

### 长期（下周）
- [ ] 实现模型版本控制（Git LFS或专用存储）
- [ ] 添加模型校验和比对
- [ ] 自动化训练→备份→部署流程

---

## ⚠️ 风险评估

**如果不修复**:
- 每次部署都会丢失新训练的模型
- 所有训练努力白费
- 系统使用的始终是第一次的旧模型

**已发生次数**: 
- 第一次训练（Dec 11 23:28）✅ 保留
- 第二次训练（Dec 12 12:07）❌ 很可能被覆盖

---

## 🔍 验证清单

部署后必须检查:
- [ ] 容器内模型时间戳是最新的
- [ ] 模型文件大小有变化（如果特征/数据改变）
- [ ] AI Engine 日志显示正确加载模型
- [ ] 本地有模型备份

---

**优先级**: 🔴 P0（紧急）  
**影响**: 所有AI训练工作

**下次部署前必须先备份模型！**
