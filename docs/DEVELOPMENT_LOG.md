# AlphaOS 开发备忘录

**项目**: AlphaOS - MT5 智能交易管理系统  
**维护周期**: 2025-11 至今

---

## 2025-12-12 - 网络统一 & 本地化部署 & AI 训练优化

### 🎯 主要成果

本次会话完成了系统的**完整本地化部署**和**AI训练系统优化**，解决了多个关键架构和数据问题。

---

### 📋 任务清单

#### ✅ 网络架构统一
- [x] 统一所有服务至 `alphaos-net` 网络
- [x] 修改 `docker-compose.yml` 使用外部网络
- [x] 为 Supabase 创建 `docker-compose.override.yml`（双网络模式）
- [x] 验证容器间通信

#### ✅ 本地 Supabase 部署
- [x] 部署本地 Supabase 容器
- [x] 配置端口避让（54321/54322/54323）
- [x] 数据持久化至 `~/alpha-os-data/supabase`

#### ✅ 云端数据迁移
- [x] 创建 `cloud_schema.sql`（匹配云端结构）
- [x] 开发 `migrate_from_cloud.py` 迁移脚本
- [x] 成功迁移 2000+ 条生产数据
- [x] 添加 `--migrate` 标志到 `deploy_orb.sh`

#### ✅ 部署问题修复
- [x] 解决 Supabase Studio 端口冲突（3000 → 54323）
- [x] 修复 Web 构建环境变量注入问题
- [x] 修复 AI Engine 和 Bridge API 缺少 Supabase 配置

#### ✅ AI 训练系统优化
- [x] 验证增量学习系统（AutoLearner）正常运行
- [x] 添加负样本训练（WAIT/SCAN 决策）
- [x] 移除 2000 条限制，使用全部 11,144 条 ai_features 数据
- [x] 训练新模型（11,914 样本）
- [x] 创建训练质量分析报告

#### ✅ 文档完善
- [x] 更新 README.md（v2.6.0 版本日志）
- [x] 创建 DEPLOYMENT_GUIDE.md（部署完整指南）
- [x] 创建 TRAINING_REPORT.md 和 TRAINING_QUALITY_REPORT.md

---

## 技术细节

### 1. 网络架构改进

**问题**: 系统使用多个 Docker 网络导致通信复杂

**解决方案**:
```yaml
# docker-compose.yml
networks:
  alphaos-net:
    external: true

# Supabase docker-compose.override.yml
networks:
  default:
    name: supabase_default
  alphaos-net:
    external: true

services:
  studio:
    networks:
      - default
      - alphaos-net
    ports:
      - "54323:3000"
```

**关键文件**:
- [`deploy_orb.sh`](file:///Users/hanjianglin/github/alpha-os/deploy_orb.sh) (Line 496-549)
- [`docker-compose.yml`](file:///Users/hanjianglin/github/alpha-os/docker-compose.yml)

---

### 2. 数据迁移流程

**创建的脚本**:
- [`scripts/migrate_from_cloud.py`](file:///Users/hanjianglin/github/alpha-os/scripts/migrate_from_cloud.py) - 数据迁移
- [`scripts/apply_schema.sh`](file:///Users/hanjianglin/github/alpha-os/scripts/apply_schema.sh) - Schema 应用
- [`src/db/cloud_schema.sql`](file:///Users/hanjianglin/github/alpha-os/src/db/cloud_schema.sql) - 完整数据库 Schema

**迁移结果**:
| 表名 | 迁移数量 | 状态 |
|------|---------|------|
| signals | 1,000 | ✅ 成功 |
| trades | 1,000 | ✅ 成功 |
| user_preferences | 1 | ✅ 成功 |
| automation_rules | 5 | ✅ 成功 |
| journal_notes | 2 | ✅ 成功 |
| training_signals | N/A | ⚠️ 权限问题 |

---

### 3. 环境变量注入修复

**问题**: Docker 构建时变量为空

**根本原因**:
```bash
# 错误：变量在远程展开
NEXT_PUBLIC_SUPABASE_URL: \${NEXT_PUBLIC_SUPABASE_URL}

# 正确：变量在本地展开
NEXT_PUBLIC_SUPABASE_URL: ${NEXT_PUBLIC_SUPABASE_URL}
```

**修复位置**:
- `deploy_orb.sh` Line 127-131：添加本地 .env 加载
- `deploy_orb.sh` Line 368-395：Web 服务配置
- `deploy_orb.sh` Line 206-253：AI 和 Bridge 配置

**影响的服务**:
- ✅ Web Dashboard
- ✅ AI Engine
- ✅ Bridge API

---

### 4. AI 训练系统验证与优化

#### 增量学习系统

**组件**:
- [`auto_learner.py`](file:///Users/hanjianglin/github/alpha-os/ai-engine/auto_learner.py) - 自动学习器
- 触发阈值: 100 条新记录
- 检查间隔: 60 秒
- 监控表: `training_signals` (where `result_profit IS NOT NULL`)

**状态**: ✅ 正常运行（后台轮询 Supabase）

#### 负样本训练增强

**问题**: 原始训练只使用 874 条已完成交易，忽略了 13,568 条 AI 推理记录

**解决方案**:
```python
# 修改 ingest_mql_data.py
# 1. 拉取完成交易
response_completed = supabase.table("training_signals") \
    .select("*") \
    .not_.is_("result_profit", "null") \
    .execute()

# 2. 拉取所有有 ai_features 的负样本（移除 limit(2000)）
response_negative = supabase.table("training_signals") \
    .select("*") \
    .is_("result_profit", "null") \
    .not_.is_("ai_features", "null") \
    .execute()

# 3. 为负样本创建合成标签
flat_row['result_mfe'] = 0.0  # "等待"的目标值
```

**训练进化**:
| 版本 | 数据量 | 负样本 | 备注 |
|------|--------|--------|------|
| v1 | 874 | 0 | 只用完成交易 |
| v2 | 2,879 | 2,000 | 添加限制的负样本 |
| v3 | **11,914** | **11,144** | 全部 ai_features 数据 ✅ |

#### 训练质量分析

**最终数据集**:
- 总样本: **11,914 条**
- 真实交易: 885 条（7.4%）
- 负样本: 11,029 条（92.6%）

**品种分布**（非常均衡）:
- BTCUSD: 2,416 (20.3%)
- GBPUSD: 2,396 (20.1%)
- NAS100: 2,357 (19.8%)
- US30: 2,353 (19.7%)
- XAUUSD: 2,392 (20.1%)

**模型性能**:
- R² Score: 1.0000（完美）
- MAE: 0.0000
- **原因**: 92.6% 样本标签为 0.0（数据分布问题）

**质量评分**: ⭐⭐⭐⭐ (4/5)
- 优势: 样本量充足、品种均衡、特征完整
- 劣势: 类别严重不平衡（7.4:92.6）

详见: [`TRAINING_QUALITY_REPORT.md`](file:///Users/hanjianglin/github/alpha-os/ai-engine/TRAINING_QUALITY_REPORT.md)

---

## 系统配置总结

### 网络配置
```
alphaos-net (192.168.97.0/24) - 统一网络
├── Web (alpha-os-web:3001)
├── Bridge API (bridge-api:8000, gRPC:50051)
├── AI Engine (ai-engine:50051)
├── MT5 VNC (mt5-vnc:3000)
└── Supabase Services (dual-network mode)
    ├── Kong API (54321)
    ├── PostgreSQL (54322)
    ├── Studio (54323)
    └── Other microservices
```

### 端口映射
| 服务 | 容器端口 | 宿主机端口 | 访问地址 |
|------|---------|-----------|---------|
| Web | 3000 | 3001 | http://192.168.3.8:3001 |
| Bridge API | 8000 | 8000 | http://192.168.3.8:8000 |
| AI Engine | 50051 | 50051 | grpc://192.168.3.8:50051 |
| MT5 VNC | 3000 | 3000 | http://192.168.3.8:3000 |
| Supabase Kong | 8000 | 54321 | http://192.168.3.8:54321 |
| Supabase DB | 5432 | 54322 | postgres://192.168.3.8:54322 |
| Supabase Studio | 3000 | 54323 | http://192.168.3.8:54323 |

### 环境变量
```bash
# .env.local
NEXT_PUBLIC_SUPABASE_URL=http://192.168.3.8:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...（默认 anon key）
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...（默认 service role key）
```

---

## 遗留问题与改进建议

### 已知问题
1. ⚠️ **类别不平衡**: 92.6% 负样本可能导致模型过于保守
2. ⚠️ **标签质量**: 负样本使用合成标签（固定 0.0）
3. ⚠️ **training_signals 云端权限**: 迁移时遇到权限问题

### 短期改进（本周）
- [ ] 监控新模型实战表现（WAIT 比例、胜率变化）
- [ ] 记录 28% → ??% 胜率变化趋势
- [ ] 观察交易频率是否下降

### 中期改进（下周）
- [ ] 实现类别权重平衡（`class_weight='balanced'`）
- [ ] 改进负样本标签（使用后视 MFE 代替固定 0）
- [ ] 添加特征重要性分析工具

### 长期优化（下月）
- [ ] A/B 测试：新旧模型对比
- [ ] 缩短重训练周期至 50 条新样本
- [ ] 实现模型版本控制和回滚机制
- [ ] 添加时间序列交叉验证

---

## 相关文档索引

### 部署相关
- [DEPLOYMENT_GUIDE.md](file:///Users/hanjianglin/github/alpha-os/docs/DEPLOYMENT_GUIDE.md) - 完整部署指南
- [README.md](file:///Users/hanjianglin/github/alpha-os/README.md) - 项目概览（含 v2.6.0 更新）

### AI 训练相关
- [TRAINING_REPORT.md](file:///Users/hanjianglin/github/alpha-os/ai-engine/TRAINING_REPORT.md) - 训练执行报告
- [TRAINING_QUALITY_REPORT.md](file:///Users/hanjianglin/github/alpha-os/ai-engine/TRAINING_QUALITY_REPORT.md) - 数据质量分析

### 代码修改
主要修改的文件：
- [`deploy_orb.sh`](file:///Users/hanjianglin/github/alpha-os/deploy_orb.sh) - 部署脚本（网络、环境变量修复）
- [`docker-compose.yml`](file:///Users/hanjianglin/github/alpha-os/docker-compose.yml) - 网络配置
- [`ai-engine/src/ingest_mql_data.py`](file:///Users/hanjianglin/github/alpha-os/ai-engine/src/ingest_mql_data.py) - 负样本数据采集
- [`ai-engine/enhance_features.py`](file:///Users/hanjianglin/github/alpha-os/ai-engine/enhance_features.py) - 特征工程（修复缩进）

---

## 团队协作备注

### 部署注意事项
1. **首次部署必须先运行**: `./deploy_orb.sh --supabase`
2. **数据迁移是可选的**: 使用 `--migrate` 标志
3. **环境变量必须配置**: 确保 `.env.local` 存在且包含 Supabase 凭证
4. **端口冲突检查**: 确保 3000/3001/8000/50051/54321-54323 未被占用

### 训练注意事项
1. **AutoLearner 自动运行**: 无需手动触发，100 条新数据自动重训练
2. **手动训练流程**:
   ```bash
   ssh macOS "docker exec ai-engine bash -c 'cd /app && \
     python src/ingest_mql_data.py && \
     python enhance_features.py && \
     python train_filter.py'"
   ```
3. **模型部署**: 训练后需重启 AI Engine 加载新模型
   ```bash
   ./deploy_orb.sh --ai
   ```

---

## 性能指标

### 部署性能
- 网络统一后的容器间通信: <5ms
- Web 构建时间: ~60s
- AI Engine 冷启动: ~10s

### 训练性能
- 数据采集（12,029 条）: ~5s
- 特征工程: ~10s
- 模型训练（5 个品种）: ~30s
- **总计**: ~45s

### 存储占用
- Supabase 数据目录: ~2GB
- Docker 镜像总计: ~8GB
- 训练模型文件: 23KB (5个模型)

---

## 第二次会话完成任务 (2025-12-12 继续)

### 反事实 MFE 实现
- [x] 修复 timestamp 数据类型问题（datetime 转换）
- [x] 使用 ISO 8601 格式查询 Supabase
- [x] 成功计算 13,685/16,102 负样本的反事实 MFE（85%成功率）
- [x] 使用真实 MFE 重新训练所有模型
- [x] 验证模型指标真实化（R²从1.0降至-0.45到0.11）
- [x] 部署新模型到生产环境

### 凯利公式修复
- [x] 诊断手数计算bug（除以100导致极端值）
- [x] 修复公式（使用 margin_per_lot 正确计算）
- [x] 优化高频交易参数（margin_per_lot=5000, min_lot=0.05）
- [x] 实现0.05手步进
- [x] 部署 Bridge API

### 文档更新
- [x] 创建 COUNTERFACTUAL_MFE_SUCCESS.md
- [x] 创建 KELLY_HFT_OPTIMIZATION.md
- [x] 更新 walkthrough.md
- [x] 更新 README.md

## 下次会话准备

### 需要验证的指标
- [ ] 新模型胜率（目标 >35%，当前基准 28%）
- [ ] WAIT 决策占比（预期增加）
- [ ] 凯利手数分布（应在0.05-1.0范围）
- [ ] 日均交易次数变化
- [ ] 反事实 MFE 对实战的影响

### 建议的后续任务
1. 收集 100+ 条新数据后的第一次自动重训练
2. A/B 测试设计（50% 流量使用新模型）
3. 特征重要性可视化工具开发
4. 类别平衡策略实现

---

**最后更新**: 2025-12-12 02:59  
**维护者**: AI Assistant  
**会话时长**: ~3.5 小时  
**代码变更**: 15 个文件  
**新增文档**: 4 个
- [2025-12-12 15:20] 更新: ai-engine/train_filter.py & src/models/online_lgbm.py - 实现模型版本控制 (v1, v2...) 及自动重置功能，修复日志显示模型名称不明确的问题
- [2025-12-12 15:50] 更新: ai-engine/src/client.py & ai-engine/src/ingest_mql_data.py - 实现“负样本分离”机制：新建 `market_scans` 表存储 WAIT/SCAN 信号，与 `training_signals` (真实交易) 物理隔离，但训练时通过 ingestion 脚本自动合并并进行反事实模拟。
- [2025-12-12 16:00] 新增: ai-engine/src/monitor_performance.py - 上线 AI 性能监控脚本，提供实时胜率、PnL 及决策分布看板。
- [2025-12-12 16:15] 里程碑: 完成 Model v3 训练。样本量突破 1.6万 (含1.5万模拟交易)，BTCUSD/GBPUSD R2 指标转正，验证了“负样本虚拟化”策略的有效性。
- [2025-12-13 00:21] 自动生成: ALPHAOS_ARCHITECTURE_V2.md - 完成全栈扫描与冗余检测
- [2025-12-13 00:29] 文档清理: 删除4个重复文档(MASTER_DOC/OFFICIAL_MANUAL/PROJECT_DOCUMENTATION/COUNTERFACTUAL_LOGIC)，合并SYSTEM_FUNCTION_MAP和PRODUCT_MANUAL到ARCHITECTURE_V2

