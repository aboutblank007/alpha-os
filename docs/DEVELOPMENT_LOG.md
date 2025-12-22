# AlphaOS 开发备忘录

**项目**: AlphaOS - MT5 智能交易管理系统  
**维护周期**: 2025-11 至今

---

## 2025-12-19 - AI 模型重训与架构修复

### DQN 模型修复
- [2025-12-19 11:46] [AI]: `ai-engine/src/train_dqn_from_db.py` - 新增训练脚本，支持 CSV 数据源（supabase_xauusd_actions.csv，6870条记录）
- [2025-12-19 11:46] [AI]: `ai-engine/src/models/dqn.py` - DuelingDQN 架构更新，匹配训练脚本（128维隐藏层+LayerNorm+Dropout）
- [2025-12-19 19:46] [AI]: DQN 模型重训完成（30维输入，6870样本，150轮，Best Loss=0.089），部署到 `/app/models/dqn_xauusd.pth`

### QuantumNet 模型修复  
- [2025-12-19 12:14] [AI]: `ai-engine/train_quantum_from_db.py` - 新增 CSV 数据源支持，调整 class_weights=[4.0, 0.5, 3.5] 以平衡类别
- [2025-12-19 13:20] [AI]: QuantumNet 重训完成（val_acc=52.12%，WAIT 概率从 50%→27%，BUY/SELL 概率从 25%→37%）

### 信号计算逻辑修复
- [2025-12-19 21:30] [AI]: `ai-engine/src/client.py` - DQN input_dim 从 34 改为 30，匹配训练模型特征数
- [2025-12-19 21:30] [AI]: `ai-engine/src/client.py` - dqn_features 列表更新为 30 个特征（移除 log_return, volatility_5 等）
- [2025-12-19 21:30] [AI]: `ai-engine/src/client.py` - s_d 计算改用 Softmax 概率差（BUY_prob - SELL_prob），避免 DQN 选 WAIT 时 s_d=0 导致 Score=0
- [2025-12-19 21:32] [AI]: `ai-engine/src/client.py` - State Shape 检查从 (34,) 改为 (30,)
- [2025-12-19 21:35] [AI]: 新增 DQN Softmax 日志输出，便于追踪概率分布

### 四模型集成架构确认
- QuantumNet (CNN+LSTM+Attention, 33维输入): 趋势识别，输出 BUY/SELL/WAIT 概率
- DQN Agent (DuelingDQN, 30维输入): 强化学习决策，输出 Q-values
- CatBoost Meta (梯度提升树): 信号质量门控，过滤低质量信号
- ARIMA-GARCH (在线时序模型): 波动预测，无需预训练

## 2025-12-18 - AI Engine XAUUSD 专用化

- [2025-12-18 00:10] [AI]: `ai-engine/src/client.py` - 仅加载 XAUUSD CatBoost 与 DQN 权重（catboost_xauusd.cbm / dqn_xauusd.pth），非 XAUUSD 请求强制 WAIT；默认信心阈值调整为 XAU 专用 0.15
- [2025-12-18 00:20] [AI]: `ai-engine/src/client.py` - 修正 CatBoost `predict_proba` 读取正类概率（以前误取第一列导致 meta_prob≈0，全局 WAIT）
- [2025-12-18 00:35] [AI]: `ai-engine/src/client.py` - 暂停 Meta 门控（meta_multiplier=1.0）以排查全 WAIT，直接使用 primary_score
- [2025-12-18 00:50] [AI]: `ai-engine/src/client.py` - 增加模型详细诊断日志（q_policy/DQN/TS/primary_score/阈值、波动率 clip 前后）便于追踪 RawMult=6 与 Score 稳定原因
- [2025-12-18 01:00] [AI]: `ai-engine/src/client.py` - 重新开启 Meta 门控（0.45/0.55/0.7 阶梯乘子）恢复过滤逻辑
- [2025-12-18 01:05] [AI]: `ai-engine/src/client.py` - 补充 QuantumNet 策略+价值日志
- [2025-12-18 01:15] [AI]: `ai-engine/src/client.py` - 修正 meta_prob 提取（CatBoost ndarray 取正类概率，避免 float() Deprecation & 误差）
- [2025-12-18 02:00] [AI]: 新增 `ai-engine/src/exit_optimizer.py` - 1m 剥头皮离场参数搜索（网格/可选贝叶斯），支持分级TP/跟踪止盈/时间止盈/信号退化阈值的重放评估
- [2025-12-18 02:30] [Bridge]: 新增 `trading-bridge/src/managers/exit_manager.py` 并在 `src/main.py` 接入，基于持仓实时回撤（0.8R 启动、回撤0.4R全量平仓）自动下发 CLOSE 指令
- [2025-12-18 03:00] [AI]: `ai-engine/src/client.py` - 修复 DQN 推理未做标准化，加载 `dqn_scaler_xauusd.npy` 应用 mean/std 以消除方向偏置（此前训练阶段已标准化）
- [2025-12-18 03:30] [AI]: 重新训练 `catboost_xauusd.cbm`（数据：training_signals_rows.csv，目标 result_profit>0，特征来自 ai_features JSON + 核心技术指标），AUC≈0.66；已覆盖模型文件待部署
- [2025-12-18 23:40] [AI]: `ai-engine/src/client.py` - DQN scaler 防御：加载后清洗 NaN/零方差（std<1e-6→1），推理阶段再 NaN to 0，避免 DQN 失配导致方向异常
- [2025-12-19 00:05] [AI]: 重训 `catboost_xauusd.cbm`（过滤 result_profit 非空的 XAUUSD 样本，样本数544，class_weights 平衡；val AUC≈0.61，val proba 均值≈0.48，防止 meta_prob 过低归零）

## 2025-12-16 - Terminal V2 前端重做（骨架稳定 + 内容自适应）

- [2025-12-16 23:45] [Frontend]: `src/app/layout.tsx` - 将 `lang` 统一为 `zh-CN`
- [2025-12-16 23:45] [Frontend]: `src/components/AppShell.tsx` - 壳层去业务化（移除顶部业务 Header），Dashboard 使用 full-bleed
- [2025-12-16 23:46] [Frontend]: `src/lib/terminal-v2/featureParity.ts` - 冻结“现有功能不丢”契约清单
- [2025-12-16 23:46] [Frontend]: `src/components/terminal/TerminalLayout.tsx` - 新增 Terminal V2 三段式骨架（Main/Side/Bottom）
- [2025-12-16 23:46] [Frontend]: `src/components/terminal/TerminalZone.tsx` - 新增 Zone 容器（标题/折叠/右侧动作/风险 tone）
- [2025-12-16 23:46] [Frontend]: `src/components/terminal/WidgetFrame.tsx` - 新增统一 Widget 外壳
- [2025-12-16 23:47] [Frontend]: `src/components/terminal/ZoneDndList.tsx` - 新增区内拖拽排序组件（强模板，不跨区）
- [2025-12-16 23:48] [Frontend]: `src/app/dashboard/page.tsx` - 迁移到 Terminal V2 骨架；实现 orders_auto（少仓主区/重仓或风险通栏 + 手动全宽）；侧栏区内拖拽排序持久化；Focus Mode V2 仅切换主区呈现

- [2025-12-17 21:00] [Deploy]: `deploy_orb.sh` - AI Engine `/app/models` 从 docker volume 改为宿主机 `ai-engine/models` bind mount（避免 OrbStack NFS 目录超时）；部署时自愈 `ai_decisions.csv` 目录/文件类型

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
- [2025-12-13 13:00] [Code]: ai-engine/src/client.py - Pivot Model v4 to Binary Classification (Precision Focus).
- [2025-12-13 13:45] [Task]: Phase 8 Offline RL - 训练 2000 Episodes DQN (System 2) 完成。
- [2025-12-13 13:55] [Code]: ai-engine/src/client.py - 集成 DQN Agent (34 Features), 修复 Data Leakage, 验证 Docker 部署成功。
- [2025-12-13 14:00] [Doc]: docs/ALPHAOS_ARCHITECTURE_V2.md - 同步 v4 双脑架构 (System 1/2)、Macro/Micro 特征及 Offline RL 流程。
- [2025-12-12 15:50] 更新: ai-engine/src/client.py & ai-engine/src/ingest_mql_data.py - 实现“负样本分离”机制：新建 `market_scans` 表存储 WAIT/SCAN 信号，与 `training_signals` (真实交易) 物理隔离，但训练时通过 ingestion 脚本自动合并并进行反事实模拟。
- [2025-12-12 16:00] 新增: ai-engine/src/monitor_performance.py - 上线 AI 性能监控脚本，提供实时胜率、PnL 及决策分布看板。
- [2025-12-12 16:15] 里程碑: 完成 Model v3 训练。样本量突破 1.6万 (含1.5万模拟交易)，BTCUSD/GBPUSD R2 指标转正，验证了“负样本虚拟化”策略的有效性。
- [2025-12-13 00:21] 自动生成: ALPHAOS_ARCHITECTURE_V2.md - 完成全栈扫描与冗余检测
- [2025-12-13 00:29] 文档清理: 删除4个重复文档(MASTER_DOC/OFFICIAL_MANUAL/PROJECT_DOCUMENTATION/COUNTERFACTUAL_LOGIC)，合并SYSTEM_FUNCTION_MAP和PRODUCT_MANUAL到ARCHITECTURE_V2

- [2025-12-13 01:09] 维护: GitHub Push - [2025-12-12 10:00] [Code]: src/main.py - 修复 gRPC 超时参数
- [2025-12-13 04:00] [Protocol]: docs/ANTIGRAVITY_WORKFLOW_SPEC.md - Established "Medallion" 5-Phase Workflow.
- [2025-12-13 12:50] [Model]: Model v4 "Golden Setup" Implemented. Pivoted to Binary Classification (MFE>2R & MAE<1R). Verified 0% Precision (Imbalance Limit).
- [2025-12-14 23:55] [Refactor]: src/env.ts & src/lib/mt5-client.ts - 统一后端配置，解决 Docker 内部主机名验证问题。
  - 放宽 `TRADING_BRIDGE_API_URL` 验证以支持 `http://bridge-api:8000`。
  - 所有 API 路由 (`status/route.ts`, `execute/route.ts`) 及 Client 统一引用 `env.ts` 配置。

- [2025-12-16 19:58] [UI]: src/app/dashboard/page.tsx - 主控制台升级为 Global Adaptive（Trading 默认态、优先级驱动重排、桌面/移动 Focus、Pin/快捷键）
- [2025-12-16 19:58] [Code]: src/lib/terminal-adaptive.ts - 新增风险/信号/设备上下文计算与布局优先级引擎（输出 effectiveLayout 与 Focus 建议）
- [2025-12-16 19:58] [Code]: src/hooks/useAdaptiveDashboard.ts - 新增自适应 Hook（useSyncExternalStore 驱动 localStorage 同步，避免 setState-in-effect）
- [2025-12-16 19:58] [UI]: src/components/dashboard/DashboardHeader.tsx - 增强为态势条（风险态/连接态/模式切换/Focus 入口）
- [2025-12-16 19:58] [UI]: src/components/dashboard/DesktopLayout.tsx & src/components/dashboard/SortableItem.tsx - 支持动态列数与 Pin（不参与自动重排）
- [2025-12-16 19:58] [UI]: src/components/dashboard/MobileLayout.tsx - FocusMode 支持 Trade/Risk 双焦点，并展示触发原因
- [2025-12-16 19:58] [UI]: src/components/RiskAlerts.tsx - 风险预警升级为流程入口（riskState/reasons/关键指标/进入 Risk Focus + 动作占位）
- [2025-12-16 19:58] [Config]: src/store/useSettingsStore.ts & src/app/settings/page.tsx - 新增本地角色/信息密度/默认模式配置
- [2025-12-16 19:58] [Fix]: src/components/AiMarketMonitor.tsx - 修复渲染期 Date.now 纯度问题（用 now 状态驱动）
- [2025-12-16 19:58] [Fix]: src/app/api/ai/settings/route.ts & src/app/api/ai/stats/route.ts - 移除 catch any，统一 unknown 错误处理
- [2025-12-16 19:58] [Fix]: src/components/charts/TradingViewChart.tsx & src/components/settings/AutomationRules.tsx & src/components/analytics/AiPerformance.tsx - 清理 any / JSX 转义，确保 eslint 无 error
- [2025-12-16 20:15] [Fix]: src/components/settings/AutomationRules.tsx - 修复 ai_mode onChange 类型收窄（保证 next build TS 编译通过）
- [2025-12-16 20:15] [Fix]: src/app/ui/page.tsx & src/components/settings/AutomationRules.tsx - 修复错误的 `@/components/Card` 引用，统一使用 `@/components/ui/Card`，确保 next build 可解析模块
- [2025-12-16 20:39] [Fix]: ai-engine/src/client.py - Relaxed Stalemate filter (ADX<5) and Confidence Thresholds (0.2) to enable trading on low vol/score setups; Synced to remote host.
- [2025-12-16 21:30] [Fix]: src/hooks/useAdaptiveDashboard.ts - Refactor zustand selectors to avoid returning new objects（修复 useSyncExternalStore getServerSnapshot 警告/避免 Dashboard 无限更新）
- [2025-12-16 21:30] [Fix]: src/components/EquityCurve.tsx - 禁用 recharts 动画并稳定尺寸计算，避免 ChartDataContextProvider/JavascriptAnimate 触发 React #185
- [2025-12-16 21:30] [Fix]: src/app/dashboard/page.tsx - 空交易时确保权益曲线至少 1 个点，避免图表空数据边界
- [2025-12-16 22:05] [UI]: src/components/dashboard/DesktopLayout.tsx & src/app/dashboard/page.tsx - 修复 Dashboard 网格跨度：将 col-span/min-h 等布局类应用到 SortableItem（Grid Item）而非 widget 内层，改善超宽屏排版
- [2025-12-16 22:15] [UI]: src/components/dashboard/DesktopLayout.tsx & src/components/dashboard/DashboardHeader.tsx & src/app/dashboard/page.tsx - Dashboard/Header/FocusOverlay 改为 full-bleed（w-full + 自适应 padding），超宽屏左右自适应并尽量完整呈现模块
- [2025-12-16 22:25] [UI]: src/hooks/useAdaptiveDashboard.ts - 超宽屏/非 compact 密度自动补齐模块（alerts/marketWatch/sessions/symbols/analytics），避免 6 列布局出现空洞，尽量完整显示内容
- [2025-12-16 22:40] [UI]: src/components/OngoingOrders.tsx & src/app/dashboard/page.tsx - MT5 持仓模块取消横向滚动（响应式隐藏次要列+table-fixed），并在超宽屏将 orders 扩宽为 2 列以提升可读性
- [2025-12-16 22:55] [UI]: src/lib/terminal-adaptive.ts & src/components/dashboard/DesktopLayout.tsx - Dashboard 对齐“数据分析页”布局策略：超宽屏优先保证卡片可读宽度（1920=4列，2560+=6列）+ grid-flow-dense 自动填洞，避免右侧大空档

- [2025-12-20 21:22] [Data]: `scripts/align_quantumnet_target.py` - 新增 QuantumNet 标签时序对齐脚本（`target_next_close_change = Close[t+1]-Close[t]`，按 symbol+timestamp 对齐，删除每组尾行）
- [2025-12-20 21:22] [Data]: `QuantumNet_Training_Data.aligned.csv` - 生成对齐后的训练数据文件（next-close 标签）

- [2025-12-20 22:48] [AI]: `legacy_backup/ai-engine-backup-20251220-222811.tgz` - 备份现有 ai-engine 框架（为“另起炉灶”的真量子引擎保留可恢复快照）
- [2025-12-20 22:48] [AI]: `quantum-engine/` - 新增独立真量子回归引擎（PennyLane + PQC 回归，输出 y_hat=target_next_close_change，不依赖旧 ai-engine 框架）
- [2025-12-20 22:48] [AI]: `quantum-engine/src/train_quantum_regressor.py` - 真量子回归训练脚本（标准化+PCA->AngleEmbedding，lightning.adjoint，早停）
- [2025-12-20 22:48] [AI]: `quantum-engine/src/infer_quantum_regressor.py` - 真量子回归推理脚本（输入 row-json，输出 y_hat）
- [2025-12-20 22:48] [AI]: `quantum-engine/requirements.txt` & `quantum-engine/scripts/setup_venv.sh` - 独立依赖与环境安装脚本（锁定 pennylane/autoray 兼容版本）

- [2025-12-21 10:30] [AI]: `quantum-engine/src/train_quantum_regressor.py` - 方案 C 优化：float64精度、QuantumFeatureTransformer 分特征定制预处理到 [-π,π]、12 qubits（M2 Pro 甜点区）、TargetScaler [-0.9,0.9]
- [2025-12-21 10:30] [AI]: `quantum-engine/src/infer_quantum_regressor.py` - 方案 C 同步：float64、QuantumFeatureTransformer 加载、TargetScaler 逆变换
- [2025-12-21 10:30] [AI]: `quantum-engine/scripts/setup_venv.sh` - 方案 C OMP 线程亲和性配置（OMP_NUM_THREADS=8 KMP_BLOCKTIME=0）
- [2025-12-21 10:30] [AI]: `quantum-engine/scripts/activate_quantum.sh` - 新增方案 C 激活脚本（自动设置 OMP/MKL 环境变量）

- [2025-12-21 13:36] [Data]: `quantum-engine/scripts/add_residuals.py` - 新增离线 Residual 标注脚本（批量推理 + 计算 residual = actual - predicted），用于 Meta-Labeling 模型训练
- [2025-12-21 13:36] [EA]: `trading-bridge/mql5/QuantumNet_Data_Miner_v2.mq5` - 新增 EA v2，扩展微观状态字段（spread、tick_rate、bid_ask_imbalance），用于 Meta-Labeling 数据采集

- [2025-12-21 14:16] [AI]: `quantum-engine/scripts/train_meta_labeling.py` - 新增 Meta-Labeling XGBoost 训练脚本（二分类：判断 QNN 信号是否值得下注）
- [2025-12-21 14:16] [AI]: `quantum-engine/models/meta_labeling_xgb.json` - 训练完成 Meta-Labeling 模型（验证集 Precision=83.62%, AUC=96.50%，Top 特征：qnn_prediction 44.8%、bid_ask_imbalance 19.0%）

- [2025-12-21 14:46] [Doc]: `docs/MT5 交易系统生产级落地方案.MD` - 整理全新 Q-Link 量子交易系统架构文档（侧车模式、ZeroMQ 中间件、三道防线风控）

- [2025-12-21 14:48] [Code]: `quantum-engine/qlink/protocol.py` - Q-Link 协议定义（端口配置、TickData/OrderCommand/AccountState/AlphaSignal/RiskDecision 数据结构）
- [2025-12-21 14:48] [Code]: `quantum-engine/qlink/alpha_engine.py` - Alpha 引擎（ZMQ PULL、量子推理服务、float64 精度）
- [2025-12-21 14:48] [Code]: `quantum-engine/qlink/risk_engine.py` - 风控引擎（三道防线：Meta-Labeling、波动率目标+凯利公式、L-VaR）
- [2025-12-21 14:48] [EA]: `trading-bridge/mql5/QuantumLink_EA.mq5` - MT5 执行端（非阻塞 ZMQ、特征计算、死人开关）
- [2025-12-21 14:48] [Script]: `quantum-engine/qlink/launch.sh` - M2 Pro 核心绑定启动脚本（P-Cores/E-Cores 分离）
## 开发日志更新

- [2025-12-21 21:40] [FEATURE]: src/types/quantum.d.ts - 创建量子 HFT 类型定义
- [2025-12-21 21:40] [FEATURE]: src/workers/marketDataParser.ts - 创建 Web Worker 用于高频 CSV 解析
- [2025-12-21 21:40] [FEATURE]: src/store/useQuantumStore.ts - 创建量子 AI 遥测状态管理
- [2025-12-21 21:40] [MODIFY]: src/store/useMarketStore.ts - 添加 subscribeWithSelector 瞬态更新能力
- [2025-12-21 21:40] [FEATURE]: src/hooks/useQuantumSocket.ts - 创建 WebSocket 连接 Hook
- [2025-12-21 21:40] [FEATURE]: src/components/dashboard/LatencyDisplay.tsx - 创建瞬态延迟显示组件
- [2025-12-21 21:40] [FEATURE]: src/components/dashboard/SystemVitals.tsx - 创建系统生命体征组件
- [2025-12-21 21:40] [FEATURE]: src/components/dashboard/RiskGauge.tsx - 创建凯利公式风险仪表
- [2025-12-21 21:40] [FEATURE]: src/components/dashboard/PanicButton.tsx - 创建紧急平仓按钮
- [2025-12-21 21:40] [FEATURE]: src/components/charts/QuantumCandleChart.tsx - 创建量子 K 线图 (影线染色)
- [2025-12-21 21:40] [FEATURE]: quantum-engine/qlink/api_gateway.py - 创建 FastAPI WebSocket 网关

- [2025-12-21 21:50] [MODIFY]: src/app/(main)/dashboard/page.tsx - 集成量子 HFT 组件到 Dashboard
- [2025-12-21 21:50] [FIX]: src/types/quantum.d.ts → quantum.ts - 修复模块解析问题

- [2025-12-21 22:00] [MODIFY]: quantum-engine/qlink/protocol.py - 统一 ZMQ 端口为 5557/5558/5559
- [2025-12-21 22:00] [MODIFY]: quantum-engine/qlink/api_gateway.py - 配置连接远程 MT5 VM (192.168.3.10)

- [2025-12-22 01:05] [NEW]: quantum-engine/data/btcusd_metalabel.csv - 生成 BTCUSD MetaLabel 数据 (495,363 行)
- [2025-12-22 01:05] [NEW]: quantum-engine/models/btc/meta_labeling_btc_xgb.json - 训练 BTCUSD Meta-Labeling 模型 (F1=0.89, AUC=0.98)

- [2025-12-22 01:10] [MODIFY]: quantum-engine/models/ - 重组目录结构，XAU 模型移动到 models/xau/
- [2025-12-22 01:10] [MODIFY]: quantum-engine/qlink/launch.sh - 支持多品种并行处理 (XAUUSD + BTCUSD)

- [2025-12-22 10:00] [DOCS]: `docs/交易风控离场策略研究.md` - Aligned documentation with XGBoost Meta-Labeling implementation and dual-level defense.
- [2025-12-22 10:00] [FIX]: `quantum-engine/qlink/protocol.py` - Updated `META_THRESHOLD` to 0.5 to balance XGBoost signal filtering.
