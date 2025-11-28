# AlphaOS 项目实施与状态报告

**日期**: 2025-11-28
**版本**: Phase 5 Complete (Distributed AI Architecture)

## 1. 项目概览

本项目旨在将 AlphaOS 从一个基础的交易监控工具升级为**机构级智能交易系统**。根据战略路线图，我们成功完成了前五个阶段（坚实基础、交易核心增强、数据智能、自动化执行、分布式 AI）的开发任务。核心架构从依赖外部 Webhook 转向了更可靠的 MT5 原生指标与 Python Bridge 的闭环系统，并最终实现了基于本地高性能算力的分布式 AI 推理架构。

**最新更新 (2025-11-28)**: 正式完成 **Phase 5: 分布式 AI 架构 (Distributed AI Architecture)** 的实施。我们构建了 "本地 M2 Pro 推理 + 云端 Bridge 转发" 的混合架构，并成功实现了 **3种核心交易模式**（经典、指标+AI、AI DOM），特别是引入了 **市场深度 (DOM)** 数据作为 AI 决策的关键维度。

### Phase 5.2: 剥头皮 AI 引擎 (Scalping AI Engine)
*核心目标: 针对超短线交易构建专用的高频特征工程与模型训练管线。*

*   **LightGBM 特征工程**:
    *   **微观结构**: 实现了 **OFI (Order Flow Imbalance)** 和 **Depth Ratio**，利用 DOM 数据捕捉瞬时买卖压力（主要决策依据）。
    *   **时段波动性**: 自动识别 Asian/London/NY 交易时段，捕捉特定市场的波动特征。
    *   **多维特征融合**: 将 EMA/RSI/MACD 等技术指标作为**市场状态上下文 (Context)** 输入 AI，辅助模型判断微观订单流信号的有效性，而非单一的硬性过滤。

*   **智能标签系统 (Triple Barrier)**:
    *   针对剥头皮策略定制了严格的标签逻辑：**TP 0.1% / SL 0.08%**，时间窗口 **5根K线**。
    *   模型输出不仅包含置信度，还自动计算 **动态止盈止损位**。

## 2. 已完成实施内容

### Phase 1: 坚实基础 (Foundation & Reliability)
*核心目标: 消除"玩具感"，建立金融级的稳定性。*

*   **状态管理重构 (Zustand Integration)**
    *   引入 `zustand` 替代 React Context。
    *   创建了三大核心 Store: `useMarketStore` (行情), `useTradeStore` (账户/持仓), `useUserStore` (配置)。
    *   **成效**: 显著减少了组件间的 Props Drilling，提升了渲染性能。

*   **系统稳定性增强**
    *   **MT5 Client**: 实现了指数退避重试 (Exponential Backoff) 和 Zod 类型验证，确保 Bridge 通信的健壮性。
    *   **Bridge Sync Hook**: 开发 `useBridgeSync`，统一管理心跳检测和状态同步。
    *   **错误边界**: 引入 `ErrorBoundary` 组件，防止局部错误导致整个应用崩溃。

*   **数据性能优化**
    *   **服务端分页**: 为 `Journal` 和 `RecentTrades` 实现了 API 级的分页逻辑。
    *   **虚拟滚动**: 前端引入 `react-virtuoso`，流畅处理大量交易记录的渲染。

*   **数据一致性保障**
    *   开发了 `/api/cron/consistency-check` 接口，定期比对 Supabase 数据库与 MT5 终端的持仓状态，自动识别"幽灵订单"或"遗漏订单"。

### Phase 2: 交易核心增强 (Advanced Trading Core)
*核心目标: 提供超越 MT5 原生的交易体验。*

*   **MQL5 指标与信号系统 (重大架构调整)**
    *   **PivotTrendSignals.mq5**: 成功将 Pine Script 逻辑移植为 MQL5 原生指标。
    *   **信号传输链路**: 指标生成 JSON 文件 -> Python Bridge 监听文件变化 -> 写入 Supabase -> 前端实时推送。
    *   **优势**: 彻底绕开了 TradingView Webhook 的订阅限制，实现了本地闭环，延迟更低，数据更安全。

*   **智能交易面板 (TradePanel 2.0)**
    *   **信号联动**: 点击实时通知 (`SignalListener`) 自动打开面板并预填 Symbol, Price, SL, TP。
    *   **风控计算器**: 集成风险计算功能，支持按风险金额或账户百分比自动计算手数 (Lots)。
    *   **高级订单**: 实现了 **OCO (One Cancels Other)** 订单逻辑和 **Auto SL/TP** (基于模拟 ATR) 功能。

*   **工程化 (DevOps)**
    *   配置了 GitHub Actions (`ci.yml`)，确保代码提交时的质量检查 (Linting & Type Checking)。

### Phase 3: 数据智能 (Data Intelligence)
*核心目标: 提供深度复盘与移动端体验优化。*

*   **移动端体验重构**
    *   **专注交易模式 (Focus Mode)**: K线图、市场监控、持仓管理黄金比例分割，移除干扰。
    *   **交互优化**: 移动端禁用拖拽，采用固定堆叠布局，优化图表缩放。
*   **工作区管理**: 实现了默认、分析、策略三种工作区预设，独立持久化。
*   **高级分析**:
    *   **执行分析 (MAE/MFE)**: 散点图展示交易偏移。
    *   **情绪热力图**: 可视化交易频率与情绪评分。
    *   **策略表现**: 按策略分类统计胜率。

### Phase 4: 自动化策略执行 (Automated Execution)
*核心目标: 实现基于规则的自动交易，减少人工干预。*

*   **自动化规则引擎**
    *   **数据库架构**: 创建 `automation_rules` 表，存储每个品种的自动交易配置（开关、手数、最大点差）。
    *   **Python Bridge 升级**: 实现 `AutomationManager` 类，自动同步 Supabase 规则，并在接收到信号时进行评估。
    *   **自动下单**: 满足规则（启用且点差合规）的信号将直接转换为交易指令推送到 MT5。

*   **前端配置界面**
    *   **AutomationRules 组件**: 在设置页面新增"自动化"标签页，允许用户添加、编辑、删除自动化规则。
    *   **风险提示**: 界面显著位置展示风险警告，确保用户知情。

### Phase 5: 分布式 AI 架构 (Distributed AI Architecture)
*核心目标: 构建混合算力架构，实现实时 AI 推理。*

*   **核心架构升级**
    *   **gRPC 双向流**: 引入 `gRPC` 协议，实现了云端 Bridge 与本地 M2 Pro 之间的毫秒级双向实时通讯。
    *   **Proto 契约**: 定义 `SignalRequest` 和 `SignalResponse` 消息格式，规范数据传输。
    *   **服务端 (Cloud)**: Bridge API 集成 gRPC Server，负责路由 MT5 信号到本地 AI 引擎。
    *   **客户端 (Local)**: 开发 `ai-engine` 客户端，运行在本地高性能环境，负责特征计算和模型推理。

*   **数据库与基础设施 (V2 Schema)**
    *   **全量整合**: 创建 `src/db/FULL_SCHEMA_V2.sql`，将分散的 SQL 文件合并为单一真理来源。
    *   **训练数据集**: 新增 `training_datasets` 表，存储 AI 推理时的市场快照 (Market Context)，为离线训练提供标注数据。
    *   **字段扩展**: `trades` 表新增 `mae`/`mfe` 字段；`automation_rules` 表新增 `ai_mode` (Legacy/Indicator+AI/Pure AI) 和 `confidence_threshold` 字段。
    *   **权限修复**: 全面修复 Supabase RLS 策略，确保 API 和前端的读写权限。

*   **前端增强**
    *   **AI 模式切换**: 在自动化规则中增加 AI 模式选择。
    *   **决策透明化**: 交易通知 (Toast) 实时显示 AI 的决策结果（推荐/拒绝）及置信度。

### Phase 5.1: 多模式交易与 DOM 集成 (Multi-Mode Trading & DOM)
*核心目标: 提供灵活的交易控制，并引入微观市场结构数据。*

*   **3种交易模式**:
    1.  **经典模式 (Legacy)**: 纯指标驱动，信号直接执行，适合趋势明确的行情。
    2.  **前端自动化：自动 + AI 过滤 (Indicator + AI)**: 指标生成信号，AI 基于 K 线历史 (Price Action) 进行二次确认，过滤假突破。此模式通过前端"自动化规则"配置，实现自动跟单。
    3.  **AI (DOM) 模式**: 在 K 线数据基础上，引入 **Level 2 市场深度 (DOM)** 数据（买卖盘口分布、挂单量），捕捉微观流动性变化。

*   **全链路 DOM 支持**:
    *   **Proto**: 扩展 `SignalRequest` 增加 `dom_bids` / `dom_asks` 结构。
    *   **MQL5**: 实现 `GET_DOM` 指令，通过 `MarketBookGet` 获取实时盘口数据。
    *   **Python Bridge**: 升级 `AutomationManager`，在 `dom_ai` 模式下自动请求 DOM 数据并打包发送给 AI 引擎。
    *   **Frontend**: 自动化规则配置界面新增 `dom_ai` 选项。

## 3. 移动端与UX 增强

### 3.1 移动端体验重构
针对移动端触摸操作的特点，进行了深度的交互优化：

*   **专注交易模式 (Focus Mode)**:
    *   **视觉升级**: 采用深邃的渐变背景和玻璃拟态 Header，提供沉浸式体验。
    *   **布局重构**: K线图 (38%)、市场监控 (Flex)、持仓管理 (22%) 黄金比例分割，操作更顺手。
    *   **功能**: 移除一切干扰，仅保留核心交易组件。
*   **图表与交互优化**:
    *   **Lightweight Charts 调优**: 修复了移动端缩放导致指标挤压的问题；调整了默认视口，使最新 K 线停留在屏幕右侧 3/4 处，符合专业看盘习惯。
    *   **性能提升**: 优化了图表滚动事件监听，解决了本地运行时的卡顿问题。

### 3.2 信号历史管理
*   **Signal Store**: 创建了 `useSignalStore` 进行全局信号状态管理。
*   **历史面板**: 新增 `SignalHistory` 组件，侧滑展示历史信号列表。
*   **联动优化**: 点击历史信号可直接调出交易面板。

### 3.3 工作区管理 (Workspace)
*   **多场景预设**: 实现了 **默认**、**分析**、**策略** 三种工作区预设，一键切换布局。
*   **独立持久化**: 每个工作区的布局调整都会独立保存，互不干扰。

## 4. 数据智能 (Phase 3 Preview)

全新上线的 `/analytics` 模块，提供深度复盘能力：

*   **执行分析 (MAE/MFE)**: `MaeMfeScatterChart` 组件，通过散点图直观展示每笔交易的最大不利偏移（MAE）和最大有利偏移（MFE），辅助优化止盈止损设置。
*   **情绪热力图**: `SentimentHeatmap` 组件，可视化展示过去 30 天的交易频率与情绪评分，帮助发现情绪波动对交易的影响。
*   **策略表现**: `StrategyBreakdown` 组件，按策略分类统计胜率和盈亏贡献。

## 5. 信号系统部署详情

### 5.1 系统架构

```
MT5 指标 (PivotTrendSignals.mq5)
    │
    ▼ 生成 JSON 信号文件
Docker 共享卷 (signal_data)
    │
    ▼ watch_signal_directory() 每 500ms 扫描
Python Bridge API
    │
    ▼ 插入数据库
Supabase (signals 表)
    │
    ▼ Realtime 推送
前端 SignalListener (Store)
    │
    ▼ Toast 通知 + 写入 SignalHistory
    │
    ▼ 用户点击 -> 打开 TradePanel
```

### 5.2 关键配置

**Docker 卷映射** (`docker-compose.yml`):
```yaml
services:
  mt5:
    volumes:
      - signal_data:/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals
  
  bridge-api:
    environment:
      - SUPABASE_URL=${NEXT_PUBLIC_SUPABASE_URL}
      - SUPABASE_KEY=${NEXT_PUBLIC_SUPABASE_ANON_KEY}
      - SIGNAL_DIR=/app/signals
    volumes:
      - signal_data:/app/signals

volumes:
  signal_data:
```

**权限配置** (必须):
```bash
docker exec mt5-vnc chmod 777 "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/AlphaOS/Signals/"
```

## 6. 遇到的问题与解决方案 (Troubleshooting Log)

| 问题分类 | 问题描述 | 解决方案 |
| :--- | :--- | :--- |
| **架构限制** | 用户无 TradingView 会员，无法使用 Webhook 推送信号。 | **方案转型**: 开发 MQL5 原生指标，配合 Python 文件监听器，实现本地信号闭环。 |
| **Docker 卷共享** | MT5 和 Bridge 容器无法共享信号文件。 | **共享卷配置**: 创建 `signal_data` 卷，分别挂载到两个容器的对应目录。 |
| **MT5 回测数据采集** | 运行回测时图表不显示指标，输出目录为空。 | **模式修正**: 必须将采集器 (`PivotTrend_DataCollector.mq5`) 作为 **Expert Advisor** 而非 Indicator 运行；**关闭优化 (Optimization: Disabled)** 以防止文件写入临时目录；逻辑改为 **Signal-First** (信号即写，平仓追加)，防止数据丢失。 |
| **数据导出** | 无法从 Docker 容器内批量提取分散的 JSON 训练数据。 | **ETL 流程**: 编写 `tar` 命令打包容器内 `Tester/Agent` 目录数据，配合 `export_training_data.py` 脚本自动清洗并合并为 CSV 格式。 |
| **文件权限** | MT5 进程 (abc 用户) 无法写入 root 拥有的目录。 | **权限修复**: `chmod 777` 信号目录，确保 MT5 可写入。 |
| **MT5 安装失败** | `c0000135` 错误或 `rosetta error` (Mac M1/M2)。 | **平台兼容性**: 放弃 Docker 内运行 MT5 (x86 Wine 兼容性差)，改为 **宿主机原生运行 MT5 + 本地 Python Bridge** 方案，极其稳定。 |
| **AI 架构** | 低内存云服务器无法运行 LLM/LightGBM 推理。 | **分布式改造**: 采用 gRPC 架构，将推理卸载到本地 M2 Pro，云端仅做转发。 |
| **数据库Schema** | 表结构分散，缺少版本控制，导致部署不一致。 | **统一管理**: 整合为 `FULL_SCHEMA_V2.sql`，作为单一真理来源。 |

## 7. 文件清单

### Phase 5.2 新增文件 (Scalping AI)
| 文件路径 | 说明 |
|----------|------|
| `ai-engine/src/features.py` | **New** 剥头皮专用特征工程 (DOM/Tech/Time) |
| `ai-engine/src/train.py` | **New** LightGBM 训练管线 (Triple Barrier Labeling) |
| `ai-engine/models/` | **New** 模型存储目录 |
| `trading-bridge/mql5/PivotTrend_DataCollector.mq5` | **New** 回测专用数据采集指标 (EA模式) |
| `ai-engine/export_training_data.py` | **New** 训练数据导出工具 |
| `src/db/data_collection.sql` | **New** 训练数据表 Schema 定义 |
| `docs/DATA_COLLECTION_GUIDE.md` | **New** 数据采集操作指南 |

### Phase 5 新增文件 (Distributed AI)
| 文件路径 | 说明 |
|----------|------|
| `src/proto/alphaos.proto` | **New** gRPC 通信协议定义 |
| `src/db/FULL_SCHEMA_V2.sql` | **New** 数据库全量 Schema (包含 AI 表) |
| `ai-engine/src/client.py` | **New** 本地 AI 客户端 (gRPC Client) |
| `ai-engine/run_ai.sh` | **New** AI 引擎启动脚本 |
| `trading-bridge/src/grpc_server.py` | **New** 云端 gRPC 服务端 |
| `trading-bridge/docker/generate_proto.sh` | **New** Proto 代码生成脚本 |

### 此前新增文件 (Phase 1-4)
| 文件路径 | 说明 |
|----------|------|
| `src/app/analytics/page.tsx` | **New** 数据智能分析页面 |
| `src/components/charts/MaeMfeScatterChart.tsx` | **New** MAE/MFE 散点图组件 |
| `src/components/SentimentHeatmap.tsx` | **New** 情绪热力图组件 |
| `src/components/StrategyBreakdown.tsx` | **New** 策略分析组件 |
| `src/store/useSignalStore.ts` | 信号状态管理 |
| `src/components/SignalHistory.tsx` | 信号历史侧边栏 |
| `src/components/MarketSessions.tsx` | 市场时段可视化组件 |
| `src/store/useMarketStore.ts` | 行情数据 Zustand Store |
| `src/store/useTradeStore.ts` | 交易数据 Zustand Store |
| `src/store/useUserStore.ts` | 用户配置 Zustand Store |
| `src/hooks/useBridgeSync.ts` | Bridge 同步 Hook |
| `src/components/SignalListener.tsx` | 信号监听组件 |
| `src/components/ErrorBoundary.tsx` | 错误边界组件 |
| `src/db/signals_table.sql` | 信号表 SQL 定义 |
| `src/app/api/cron/consistency-check/route.ts` | 数据一致性检查 API |
| `trading-bridge/mql5/PivotTrendSignals.mq5` | MQL5 信号指标 |
| `.github/workflows/ci.yml` | CI/CD 配置 |
| `docs/SIGNAL_SYSTEM.md` | 信号系统文档 |

### 修改文件
| 文件路径 | 修改内容 |
|----------|----------|
| `src/app/dashboard/page.tsx` | 集成工作区切换、专注模式、市场时段组件 |
| `src/components/charts/TradingViewChart.tsx` | 修复缩放挤压、指标渲染、性能优化 |
| `src/components/charts/plugins/CloudSeries.ts` | 修复指标连线伪影 |
| `src/components/TradePanel.tsx` | 添加风控计算器、OCO、Auto SL/TP、信号预填 |
| `src/components/AppShell.tsx` | 集成 SignalListener，添加铃铛图标联动 |
| `src/hooks/useBridgeStatus.ts` | 改为更新 Zustand Store |
| `trading-bridge/src/main.py` | 添加 gRPC Server、信号文件监听器、AI 自动化逻辑 |
| `trading-bridge/docker/docker-compose.yml` | 添加 gRPC 端口映射、共享卷 |

## 8. 下一步计划 (Data & Training)

我们已完成架构搭建，接下来的重点将转向**数据积累与模型迭代**：

1.  **数据积累**: 保持系统运行 1-2 周，收集覆盖不同市场状态（趋势/震荡）的样本数据。
2.  **模型训练**: 开发本地 Python 脚本，从 Supabase 拉取数据，训练 LightGBM 模型。
3.  **模型热更新**: 实现模型文件的动态加载机制。

## 9. 优化与深化方向 (Optimization & Deepening)

为了进一步提升系统的实战能力，我们计划在以下几个方向进行深化：

### 9.1 模型与算法深化
*   **真实模型训练**: 当前 AI 引擎使用模拟逻辑。下一步需收集真实数据，训练 **LightGBM** 或 **XGBoost** 模型，重点识别假突破和流动性陷阱。
*   **特征工程扩展**:
    *   **订单流不平衡 (OFI)**: 基于 DOM 数据计算买卖压力差。
    *   **微观结构特征**: 盘口厚度、大单挂单位置、成交速率。
    *   **多模型融合**: 针对不同品种（如黄金 vs 外汇）训练专用模型，或采用 Ensemble Learning 提升鲁棒性。

### 9.2 性能与延迟优化
*   **通信协议升级**: 目前 Bridge 与 MT5 采用 HTTP 轮询 (Polling)，延迟在 500ms-1s。计划升级为 **ZeroMQ** 或 **Named Pipes**，实现毫秒级双向推送。
*   **本地化部署**: 探索将 Python Bridge 直接部署在 Windows (MT5 宿主机) 上，减少网络跳数。

### 9.3 风险管理智能化
*   **动态仓位管理**: 基于 AI 置信度 (Confidence Score) 动态调整开仓手数。高置信度重仓，低置信度轻仓或观望。
*   **智能止损**: 利用 ATR 和波动率预测，动态设置止损位，而非固定点数。
