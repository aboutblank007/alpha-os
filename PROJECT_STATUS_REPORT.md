# AlphaOS 项目实施与状态报告

**日期**: 2024-11-26
**版本**: Phase 4 Preview (Automated Strategy Execution)

## 1. 项目概览

本项目旨在将 AlphaOS 从一个基础的交易监控工具升级为**机构级智能交易系统**。根据战略路线图，我们成功完成了前三个阶段（坚实基础、交易核心增强、数据智能）的开发任务。核心架构从依赖外部 Webhook 转向了更可靠的 MT5 原生指标与 Python Bridge 的闭环系统。

**最新更新 (2024-11-26)**: 正式启动 **Phase 4: 自动化策略执行 (Automated Execution)** 预览，实现了基于 MQL5 信号的自动下单功能。用户可以在前端配置每个品种的自动化规则（手数、最大点差），Python Bridge 会在接收到信号时自动评估并执行交易。同时，修复了 MT5 持仓信息在前端显示的多个问题。

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

*   **调试与优化**
    *   **持仓显示修复**: 修复了 `OngoingOrders` 组件在电脑端不显示 SL/TP、开仓价、现价的问题，并恢复了之前的 Badge 样式。
    *   **权限修复**: 解决了 Supabase RLS 策略导致自动化规则无法保存的问题。
    *   **日志增强**: Python Bridge 增加了详细的 Debug 日志，便于追踪规则同步和信号评估过程。

## 3. 移动端与UX 增强 (2024-11-26 新增)

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

## 6. 遇到的问题与解决方案

| 问题分类 | 问题描述 | 解决方案 |
| :--- | :--- | :--- |
| **架构限制** | 用户无 TradingView 会员，无法使用 Webhook 推送信号。 | **方案转型**: 开发 MQL5 原生指标，配合 Python 文件监听器，实现本地信号闭环。 |
| **Docker 卷共享** | MT5 和 Bridge 容器无法共享信号文件。 | **共享卷配置**: 创建 `signal_data` 卷，分别挂载到两个容器的对应目录。 |
| **文件权限** | MT5 进程 (abc 用户) 无法写入 root 拥有的目录。 | **权限修复**: `chmod 777` 信号目录，确保 MT5 可写入。 |
| **移动端交互** | 拖拽排序与页面滚动冲突，操作不流畅。 | **交互重构**: 移动端禁用拖拽，采用固定堆叠布局；开发“专注模式”提供纯粹体验。 |
| **图表渲染** | 移动端缩放时指标变形，连线混乱。 | **逻辑修正**: 移除 `fitContent` 强制缩放；在指标渲染器中增加无效点检测，断开重绘路径。 |
| **性能优化** | 图表拖动时本地运行卡顿。 | **事件优化**: 移除冗余的 `updateLabels` 频繁调用，优化 ResizeObserver 回调。 |

## 7. 文件清单

### 新增文件

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
| `trading-bridge/src/main.py` | 添加信号文件监听器 |
| `trading-bridge/docker/docker-compose.yml` | 添加共享卷和环境变量映射 |

## 8. 下一步计划 (Phase 5 & Beyond)

当前系统已具备了**自动化交易执行**能力，下一阶段将聚焦于 **AI 交易助手与多账户管理**：

1.  **AI 交易助手**: 基于 LLM 分析当前市场情绪和新闻，结合技术指标给出辅助建议。
2.  **多账户管理**: 支持同时管理多个 MT5 账户，实现跟单或多策略组合。
3.  **回测系统**: 利用积累的历史数据，在前端直接运行策略回测。

---
*Report generated by AlphaOS AI Assistant*
*Last Updated: 2024-11-26*
