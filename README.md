# AlphaOS - MT5 智能交易管理系统

> 一个现代化的 MT5 交易管理和分析平台，集成实时交易执行、数据分析、交易日志以及**分布式 AI 信号过滤**功能。

---

## 📋 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [功能模块](#功能模块)
- [部署指南](#部署指南)
- [技术栈](#技术栈)
- [项目结构](#项目结构)

---

## ✨ 核心特性

### 1. 🔗 MT5 交易桥接 (AlphaBridge)

- ✅ **Docker 容器化架构**: MT5 终端 + Wine 环境 + Python API 桥接
- ✅ **ZeroMQ 高速通信**: 实现毫秒级指令传输
- ✅ **远程部署支持**: 支持 Ubuntu 服务器一键部署
- ✅ **VNC 可视化管理**: 提供 Web VNC 界面远程管理 MT5
- ✅ **REST API**: 标准化接口对接前端
- ✅ **实时价格数据**: MT5 实时行情推送
- ✅ **自动交易执行**: 市价单、挂单、平仓操作

### 2. 🧠 分布式 AI 引擎 (Phase 5)

- ✅ **本地高性能推理**: 利用本地 M2 Pro/M3 Max 芯片进行 AI 运算，无需昂贵的云 GPU
- ✅ **gRPC 实时流**: 云端 Bridge 与本地 AI 引擎通过 gRPC 双向实时通信
- ✅ **双模式支持**:
    - **Pure AI**: 纯 AI 驱动的信号生成
    - **Indicator + AI**: 传统指标信号 + AI 二次确认过滤
- ✅ **元标注 (Meta-Labeling)**: 自动收集交易上下文用于模型迭代训练

### 3. 📊 智能仪表盘

- ✅ **MT5 实时净资产同步**: 直接显示 MT5 账户净值和浮动盈亏
- ✅ **交易统计分析**: 总盈亏、胜率、盈亏比、最大回撤
- ✅ **持仓订单管理**: 实时持仓显示，一键平仓
- ✅ **市场行情监控**: 多品种实时报价（外汇、黄金、加密货币）
- ✅ **权益曲线图**: 可视化账户净值变化
- ✅ **品种表现分析**: 不同交易品种的盈亏统计
- ✅ **风险警报系统**: 智能监控风险指标
- ✅ **拖拽式布局**: 自定义组件排列，本地保存

### 4. 📝 交易日志 (Journal)

- ✅ **月度日历视图**: 直观展示每日交易盈亏
- ✅ **交易笔记功能**: 支持心情、市场观察、学习要点
- ✅ **标签和策略**: 为交易添加标签和策略分类
- ✅ **CSV 批量导入**: 支持 DeepSeek、ThinkMarkets 等格式
- ✅ **自动去重**: 基于订单编号智能去重
- ✅ **智能数量转换**: 不同平台格式自动统一

### 5. 📈 数据分析 (Analytics)

- ✅ **多维度统计**: 日、周、月、年度交易分析
- ✅ **品种表现榜**: 各交易品种收益排行
- ✅ **策略分析**: 不同策略的胜率和收益对比
- ✅ **盈亏比分析**: 计算平均盈亏比和夏普比率
- ⏳ **MAE/MFE 分析**: 最大不利/有利偏移分析（待实现）
- ⏳ **回撤分析**: 详细的回撤曲线和统计（待实现）

### 6. ⚙️ 用户设置

- ✅ **个人信息管理**: 用户名、邮箱、时区设置
- ✅ **交易偏好**: 默认货币、风险等级
- ✅ **主题定制**: 深色模式、强调色选择
- ✅ **通知配置**: 邮件通知、交易提醒、风险警报
- ✅ **自动化规则**: 配置 AI 模式和信心阈值

---

## 🏗 系统架构

```mermaid
graph TB
    Frontend[AlphaOS Frontend\nNext.js 16] -- REST API --> Bridge[Cloud Bridge API\nFastAPI]
    Bridge -- ZeroMQ --> MT5[MetaTrader 5\nWine Container]
    Bridge -- gRPC Stream <--> LocalAI[Local AI Engine\nPython Client (M2 Pro)]
    MT5 -- Sync --> Supabase[(Supabase DB)]
    Bridge -- Sync --> Supabase
    Frontend -- Realtime --> Supabase
```

### 分布式拓扑
- **Cloud**: 运行前端、Bridge API、MT5 容器和 Supabase。负责路由转发和数据持久化。
- **Local**: 运行 AI Engine。利用本地算力进行特征计算和模型推理，保护策略隐私并降低成本。

---

## 🚀 快速开始

### 前置要求

- **Node.js**: >= 18.0
- **Docker**: 用于部署 MT5 Bridge
- **Supabase**: 数据库账户
- **Python 3.9+**: 用于本地 AI 引擎

### 1. 克隆项目

```bash
git clone <repository-url>
cd alpha-os
```

### 2. 安装前端依赖

```bash
npm install
```

### 3. 配置环境变量

在项目根目录创建 `.env.local` 文件：

```bash
# Supabase 配置（数据库）
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key

# MT5 Trading Bridge API
TRADING_BRIDGE_API_URL=http://your-server-ip:8000
```

### 4. 设置数据库

在 Supabase 项目的 SQL 编辑器中执行：

```bash
# 查看完整 SQL 脚本
cat src/db/FULL_SCHEMA_V2.sql
```

将 `src/db/FULL_SCHEMA_V2.sql` 的内容粘贴到 Supabase SQL Editor 并执行。

### 5. 启动开发服务器

```bash
npm run dev
```

访问: `http://localhost:3000`

---

## 🐳 部署指南

### 1. 云端服务部署 (Docker)

项目已包含完整的 Docker 配置，支持一键部署前端和 MT5 Bridge。

```bash
# 1. 部署所有服务 (MT5 + Bridge API + Frontend)
./deploy_service.sh all

# 2. 或者仅部署 Bridge 和 MT5
./deploy_service.sh bridge-api
./deploy_service.sh mt5
```

### 2. 本地 AI 引擎启动 (Mac/Local)

在本地高性能机器（如 Mac M2/M3）上运行 AI 推理引擎：

```bash
cd ai-engine

# 1. 首次安装环境
./setup_local.sh

# 2. 启动客户端 (连接到云端 Bridge)
source venv/bin/activate
export CLOUD_BRIDGE_URL=your-cloud-ip:50051
python src/client.py
```

---

## 📖 功能模块

### 仪表盘 (Dashboard)
**路径**: `/dashboard`
- **实时净资产**: 显示 MT5 账户净值和浮动盈亏
- **AI 信号通知**: 实时接收并展示经过 AI 过滤的交易信号
- **TradingView 图表**: 集成 TradingView 轻量级图表库

### 自动化设置 (Settings -> Automation)
**路径**: `/settings`
- **AI 模式选择**: 切换 "经典"、"指标+AI" 或 "纯 AI" 模式
- **信心阈值**: 设置 AI 介入的最低信心分数 (0.5 - 0.95)

---

## 🛠 技术栈

### 前端
| 技术 | 版本 | 说明 |
|------|------|------|
| Next.js | 16.0.3 | React 框架（App Router） |
| React | 19.2.0 | UI 库 |
| TypeScript | 5.x | 类型安全 |
| Tailwind CSS | 4.x | 样式库 |
| Lightweight Charts | 5.0.9 | 图表库 |

### 后端 (Cloud Bridge)
| 技术 | 说明 |
|------|------|
| Python FastAPI | API 服务网关 |
| gRPC (AsyncIO) | 实时双向流通信 |
| ZeroMQ | 与 MT5 进程内通信 |
| Docker | 容器化部署 |

### AI Engine (Local)
| 技术 | 说明 |
|------|------|
| LightGBM | 梯度提升决策树模型 |
| scikit-learn | 特征工程管道 |
| gRPC Client | 长连接通信 |

---

## 📝 更新日志

### v2.1.0 (2025-11-27) - AI 驱动版
**Phase 5: Distributed AI Architecture**
- ✅ **架构升级**: 引入 gRPC 分布式架构，实现 Local AI + Cloud Bridge。
- ✅ **AI 引擎**: 新增 `ai-engine` 模块，支持本地特征计算和推理。
- ✅ **前端增强**: 自动化规则支持配置 AI 模式和信心阈值；信号通知支持显示 AI 决策。
- ✅ **数据库合并**: 所有 SQL 脚本整合为 `src/db/FULL_SCHEMA_V2.sql`。

### v2.0.0 (2025-11-24)
- ✅ 完整的 MT5 交易桥接系统
- ✅ Docker 容器化部署
- ✅ 优化的数据库架构

---

## 📄 许可证

私有项目 - 保留所有权利
