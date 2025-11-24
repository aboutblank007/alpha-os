# AlphaOS - MT5 智能交易管理系统

> 一个现代化的 MT5 交易管理和分析平台，集成实时交易执行、数据分析、交易日志等功能。

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

### 2. 📊 智能仪表盘

- ✅ **MT5 实时净资产同步**: 直接显示 MT5 账户净值和浮动盈亏
- ✅ **交易统计分析**: 总盈亏、胜率、盈亏比、最大回撤
- ✅ **持仓订单管理**: 实时持仓显示，一键平仓
- ✅ **市场行情监控**: 多品种实时报价（外汇、黄金、加密货币）
- ✅ **权益曲线图**: 可视化账户净值变化
- ✅ **品种表现分析**: 不同交易品种的盈亏统计
- ✅ **风险警报系统**: 智能监控风险指标
- ✅ **拖拽式布局**: 自定义组件排列，本地保存

### 3. 📝 交易日志 (Journal)

- ✅ **月度日历视图**: 直观展示每日交易盈亏
- ✅ **交易笔记功能**: 支持心情、市场观察、学习要点
- ✅ **标签和策略**: 为交易添加标签和策略分类
- ✅ **CSV 批量导入**: 支持 DeepSeek、ThinkMarkets 等格式
- ✅ **自动去重**: 基于订单编号智能去重
- ✅ **智能数量转换**: 不同平台格式自动统一

### 4. 📈 数据分析 (Analytics)

- ✅ **多维度统计**: 日、周、月、年度交易分析
- ✅ **品种表现**: 每个交易品种的详细统计
- ✅ **策略分析**: 不同策略的胜率和收益对比
- ✅ **盈亏比分析**: 计算平均盈亏比和夏普比率
- ⏳ **MAE/MFE 分析**: 最大不利/有利偏移分析（待实现）
- ⏳ **回撤分析**: 详细的回撤曲线和统计（待实现）

### 5. ⚙️ 用户设置

- ✅ **个人信息管理**: 用户名、邮箱、时区设置
- ✅ **交易偏好**: 默认货币、风险等级
- ✅ **主题定制**: 深色模式、强调色选择
- ✅ **通知配置**: 邮件通知、交易提醒、风险警报

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     AlphaOS Frontend                     │
│          (Next.js 16 + React 19 + TypeScript)           │
│    仪表盘 | 交易日志 | 数据分析 | 设置                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ REST API
                     │
┌────────────────────▼────────────────────────────────────┐
│              Python FastAPI Bridge                       │
│            (trading-bridge/src/main.py)                  │
│    • 接收前端指令                                        │
│    • 交易数据同步到 Supabase                             │
│    • 账户信息同步                                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ ZeroMQ (REQ/REP)
                     │
┌────────────────────▼────────────────────────────────────┐
│               MetaTrader 5 (Wine)                        │
│          BridgeEA.mq5 Expert Advisor                     │
│    • 接收交易指令                                        │
│    • 执行市价单/挂单/平仓                                │
│    • 推送实时价格和持仓                                  │
│    • 报告交易结果                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Supabase Database                     │
│              (PostgreSQL + Real-time)                    │
│    • accounts: 账户信息                                  │
│    • trades: 交易记录                                    │
│    • journal_notes: 交易笔记                             │
│    • user_preferences: 用户设置                          │
└─────────────────────────────────────────────────────────┘
```

### 数据流

```
用户操作 → 前端组件 → Next.js API → Python Bridge → ZeroMQ → MT5 EA → 经纪商服务器
                                          ↓
                                    Supabase 数据库
                                          ↓
                          实时订阅 ← 前端组件更新
```

---

## 🚀 快速开始

### 前置要求

- **Node.js**: >= 18.0
- **Docker**: 用于部署 MT5 Bridge（可选，本地开发不需要）
- **Supabase**: 数据库账户
- **MT5 账户**: 模拟或实盘账户

### 1. 克隆项目

```bash
git clone <repository-url>
cd alpha-os
```

### 2. 安装依赖

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
TRADING_BRIDGE_API_URL=http://api.lootool.cn:8000
# 或本地开发: http://localhost:8000
```

### 4. 设置数据库

在 Supabase 项目的 SQL 编辑器中执行：

```bash
# 查看完整 SQL 脚本
cat OPTIMIZED_DATABASE_SETUP.sql
```

将脚本内容粘贴到 Supabase SQL Editor 并执行。

### 5. 启动开发服务器

```bash
npm run dev
```

访问: `http://localhost:3000`

### 6. 验证配置

访问调试页面检查配置状态：

```
http://localhost:3000/debug
```

应该看到：
- ✅ Supabase 连接正常
- ✅ MT5 Bridge 连接正常（如果已部署）

---

## 📖 功能模块

### 仪表盘 (Dashboard)

**路径**: `/dashboard`

**主要功能**:
- **实时净资产**: 显示 MT5 账户净值和浮动盈亏
- **交易统计卡片**: 净盈亏、胜率、最大回撤、交易笔数
- **市场行情**: 多品种实时报价和一键交易
- **TradingView 图表**: 支持 1分/5分/15分/30分/1小时/4小时/日线
- **持仓订单**: 实时持仓列表，一键平仓
- **权益曲线**: 可视化账户表现
- **品种表现**: 各品种盈亏统计
- **风险警报**: 实时监控交易风险

**快捷操作**:
- 拖拽调整组件顺序
- 切换图表时间周期
- 市场行情快速交易
- 持仓一键平仓

### 交易日志 (Journal)

**路径**: `/journal`

**主要功能**:
- **日历视图**: 月度交易日历，每日盈亏一目了然
- **交易列表**: 详细的交易记录，支持筛选和排序
- **交易笔记**: 记录心情、市场观察、学习要点
- **CSV 导入**: 批量导入历史交易数据
- **标签管理**: 为交易添加自定义标签

**CSV 导入支持**:

1. **DeepSeek 格式**
   - 列名：交易方向、产品代码、数量、持仓、平仓、净盈亏

2. **ThinkMarkets 格式**
   - 列名：商品代码、买/卖、类型、数量、已成交数量

**特殊处理**:
- ✅ 自动过滤非市价订单
- ✅ 自动过滤已取消订单
- ✅ 基于订单编号自动去重
- ✅ 智能数量转换（USDJPY: 20000 单位 → 0.2 手）

### 数据分析 (Analytics)

**路径**: `/analytics`

**主要功能**:
- **多时间段统计**: 日、周、月、年度数据
- **品种表现榜**: 各交易品种收益排行
- **策略分析**: 不同策略的胜率和收益对比
- **盈亏比计算**: 平均盈亏比、夏普比率
- **胜率趋势**: 胜率随时间变化

### 设置 (Settings)

**路径**: `/settings`

**主要功能**:
- **个人信息**: 用户名、邮箱、时区
- **交易偏好**: 默认货币、风险等级
- **界面定制**: 主题、强调色、图表主题
- **通知设置**: 邮件通知、交易提醒、风险警报

---

## 🐳 部署指南

### Docker 部署（推荐）

项目已包含完整的 Docker 配置，支持一键部署。

#### 1. 部署前端

```bash
# 本地构建
docker-compose up -d --build web

# 或使用部署脚本（远程服务器）
./deploy_service.sh frontend
```

#### 2. 部署 MT5 Bridge

```bash
# 进入 trading-bridge 目录
cd trading-bridge/docker

# 配置环境变量（.env 文件）
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key

# 启动服务
docker-compose up -d --build

# 或使用部署脚本
cd ../..
./deploy_service.sh bridge-api
./deploy_service.sh mt5
```

#### 3. 访问服务

- **前端**: `http://your-server-ip:3001`
- **Bridge API**: `http://your-server-ip:8000`
- **MT5 VNC**: `http://your-server-ip:3000` (密码在 docker-compose.yml 中)

### 本地开发

```bash
# 前端
npm run dev

# MT5 Bridge (Python)
cd trading-bridge/src
python main.py
```

---

## 🛠 技术栈

### 前端

| 技术 | 版本 | 说明 |
|------|------|------|
| Next.js | 16.0.3 | React 框架（App Router + Turbopack） |
| React | 19.2.0 | UI 库 |
| TypeScript | 5.x | 类型安全 |
| Tailwind CSS | 4.x | 原子化 CSS |
| Lightweight Charts | 5.0.9 | TradingView 图表库 |
| @dnd-kit | 6.3.1 | 拖拽功能 |
| Lucide React | 0.554.0 | 图标库 |
| Recharts | 3.4.1 | 图表库 |

### 后端

| 技术 | 说明 |
|------|------|
| Next.js API Routes | 前端 API 路由 |
| Python FastAPI | MT5 桥接 API |
| Supabase | PostgreSQL 数据库 + 实时订阅 |
| ZeroMQ | 高性能消息队列 |
| Wine | Linux 上运行 MT5 |

### 基础设施

| 技术 | 说明 |
|------|------|
| Docker | 容器化 |
| Docker Compose | 服务编排 |
| Ubuntu Server | 生产环境 |
| noVNC | Web VNC 远程桌面 |

---

## 📁 项目结构

```
alpha-os/
├── src/
│   ├── app/
│   │   ├── dashboard/          # 仪表盘页面
│   │   ├── journal/            # 交易日志页面
│   │   ├── analytics/          # 数据分析页面
│   │   ├── settings/           # 设置页面
│   │   ├── debug/              # 调试页面
│   │   └── api/                # API 路由
│   │       ├── bridge/         # MT5 Bridge 代理
│   │       ├── trades/         # 交易数据 API
│   │       ├── journal/        # 日志 API
│   │       ├── account/        # 账户 API
│   │       └── prices/         # 价格 API
│   ├── components/
│   │   ├── charts/             # 图表组件
│   │   ├── journal/            # 日志相关组件
│   │   ├── market/             # 市场行情组件
│   │   ├── dashboard/          # 仪表盘组件
│   │   └── ui/                 # UI 组件库
│   ├── hooks/
│   │   └── useBridgeStatus.ts  # MT5 连接状态 Hook
│   └── lib/
│       ├── supabase.ts         # Supabase 客户端
│       ├── mt5-client.ts       # MT5 Bridge 客户端
│       ├── bridge-client.ts    # Bridge API 客户端
│       └── utils.ts            # 工具函数
├── trading-bridge/
│   ├── mql5/
│   │   └── BridgeEA.mq5        # MT5 Expert Advisor
│   ├── docker/
│   │   ├── docker-compose.yml  # Docker 配置
│   │   ├── Dockerfile          # MT5 容器
│   │   └── Dockerfile.api      # Python API 容器
│   └── src/
│       └── main.py             # Python FastAPI 服务
├── public/                      # 静态资源
├── OPTIMIZED_DATABASE_SETUP.sql # 数据库初始化脚本
├── DATABASE_README.md           # 数据库文档
├── DATABASE_MT5_SYNC_UPDATE.md  # MT5 同步文档
├── deploy_service.sh            # 部署脚本
├── docker-compose.yml           # 前端 Docker 配置
├── Dockerfile                   # 前端 Dockerfile
├── package.json                 # NPM 依赖
├── tailwind.config.ts           # Tailwind 配置
├── tsconfig.json                # TypeScript 配置
└── .env.local                   # 环境变量（需创建）
```

---

## 🔐 安全注意事项

1. **永远不要提交 `.env.local` 到 Git**
   - 已在 `.gitignore` 中配置

2. **保护您的密钥**
   - Supabase Keys
   - MT5 账户信息

3. **生产环境建议**
   - 使用 HTTPS
   - 启用防火墙
   - 定期备份数据库
   - 使用强密码

4. **使用模拟账户进行开发**
   - 避免在开发中使用实盘账户

---

## 📊 性能优化

- ✅ Turbopack 加快开发构建（Next.js 16）
- ✅ React 19 Server Components
- ✅ 按需加载组件
- ✅ 图表数据优化
- ✅ 数据库索引优化
- ✅ 实时订阅节流
- ✅ API 响应缓存

---

## 🐛 故障排除

### 常见问题

#### 1. Supabase 连接失败

**症状**: Dashboard 无法加载交易数据

**解决步骤**:
1. 访问 `/debug` 检查连接状态
2. 确认 `.env.local` 中 Supabase 配置正确
3. 确认 Supabase 项目未暂停
4. 完全重启开发服务器（Ctrl+C 然后 `npm run dev`）

#### 2. MT5 Bridge 连接失败

**症状**: 无法执行交易，持仓不显示

**解决步骤**:
1. 检查 Bridge API 是否运行：`docker-compose ps`
2. 查看日志：`docker-compose logs -f bridge-api`
3. 确认 MT5 容器运行正常：`docker-compose logs -f mt5`
4. 检查 VNC 界面，确认 EA 已加载

#### 3. 环境变量不生效

**症状**: 配置更改后无效果

**解决步骤**:
1. 完全停止开发服务器（Ctrl+C）
2. 等待完全停止
3. 重新启动：`npm run dev`
4. 注意：热重载不会加载新的环境变量

#### 4. 交易执行失败

**症状**: 点击交易按钮无反应或报错

**解决步骤**:
1. 检查 MT5 EA 是否允许自动交易
2. 确认账户有足够保证金
3. 检查品种是否在市场开盘时间
4. 查看 Browser Console 和 Network 标签

### 验证清单

启动后依次检查：

1. ✅ **调试页面**: `http://localhost:3000/debug`
2. ✅ **Dashboard**: `http://localhost:3000/dashboard`
3. ✅ **MT5 Bridge Status**: `http://api.lootool.cn:8000/status`
4. ✅ **数据库连接**: 在 Supabase 控制台检查表数据

---

## 🤝 贡献

当前为个人项目，暂不接受外部贡献。

---

## 📄 许可证

私有项目 - 保留所有权利

---

## 📝 更新日志

### v2.0.0 (2025-11-24) - 最新

**重大更新**:
- ✅ 完整的 MT5 交易桥接系统
- ✅ Docker 容器化部署
- ✅ MT5 实时净资产同步
- ✅ 优化的数据库架构（v2.0）
- ✅ 拖拽式仪表盘布局
- ✅ 15分/30分图表支持
- ✅ 防重复交易数据插入

**移除**:
- ❌ OANDA API 集成（已替换为 MT5）
- ❌ TradingView Chrome 扩展（已弃用）

**数据库优化**:
- ✅ 新增 `external_order_id` 字段（防重复）
- ✅ 新增 `position_id`, `mae`, `mfe` 等分析字段
- ✅ 新增 `user_preferences` 表
- ✅ 优化索引和视图
- ✅ RLS 安全策略

### v1.2.0 (2025-11-21)

**新增**:
- ✅ 实时价格集成
- ✅ TradingView Lightweight Charts
- ✅ CSV 导入功能
- ✅ 智能降级机制

### v1.0.0 (初始版本)

**核心功能**:
- ✅ Dashboard 仪表板
- ✅ 交易数据管理
- ✅ Supabase 数据库集成

---

## 📞 支持

如遇问题，请检查：
1. 本 README 的故障排除部分
2. `/debug` 诊断页面
3. 浏览器开发者工具的 Console 和 Network 标签
4. Docker 容器日志：`docker-compose logs -f`

---

**最后更新**: 2025-11-24  
**当前版本**: v2.0.0  
**作者**: AlphaOS Team
