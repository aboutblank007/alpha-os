# AlphaOS - 智能交易管理系统

一个现代化的交易管理和分析平台，集成实时价格数据、交易日志、数据分析等功能。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [功能说明](#功能说明)
- [故障排除](#故障排除)
- [技术栈](#技术栈)

---

## ✨ 功能特性

### ✅ 已完成功能

#### 1. 交易数据管理
- ✅ 交易记录的增删改查
- ✅ CSV 批量导入交易记录
- ✅ 支持多种 CSV 格式（DeepSeek、ThinkMarkets 等）
- ✅ 自动去重功能（基于订单编号）
- ✅ 自动过滤非市价订单和已取消订单
- ✅ 智能数量转换（不同平台格式统一）

#### 2. 实时价格系统 ⭐ 最新
- ✅ 集成 OANDA API 获取真实市场价格
- ✅ 支持外汇、贵金属（黄金、白银）
- ✅ 每 2 秒自动刷新价格
- ✅ 实时计算浮动盈亏
- ✅ 连接状态监控
- ✅ 智能降级（API 不可用时使用模拟数据）
- ✅ TradingView Lightweight Charts 图表集成

#### 3. 仪表板 (Dashboard)
- ✅ 实时持仓订单显示
- ✅ 净资产实时更新（初始资金 + 已实现盈亏 + 浮动盈亏）
- ✅ 交易统计（总盈亏、胜率、交易笔数等）
- ✅ 净值曲线图
- ✅ 品种表现分析
- ✅ 最近交易记录
- ✅ 风险警报

#### 4. 交易日志 (Journal)
- ✅ 月度日历视图
- ✅ 每日交易盈亏显示
- ✅ 交易笔记功能（支持心情和标签）
- ✅ 每日交易摘要
- ✅ CSV 导入功能

#### 5. 数据分析 (Analytics)
- ✅ 基础交易指标分析
- ✅ 盈亏比、夏普比率等
- ⏳ MAE/MFE 分析（待实现）
- ⏳ 回撤分析（待实现）

#### 6. 用户设置
- ✅ 基础设置页面
- ✅ 个人信息管理
- ✅ 交易偏好设置

#### 7. Chrome 扩展
- ✅ TradingView 订单同步
- ✅ 自动推送新订单到后端
- ✅ 持仓同步功能

#### 8. 跨平台交易桥接 (AlphaBridge) ⭐ 新增
- ✅ **Docker 容器化架构**: 包含 MT5 终端、Wine 环境和 Python API 桥接
- ✅ **ZeroMQ 高速通信**: 实现毫秒级指令传输
- ✅ **远程部署**: 支持 Ubuntu 服务器一键部署
- ✅ **VNC 可视化**: 提供 Web VNC 界面进行远程管理
- ✅ **REST API**: 标准化接口对接 AlphaOS 前端

### ⏳ 计划中功能

- [ ] WebSocket 实时价格推送（替代轮询）
- [ ] 价格预警功能
- [ ] 移动端适配
- [ ] 多账户支持
- [ ] 数据导出（PDF 报告）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

在项目根目录创建 `.env.local` 文件：

```bash
# OANDA API 配置（实时价格）
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice

# Supabase 配置（数据库）
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. 设置数据库

在 Supabase 项目的 SQL 编辑器中执行：

```bash
# 查看完整 SQL 脚本
cat COMPLETE_DATABASE_SETUP.sql
```

### 4. 启动开发服务器

```bash
npm run dev
```

访问: `http://localhost:3000`

### 5. 验证配置

访问以下 URL 验证环境变量配置：

```
http://localhost:3000/api/test-env
```

应该看到所有配置项都是 ✅

---

## 🔧 环境配置

### OANDA API 配置

#### 获取 API 密钥

1. 访问 [OANDA 开发者页面](https://www.oanda.com/account/tpa/personal_token)
2. 注册或登录（建议使用 Practice 练习账户）
3. 生成 Personal Access Token
4. 记录 API Key 和 Account ID

#### 环境说明

- **Practice**: 练习环境（免费、无限制、推荐）
  - API: `https://api-fxpractice.oanda.com`
- **Live**: 实盘环境（需要实盘账户）
  - API: `https://api-fxtrade.oanda.com`

#### 支持的交易品种

- ✅ 外汇对（EUR/USD, USD/JPY, GBP/USD 等）
- ✅ 贵金属（XAU/USD 黄金, XAG/USD 白银）
- ✅ 部分商品和指数 CFD
- ❌ 加密货币（不支持）

### Supabase 配置

1. 访问 [Supabase](https://supabase.com)
2. 创建新项目
3. 在项目设置中找到 API URL 和 anon key
4. 执行 `COMPLETE_DATABASE_SETUP.sql` 创建数据库表

---

## 📖 功能说明

### 实时价格系统

#### 工作原理

```
OANDA API → /api/prices → OngoingOrders 组件 → 实时盈亏计算
     ↓
每 2 秒刷新
```

#### 使用方法

**自动使用**（无需代码修改）：
- 配置 OANDA API 密钥后，系统自动使用真实价格
- 在 Dashboard 的"持仓订单"面板查看实时更新

**手动调用 API**：
```typescript
// 获取实时价格
const res = await fetch('/api/prices?symbols=USDJPY,EURUSD,XAUUSD');
const data = await res.json();
console.log(data.prices); // { USDJPY: 156.789, ... }

// 获取历史K线
const res = await fetch('/api/prices', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    instrument: 'USDJPY',
    granularity: 'M5',  // 5分钟K线
    count: 200
  })
});
```

**添加价格图表**：
```tsx
import { TradingViewChart } from '@/components/charts/TradingViewChart';

<TradingViewChart symbol="USDJPY" height={400} />
```

### CSV 导入功能

#### 支持的格式

系统会自动识别以下 CSV 格式：

1. **DeepSeek 格式**
   - 列名：交易方向、产品代码、数量、持仓、平仓、净盈亏等
   
2. **ThinkMarkets 格式**
   - 列名：商品代码、买/卖、类型、数量、已成交数量等

#### 使用方法

1. 进入"交易日志"页面
2. 点击"导入 CSV"按钮
3. 选择 CSV 文件
4. 预览数据
5. 点击"导入交易"

#### 特殊处理

- ✅ 自动过滤非市价订单（止损、止盈等）
- ✅ 自动过滤已取消订单
- ✅ 自动去重（基于订单编号）
- ✅ 智能数量转换：
  - USDJPY: 20000 单位 → 0.2 手
  - XAUUSD: 200 盎司 → 2 手

### 净资产计算

```
净资产 = 初始资金 + 已实现盈亏 + 浮动盈亏

其中：
- 初始资金：在 accounts 表中设置
- 已实现盈亏：所有 status='closed' 的交易的 pnl_net 总和
- 浮动盈亏：所有 status='open' 的交易根据实时价格计算
```

实时更新周期：30 秒

### 盈亏计算公式

#### JPY 货币对（如 USDJPY）
```
盈亏 = (当前价 - 开仓价) × 手数 × 1000
```

#### 黄金（XAUUSD）
```
盈亏 = (当前价 - 开仓价) × 手数 × 100
```

#### 其他外汇对
```
盈亏 = (当前价 - 开仓价) × 手数 × 100000
```

**注意**: 卖单的盈亏需要取反

---

## 🐛 故障排除

### 常见问题

#### 1. "OANDA API 未配置，返回模拟价格"

**原因**: 环境变量未加载

**解决**:
1. 确认 `.env.local` 文件存在于项目根目录
2. 确认变量名是 `OANDA_API_KEY`（不是其他名称）
3. **完全重启开发服务器**（Ctrl+C 停止，然后 `npm run dev`）
4. 访问 `/api/test-env` 验证配置

#### 2. "Error fetching trades: {}"

**原因**: Supabase 环境变量未加载或配置错误

**解决步骤**:

1. **验证环境变量配置**
   ```
   访问: http://localhost:3000/api/test-env
   ```
   确认看到：`"supabase": "✅ Supabase 配置完整"`

2. **检查 .env.local 文件**
   确保文件中包含（注意 `NEXT_PUBLIC_` 前缀）：
   ```bash
   NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJxxx...
   ```

3. **完全重启开发服务器**（关键！）
   ```bash
   # 在运行 npm run dev 的终端中按 Ctrl+C 完全停止
   # 然后重新运行
   npm run dev
   ```

4. **检查 Supabase 项目状态**
   - 登录 Supabase 控制台
   - 确认项目未暂停（免费计划会自动暂停）
   - 如果暂停，点击"Resume"恢复

5. **查看详细错误**
   打开浏览器开发者工具（F12）：
   - Console 标签查看错误详情
   - Network 标签查看失败的请求
   - 现在会显示更详细的错误信息

#### 3. 持仓订单不显示

**原因**: 数据库中 status 字段不正确

**解决**:
```sql
-- 检查持仓订单的 status
SELECT id, symbol, side, status FROM trades WHERE status = 'open';

-- 如果需要，手动更新
UPDATE trades SET status = 'open' 
WHERE external_order_id IN ('订单号1', '订单号2');
```

#### 4. 环境变量不生效

**关键**: 修改 `.env.local` 后，必须**完全重启**开发服务器！

```bash
# 1. 停止服务器（在运行 npm run dev 的终端按 Ctrl+C）
# 2. 等待完全停止
# 3. 重新启动
npm run dev
```

热重载（Hot Reload）不会加载新的环境变量！

### 验证清单

启动后依次检查：

1. **✅ 环境变量**: `http://localhost:3000/api/test-env`
2. **✅ 实时价格**: `http://localhost:3000/api/prices?symbols=USDJPY`
3. **✅ Dashboard**: `http://localhost:3000/dashboard`

### 日志分析

**正常日志**:
```bash
✓ Ready in 2.3s
GET /api/prices?symbols=USDJPY 200 in 234ms
```

**异常日志**:
```bash
OANDA API 未配置，返回模拟价格 ❌
Error fetching trades: {} ❌
```

---

## 🛠 技术栈

### 前端
- **框架**: Next.js 16 (App Router + Turbopack)
- **语言**: TypeScript
- **样式**: Tailwind CSS
- **图表**: TradingView Lightweight Charts
- **UI 组件**: 自定义组件库

### 后端
- **运行时**: Next.js API Routes
- **数据库**: Supabase (PostgreSQL)
- **实时数据**: OANDA REST API v20

### 工具
- **Chrome 扩展**: 用于 TradingView 集成
- **版本控制**: Git

---

## 📁 项目结构

```
alpha-os/
├── src/
│   ├── app/
│   │   ├── dashboard/          # 仪表板页面
│   │   ├── journal/            # 交易日志页面
│   │   ├── analytics/          # 数据分析页面
│   │   ├── settings/           # 设置页面
│   │   └── api/                # API 路由
│   │       ├── prices/         # 实时价格 API
│   │       ├── trades/         # 交易数据 API
│   │       ├── journal/        # 日志 API
│   │       └── test-env/       # 环境测试 API
│   ├── components/
│   │   ├── charts/             # 图表组件
│   │   ├── journal/            # 日志相关组件
│   │   └── ui/                 # UI 组件库
│   └── lib/
│       ├── oanda.ts            # OANDA API 客户端
│       └── supabase.ts         # Supabase 客户端
├── alpha-link-extension/       # Chrome 扩展
├── COMPLETE_DATABASE_SETUP.sql # 数据库设置脚本
└── .env.local                  # 环境变量（需创建）
```

---

## 🔐 安全注意事项

1. **永远不要提交 `.env.local` 到 Git**
   - 已在 `.gitignore` 中配置
   
2. **保护您的 API 密钥**
   - OANDA API Key
   - Supabase Keys
   
3. **定期轮换密钥**
   - 建议每 3-6 个月更换一次

4. **使用 Practice 环境进行开发**
   - 避免在开发中使用实盘账户

---

## 📊 性能优化

- ✅ 使用 Turbopack 加快开发构建
- ✅ API 响应缓存
- ✅ 按需加载组件
- ✅ 图表数据优化
- ✅ 数据库查询优化（索引、视图）

---

## 🤝 贡献

当前为个人项目，暂不接受外部贡献。

---

## 📄 许可证

私有项目 - 保留所有权利

---

## 📞 支持

如遇问题，请检查：
1. 本 README 的故障排除部分
2. 浏览器开发者工具的 Console 和 Network 标签
3. 开发服务器的日志输出

---

## 📝 更新日志

### v1.2.0 (2025-11-21) - 最新

**新增**:
- ✅ OANDA API 实时价格集成
- ✅ TradingView Lightweight Charts 图表
- ✅ 智能降级机制
- ✅ 环境变量测试端点 (`/api/test-env`)
- ✅ 详细的故障排除文档

**优化**:
- ✅ CSV 导入支持更多格式
- ✅ 盈亏计算更精确
- ✅ 净资产实时更新
- ✅ 数量单位自动转换

### v1.1.0 (2025-11-20)

**新增**:
- ✅ 交易日志日历功能
- ✅ 交易笔记（心情 + 标签）
- ✅ 用户设置页面
- ✅ CSV 批量导入
- ✅ 去重功能

### v1.0.0 (初始版本)

**核心功能**:
- ✅ Dashboard 仪表板
- ✅ 交易数据管理
- ✅ Supabase 数据库集成
- ✅ Chrome 扩展

---

**最后更新**: 2025-11-21  
**当前版本**: v1.2.0

