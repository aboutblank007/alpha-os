# 开发指南

本文档包含开发过程中需要的技术细节和 API 参考。

## 📋 目录

- [环境变量](#环境变量)
- [API 参考](#api-参考)
- [数据库架构](#数据库架构)
- [组件使用](#组件使用)
- [开发工作流](#开发工作流)

---

## 🔧 环境变量

### 必需的环境变量

```bash
# .env.local

# OANDA API（实时价格）
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_ID=your_account_id  
OANDA_ENVIRONMENT=practice        # practice 或 live

# Supabase（数据库）
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJxxx...
```

### 注意事项

1. **客户端变量需要 `NEXT_PUBLIC_` 前缀**
   - Supabase 变量在客户端使用，所以需要前缀
   - OANDA 变量只在服务端使用，不需要前缀

2. **修改后必须重启服务器**
   - 热重载不会加载新的环境变量
   - 完全停止（Ctrl+C）并重新运行 `npm run dev`

3. **验证配置**
   ```
   访问: http://localhost:3000/api/test-env
   ```

---

## 🌐 API 参考

### 1. 实时价格 API

#### GET /api/prices

获取多个交易品种的实时价格。

**请求**:
```http
GET /api/prices?symbols=USDJPY,EURUSD,XAUUSD
```

**响应**:
```json
{
  "prices": {
    "USDJPY": 156.789,
    "EURUSD": 1.0854,
    "XAUUSD": 4087.25
  },
  "timestamp": "2025-11-21T18:30:00.000Z",
  "source": "oanda"  // 或 "mock" (降级模式)
}
```

#### POST /api/prices

获取历史K线数据。

**请求**:
```json
{
  "instrument": "USDJPY",
  "granularity": "M5",  // S5,M1,M5,H1,D,W,M
  "count": 200
}
```

**响应**:
```json
{
  "instrument": "USD_JPY",
  "candles": [
    {
      "time": 1700000000,
      "open": 156.50,
      "high": 156.80,
      "low": 156.40,
      "close": 156.70,
      "volume": 1500
    }
  ],
  "granularity": "M5",
  "count": 200
}
```

### 2. 交易数据 API

#### GET /api/trades (暂未实现)

获取所有交易记录。

#### POST /api/trades

创建单个交易记录。

**请求**:
```json
{
  "symbol": "USDJPY",
  "side": "buy",
  "entry_price": 156.50,
  "quantity": 0.1,
  "status": "open",
  "notes": "测试订单"
}
```

#### POST /api/trades/import

批量导入交易记录。

**请求**:
```json
{
  "trades": [
    {
      "symbol": "USDJPY",
      "side": "buy",
      "entry_price": 156.50,
      "quantity": 0.1,
      "pnl_net": 10.5,
      "external_order_id": "123456"  // 用于去重
    }
  ]
}
```

**响应**:
```json
{
  "success": true,
  "count": 10,           // 成功导入数量
  "skipped": 2,          // 跳过的重复订单
  "errors": [],          // 错误详情（如有）
  "skippedDetails": []   // 跳过的订单详情
}
```

### 3. 账户余额 API

#### GET /api/account/balance

获取账户净资产和盈亏信息。

**响应**:
```json
{
  "initialBalance": 1084.8,
  "realizedPnl": 125.50,
  "floatingPnl": 15.25,
  "netAsset": 1225.55,
  "totalPnlChange": 140.75
}
```

### 4. 交易日志 API

#### GET /api/journal/notes

获取日期范围内的笔记。

**请求**:
```
GET /api/journal/notes?startDate=2025-11-01&endDate=2025-11-30
```

#### POST /api/journal/notes

创建或更新笔记。

**请求**:
```json
{
  "date": "2025-11-21",
  "note_content": "今天交易很顺利",
  "mood": "confident",
  "tags": ["盈利", "策略A"]
}
```

### 5. 每日统计 API

#### GET /api/trades/daily-stats

获取每日交易统计。

**请求**:
```
GET /api/trades/daily-stats?startDate=2025-11-01&endDate=2025-11-30
```

**响应**:
```json
[
  {
    "date": "2025-11-21",
    "netPnl": 125.50,
    "tradeCount": 5,
    "winRate": 0.8
  }
]
```

---

## 💾 数据库架构

### 核心表

#### accounts 表
```sql
CREATE TABLE accounts (
  id UUID PRIMARY KEY,
  created_at TIMESTAMP,
  initial_balance DECIMAL,
  current_balance DECIMAL,
  currency VARCHAR(3)
);
```

#### trades 表
```sql
CREATE TABLE trades (
  id UUID PRIMARY KEY,
  created_at TIMESTAMP,
  account_id UUID,
  symbol VARCHAR(20),
  side VARCHAR(4),         -- 'buy' or 'sell'
  entry_price DECIMAL,
  exit_price DECIMAL,
  quantity DECIMAL,
  pnl_net DECIMAL,
  pnl_gross DECIMAL,
  commission DECIMAL,
  swap DECIMAL,
  status VARCHAR(10),      -- 'open' or 'closed'
  notes TEXT,
  external_order_id VARCHAR(50) UNIQUE  -- 用于去重
);
```

#### journal_notes 表
```sql
CREATE TABLE journal_notes (
  id UUID PRIMARY KEY,
  created_at TIMESTAMP,
  date DATE UNIQUE,
  note_content TEXT,
  mood VARCHAR(50),
  tags TEXT[]
);
```

### 视图

#### daily_stats_view (自动计算每日统计)
```sql
CREATE VIEW daily_stats_view AS
SELECT 
  DATE(created_at) as date,
  COUNT(*) as trade_count,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END)::float / COUNT(*) as win_rate
FROM trades
WHERE status = 'closed'
GROUP BY DATE(created_at);
```

---

## 🧩 组件使用

### TradingView 图表组件

```tsx
import { TradingViewChart } from '@/components/charts/TradingViewChart';

function MyPage() {
  return (
    <TradingViewChart 
      symbol="USDJPY"     // 交易品种
      height={400}        // 图表高度
      width={800}         // 图表宽度（可选，默认100%）
    />
  );
}
```

### 导入交易 Modal

```tsx
import { ImportTradesModal } from '@/components/journal/ImportTradesModal';

function MyPage() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <ImportTradesModal
      open={isOpen}
      onOpenChange={setIsOpen}
      onSuccess={() => {
        console.log('导入成功');
        // 刷新数据
      }}
    />
  );
}
```

### 持仓订单组件

```tsx
import { OngoingOrders } from '@/components/OngoingOrders';

function Dashboard() {
  const openTrades = trades.filter(t => t.status === 'open');
  
  return <OngoingOrders orders={openTrades} />;
}
```

---

## 🔄 开发工作流

### 1. 添加新功能

```bash
# 1. 创建功能分支（如果使用 git）
git checkout -b feature/new-feature

# 2. 开发功能

# 3. 测试
npm run dev

# 4. 检查 linter
npm run lint

# 5. 构建测试
npm run build
```

### 2. 数据库迁移

```bash
# 1. 在 Supabase SQL 编辑器中编写 SQL

# 2. 保存到项目中
# 创建新文件: migrations/YYYYMMDD_description.sql

# 3. 在 Supabase 执行 SQL

# 4. 更新 TypeScript 类型（如需要）
# 编辑 src/lib/supabase.ts
```

### 3. 添加新的 API 端点

```typescript
// src/app/api/your-endpoint/route.ts

import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    // 处理逻辑
    return NextResponse.json({ data: 'success' });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
```

### 4. 盈亏计算公式

在组件中实现自定义盈亏计算：

```typescript
function calculatePnL(
  entryPrice: number,
  currentPrice: number,
  quantity: number,
  side: 'buy' | 'sell',
  symbol: string
): number {
  const diff = currentPrice - entryPrice;
  let pnl = 0;
  
  // 根据品种选择乘数
  if (symbol.includes('JPY')) {
    pnl = diff * quantity * 1000;
  } else if (symbol.includes('XAU')) {
    pnl = diff * quantity * 100;
  } else {
    pnl = diff * quantity * 100000;
  }
  
  // 卖单需要反转
  return side === 'sell' ? -pnl : pnl;
}
```

---

## 🧪 测试

### 环境变量测试

```
http://localhost:3000/api/test-env
```

### API 测试

使用浏览器或 curl：

```bash
# 测试实时价格
curl "http://localhost:3000/api/prices?symbols=USDJPY"

# 测试账户余额
curl "http://localhost:3000/api/account/balance"
```

### 组件测试

在开发环境中直接访问页面：

- Dashboard: `http://localhost:3000/dashboard`
- Journal: `http://localhost:3000/journal`
- Analytics: `http://localhost:3000/analytics`
- Settings: `http://localhost:3000/settings`

---

## 📦 构建和部署

### 本地构建

```bash
npm run build
```

### 生产环境变量

确保在生产环境设置所有环境变量：

```bash
# Vercel/Netlify 等平台
OANDA_API_KEY=xxx
OANDA_ACCOUNT_ID=xxx
OANDA_ENVIRONMENT=practice
NEXT_PUBLIC_SUPABASE_URL=xxx
NEXT_PUBLIC_SUPABASE_ANON_KEY=xxx
```

---

## 🔍 调试技巧

### 1. 查看实时价格源

在浏览器控制台：

```javascript
fetch('/api/prices?symbols=USDJPY')
  .then(r => r.json())
  .then(data => console.log('Source:', data.source));
// 应该输出: "oanda" (真实价格) 或 "mock" (模拟)
```

### 2. 检查环境变量

```javascript
// 客户端（浏览器控制台）
console.log('Supabase URL:', process.env.NEXT_PUBLIC_SUPABASE_URL);

// 服务端（API 路由）
console.log('OANDA Key:', process.env.OANDA_API_KEY);
```

### 3. Supabase 查询调试

```typescript
const { data, error } = await supabase
  .from('trades')
  .select('*');

console.log('Data:', data);
console.log('Error:', error);  // 查看详细错误信息
```

---

## 📝 代码规范

### TypeScript

- 使用 `interface` 而不是 `type`（除非必要）
- 为所有函数参数添加类型
- 避免使用 `any`

### 组件

- 使用函数组件和 Hooks
- 组件文件使用 PascalCase
- 一个文件一个主组件

### API 路由

- 始终返回 JSON
- 使用标准 HTTP 状态码
- 包含错误处理

### 命名约定

- 文件：`kebab-case.tsx`
- 组件：`PascalCase`
- 函数：`camelCase`
- 常量：`UPPER_SNAKE_CASE`

---

**最后更新**: 2025-11-21  
**适用版本**: v1.2.0

