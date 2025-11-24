# AlphaOS 数据库配置指南

## 📋 概述

本目录包含完整的数据库设置和迁移脚本，适配前端所有功能：
- ✅ **仪表盘** - 实时统计、权益曲线、持仓订单
- ✅ **交易日志** - 日历视图、每日统计、交易笔记
- ✅ **数据分析** - MAE/MFE分析、策略表现、风险分析
- ✅ **设置** - 用户偏好、通知配置、主题设置

## 📁 文件说明

### 核心脚本

| 文件名 | 用途 | 适用场景 |
|-------|------|---------|
| `OPTIMIZED_DATABASE_SETUP.sql` | 完整数据库设置脚本 | 全新安装或完全重建 |
| `MIGRATION_TO_V2.sql` | 数据库迁移脚本 | 从v1.0升级到v2.0 |
| `cleanup_old_trades.sql` | 清理旧测试数据 | 删除无效/重复数据 |
| `SET_INITIAL_BALANCE.sql` | 设置初始本金 | 配置账户余额 |

### 数据表结构

#### 1. accounts (账户表)
存储交易账户信息和余额。

**核心字段：**
- `initial_balance` - 初始本金
- `current_balance` - 当前余额（自动计算）
- `account_type` - 账户类型（demo/live）
- `broker` - 经纪商名称

#### 2. trades (交易记录表) - 增强版
完整的交易信息，支持所有前端功能。

**基本信息：**
- `symbol`, `side`, `entry_price`, `exit_price`, `quantity`
- `entry_time`, `exit_time` - 精确的进出场时间

**盈亏分析：**
- `pnl_net`, `pnl_gross` - 净盈亏和毛盈亏
- `commission`, `swap` - 手续费和隔夜利息
- `mae`, `mfe` - 最大不利偏移和最大有利偏移

**风险管理：**
- `stop_loss`, `take_profit` - 止损止盈价格
- `risk_reward_ratio` - 风险回报比
- `position_size_pct` - 仓位百分比

**交易分析：**
- `holding_time_seconds` - 持仓时间（秒）
- `strategies[]` - 使用的策略（数组）
- `tags[]` - 自定义标签
- `emotion_score`, `confidence_level` - 情绪和信心评分

**系统字段：**
- `external_order_id` - MT5 Position ID（唯一）
- `source` - 数据来源（manual/mt5/api/import）
- `status` - 状态（open/closed/cancelled）

#### 3. journal_notes (交易笔记表) - 增强版
每日交易复盘和心得记录。

**核心内容：**
- `content` - 主要笔记内容（支持 Markdown）
- `summary` - 内容摘要
- `date` - 日期（唯一）

**情绪追踪：**
- `mood` - 心情（confident/calm/anxious/frustrated/excited/tired/focused）
- `energy_level` - 精力水平（1-5）
- `mental_state` - 心理状态（clear/distracted/stressed/relaxed）

**市场观察：**
- `market_condition` - 市场状况描述
- `market_sentiment` - 市场情绪（bullish/bearish/neutral/volatile）

**学习改进：**
- `lessons_learned[]` - 学到的经验
- `mistakes_made[]` - 犯的错误
- `improvement_notes` - 改进建议

**统计快照：**
- `daily_pnl` - 当日盈亏
- `daily_trades_count` - 当日交易数
- `daily_win_rate` - 当日胜率

#### 4. user_preferences (用户偏好表) - 新增
所有前端设置和偏好配置。

**个人信息：**
- `display_name`, `email`, `timezone`, `language`

**交易偏好：**
- `default_currency` - 默认货币
- `default_risk_percentage` - 默认风险百分比
- `favorite_symbols[]` - 收藏的交易品种
- `risk_level` - 风险级别

**显示设置：**
- `show_live_price` - 显示实时价格
- `auto_sync` - 自动同步
- `dashboard_layout` - 仪表盘布局（JSON）
- `chart_default_timeframe` - 默认图表周期

**通知设置：**
- `email_notifications`, `trade_alerts`, `risk_alerts`
- `win_notification`, `loss_notification`
- `daily_summary`

**主题设置：**
- `theme` - 主题（light/dark/auto）
- `accent_color` - 强调色
- `chart_theme` - 图表主题

### 数据视图

#### daily_trade_stats
每日交易统计，用于交易日志页面。

```sql
SELECT * FROM daily_trade_stats WHERE trade_date = CURRENT_DATE;
```

#### symbol_performance
品种表现分析，用于数据分析页面。

```sql
SELECT * FROM symbol_performance ORDER BY total_pnl DESC LIMIT 10;
```

#### monthly_stats
月度统计，用于仪表盘。

```sql
SELECT * FROM monthly_stats ORDER BY month DESC LIMIT 12;
```

#### strategy_performance
策略表现分析。

```sql
SELECT * FROM strategy_performance WHERE total_trades >= 10;
```

## 🚀 使用指南

### 场景1：全新安装（推荐）

1. **登录 Supabase 控制台**
   - 访问 https://app.supabase.com
   - 进入您的项目
   - 点击 SQL Editor

2. **执行完整设置脚本**
   ```sql
   -- 复制并执行 OPTIMIZED_DATABASE_SETUP.sql
   ```

3. **设置初始本金**
   ```sql
   -- 复制并执行 SET_INITIAL_BALANCE.sql
   -- 修改本金金额为您的实际金额
   ```

4. **刷新前端**
   - 打开 http://localhost:3001 或您的域名
   - 所有功能应正常工作

### 场景2：从v1.0升级

1. **备份现有数据**
   ```sql
   -- 在 Supabase 控制台导出 trades 表
   SELECT * FROM trades ORDER BY created_at DESC;
   ```

2. **执行迁移脚本**
   ```sql
   -- 复制并执行 MIGRATION_TO_V2.sql
   ```

3. **清理旧测试数据（可选）**
   ```sql
   -- 执行 cleanup_old_trades.sql
   DELETE FROM trades WHERE external_order_id IS NULL;
   ```

4. **验证迁移**
   ```sql
   -- 检查新字段是否存在
   SELECT column_name, data_type 
   FROM information_schema.columns 
   WHERE table_name = 'trades' 
   AND column_name IN ('entry_time', 'mae', 'mfe', 'holding_time_seconds');
   ```

### 场景3：清理测试数据

执行 `cleanup_old_trades.sql` 或直接运行：

```sql
-- 查看要删除的数据
SELECT * FROM trades WHERE external_order_id IS NULL;

-- 确认后删除
DELETE FROM trades WHERE external_order_id IS NULL;

-- 验证
SELECT COUNT(*) FROM trades WHERE external_order_id IS NULL;
-- 应该返回 0
```

## 🔧 常用维护命令

### 检查数据完整性

```sql
-- 检查所有表的记录数
SELECT 
  'accounts' as table_name, COUNT(*) as records FROM accounts
UNION ALL
SELECT 'trades', COUNT(*) FROM trades
UNION ALL
SELECT 'journal_notes', COUNT(*) FROM journal_notes
UNION ALL
SELECT 'user_preferences', COUNT(*) FROM user_preferences;
```

### 计算最大回撤

```sql
-- 使用内置函数
SELECT * FROM calculate_max_drawdown();

-- 或针对特定账户
SELECT * FROM calculate_max_drawdown('00000000-0000-0000-0000-000000000001'::UUID);
```

### 更新账户余额

```sql
-- 手动重新计算账户余额
UPDATE accounts 
SET current_balance = initial_balance + (
  SELECT COALESCE(SUM(pnl_net), 0) 
  FROM trades 
  WHERE account_id = accounts.id AND status = 'closed'
)
WHERE id = '00000000-0000-0000-0000-000000000001';
```

### 查看今日交易统计

```sql
SELECT 
  COUNT(*) as 交易数,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as 盈利笔数,
  ROUND(SUM(pnl_net), 2) as 净盈亏,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) || '%' as 胜率
FROM trades
WHERE DATE(entry_time) = CURRENT_DATE AND status = 'closed';
```

### 导出交易记录

```sql
-- 导出最近30天的交易
SELECT 
  entry_time as 进场时间,
  symbol as 品种,
  side as 方向,
  entry_price as 开仓价,
  exit_price as 平仓价,
  quantity as 手数,
  pnl_net as 净盈亏,
  holding_time_seconds / 3600.0 as 持仓小时,
  strategies as 策略,
  notes as 备注
FROM trades
WHERE entry_time >= CURRENT_DATE - INTERVAL '30 days'
  AND status = 'closed'
ORDER BY entry_time DESC;
```

## 🐛 故障排查

### 问题1：仪表盘不显示数据

**检查：**
```sql
SELECT COUNT(*) FROM trades WHERE status = 'closed';
```

**解决：**
- 如果返回0，说明没有已平仓的交易
- 确保 MT5 同步正常工作
- 检查 `external_order_id` 不为 NULL

### 问题2：交易日志日历没有数据

**检查：**
```sql
SELECT trade_date, total_trades, total_pnl 
FROM daily_trade_stats 
ORDER BY trade_date DESC 
LIMIT 7;
```

**解决：**
- 确保 trades 表有 `entry_time` 字段
- 运行迁移脚本更新数据

### 问题3：重复的交易记录

**检查：**
```sql
SELECT external_order_id, COUNT(*) as count
FROM trades
WHERE external_order_id IS NOT NULL
GROUP BY external_order_id
HAVING COUNT(*) > 1;
```

**解决：**
```sql
-- 保留最新的记录，删除旧的
WITH duplicates AS (
  SELECT id, 
    ROW_NUMBER() OVER (PARTITION BY external_order_id ORDER BY created_at DESC) as rn
  FROM trades
  WHERE external_order_id IS NOT NULL
)
DELETE FROM trades
WHERE id IN (SELECT id FROM duplicates WHERE rn > 1);
```

### 问题4：账户余额不准确

**重新计算：**
```sql
UPDATE accounts 
SET current_balance = initial_balance + (
  SELECT COALESCE(SUM(pnl_net), 0) 
  FROM trades 
  WHERE account_id = accounts.id AND status = 'closed'
);

-- 验证
SELECT 
  name,
  initial_balance as 初始本金,
  current_balance as 当前余额,
  current_balance - initial_balance as 累计盈亏
FROM accounts;
```

## 📊 性能优化建议

1. **定期清理旧数据**
   ```sql
   -- 删除1年前的测试数据
   DELETE FROM trades 
   WHERE source = 'manual' 
     AND created_at < CURRENT_DATE - INTERVAL '1 year'
     AND status = 'closed';
   ```

2. **重建索引**（如果查询变慢）
   ```sql
   REINDEX TABLE trades;
   REINDEX TABLE journal_notes;
   ```

3. **分析表统计信息**
   ```sql
   ANALYZE trades;
   ANALYZE journal_notes;
   ANALYZE accounts;
   ```

## 🔗 相关资源

- [Supabase 文档](https://supabase.com/docs)
- [PostgreSQL 数据类型](https://www.postgresql.org/docs/current/datatype.html)
- [SQL 性能优化](https://wiki.postgresql.org/wiki/Performance_Optimization)

## 📝 更新日志

### v2.0.0 (2025-11-25)
- ✨ 新增 `user_preferences` 表
- ✨ 增强 `trades` 表字段（MAE/MFE、风险管理）
- ✨ 增强 `journal_notes` 表字段（情绪追踪、市场观察）
- ✨ 新增多个分析视图
- ✨ 添加自动触发器（持有时间、余额更新）
- 🐛 修复前端重复插入数据问题
- 📚 完善文档和示例

### v1.0.0 (2025-11-21)
- 🎉 初始版本
- ✅ 基本的 accounts、trades、journal_notes 表
- ✅ RLS 策略配置
- ✅ 基础视图和索引

