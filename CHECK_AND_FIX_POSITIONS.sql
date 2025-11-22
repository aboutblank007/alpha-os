-- ============================================
-- 检查和修复持仓订单问题
-- ============================================

-- 第一步：检查所有交易记录
SELECT 
  id,
  symbol,
  side,
  entry_price,
  exit_price,
  quantity,
  status,
  created_at,
  pnl_net
FROM trades
ORDER BY created_at DESC
LIMIT 20;

-- 第二步：检查持仓订单（status = 'open'）
SELECT 
  symbol as 品种,
  side as 方向,
  entry_price as 开仓价,
  quantity as 数量,
  pnl_net as 浮动盈亏,
  status as 状态,
  created_at as 开仓时间
FROM trades
WHERE status = 'open'
ORDER BY created_at DESC;

-- 第三步：如果您的TradingView持仓应该显示但没有显示，执行下面的修复

-- 修复方式1: 将特定的已平仓订单改为持仓
-- 例如：XAUUSD 和 USDJPY
-- UPDATE trades 
-- SET status = 'open', exit_price = NULL
-- WHERE symbol IN ('XAUUSD', 'USDJPY') 
-- AND created_at > NOW() - INTERVAL '1 day'
-- AND status = 'closed';

-- 修复方式2: 手动创建测试持仓（根据您的实际持仓）
-- 请根据您在TradingView中看到的持仓修改下面的数据
INSERT INTO trades (
  account_id,
  symbol,
  side,
  entry_price,
  quantity,
  pnl_net,
  status,
  created_at,
  external_order_id
) VALUES
  -- XAUUSD 持仓（买入）
  (
    '00000000-0000-0000-0000-000000000001',
    'XAUUSD',
    'buy',
    4087.67,      -- 您的开仓价
    0.02,         -- 您的数量
    1.00,         -- 当前浮动盈亏（会自动更新）
    'open',       -- 重要：必须是 'open'
    NOW(),
    'TV-XAUUSD-' || EXTRACT(EPOCH FROM NOW())::TEXT
  ),
  -- USDJPY 持仓（做空）
  (
    '00000000-0000-0000-0000-000000000001',
    'USDJPY',
    'sell',
    156.799,      -- 您的开仓价
    0.2,          -- 您的数量（0.2手）
    3.19,         -- 当前浮动盈亏（2000点 * 0.2手的实际盈亏）
    'open',       -- 重要：必须是 'open'
    NOW(),
    'TV-USDJPY-' || EXTRACT(EPOCH FROM NOW())::TEXT
  )
ON CONFLICT (external_order_id) DO NOTHING;

-- 第四步：验证持仓已创建
SELECT 
  symbol as 品种,
  side as 方向,
  entry_price as 开仓价,
  quantity as 数量,
  pnl_net as 浮动盈亏,
  status as 状态,
  DATE(created_at) as 日期
FROM trades
WHERE status = 'open'
ORDER BY created_at DESC;

-- 第五步：检查Chrome扩展同步的数据
-- 如果使用Chrome扩展同步，检查最近同步的数据
SELECT 
  symbol,
  side,
  status,
  created_at,
  notes
FROM trades
WHERE notes LIKE '%扩展%' OR notes LIKE '%extension%' OR notes LIKE '%Auto-synced%'
ORDER BY created_at DESC
LIMIT 10;

-- ============================================
-- 常见问题诊断
-- ============================================

-- 问题1: 持仓数量统计
SELECT 
  status,
  COUNT(*) as 数量
FROM trades
GROUP BY status;

-- 问题2: 今日交易统计
SELECT 
  status,
  COUNT(*) as 数量,
  SUM(pnl_net) as 总盈亏
FROM trades
WHERE DATE(created_at) = CURRENT_DATE
GROUP BY status;

-- 问题3: 检查是否有exit_price但status仍为open的异常数据
SELECT 
  symbol,
  side,
  entry_price,
  exit_price,
  status,
  created_at
FROM trades
WHERE exit_price IS NOT NULL AND status = 'open'
ORDER BY created_at DESC;

-- 修复异常数据（如果有）
-- UPDATE trades 
-- SET status = 'closed' 
-- WHERE exit_price IS NOT NULL AND status = 'open';

-- ============================================
-- 快速测试持仓显示
-- ============================================

-- 创建一笔简单的测试持仓
INSERT INTO trades (
  account_id,
  symbol,
  side,
  entry_price,
  quantity,
  pnl_net,
  status
) VALUES (
  '00000000-0000-0000-0000-000000000001',
  'EURUSD',
  'buy',
  1.0850,
  0.01,
  0.50,
  'open'
);

-- 刷新浏览器页面，应该能在仪表板看到这笔持仓

