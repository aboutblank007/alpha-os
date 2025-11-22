-- ============================================
-- AlphaOS 完整数据库设置脚本
-- ============================================
-- 版本: v1.0.0
-- 日期: 2025-11-21
-- 说明: 此脚本将创建所有必要的表、索引、触发器和示例数据
-- ============================================

-- ============================================
-- 第一步: 清理旧数据（可选，首次安装请跳过）
-- ============================================
-- 警告: 取消注释下面的命令将删除所有现有数据！
-- DROP TABLE IF EXISTS journal_notes CASCADE;
-- DROP TABLE IF EXISTS trades CASCADE;
-- DROP TABLE IF EXISTS accounts CASCADE;
-- DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- ============================================
-- 第二步: 创建账户表
-- ============================================
CREATE TABLE IF NOT EXISTS accounts (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  name VARCHAR(100) DEFAULT '默认账户',
  initial_balance DECIMAL(12, 2) NOT NULL DEFAULT 10000.00,
  current_balance DECIMAL(12, 2) NOT NULL DEFAULT 10000.00,
  currency VARCHAR(10) DEFAULT 'USD'
);

-- ============================================
-- 第三步: 创建交易记录表
-- ============================================
CREATE TABLE IF NOT EXISTS trades (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
  symbol VARCHAR(20) NOT NULL,
  side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
  entry_price DECIMAL(12, 5) NOT NULL,
  exit_price DECIMAL(12, 5),
  quantity DECIMAL(12, 2) NOT NULL,
  pnl_net DECIMAL(12, 2) DEFAULT 0,
  pnl_gross DECIMAL(12, 2) DEFAULT 0,
  commission DECIMAL(12, 2) DEFAULT 0,
  swap DECIMAL(12, 2) DEFAULT 0,
  status VARCHAR(10) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
  notes TEXT,
  emotion_score INTEGER CHECK (emotion_score BETWEEN 1 AND 5),
  strategies TEXT[],
  external_order_id VARCHAR(50) UNIQUE
);

-- ============================================
-- 第四步: 创建交易笔记表
-- ============================================
CREATE TABLE IF NOT EXISTS journal_notes (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  date DATE NOT NULL UNIQUE,
  content TEXT NOT NULL,
  mood VARCHAR(20),
  tags TEXT[]
);

-- ============================================
-- 第五步: 创建索引以提升查询性能
-- ============================================

-- trades 表索引
CREATE INDEX IF NOT EXISTS idx_trades_account_id ON trades(account_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_external_order_id 
  ON trades(external_order_id) 
  WHERE external_order_id IS NOT NULL;

-- journal_notes 表索引
CREATE INDEX IF NOT EXISTS idx_journal_notes_date ON journal_notes(date DESC);

-- ============================================
-- 第六步: 创建自动更新时间戳的触发器函数
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 第七步: 为 journal_notes 表创建触发器
-- ============================================
DROP TRIGGER IF EXISTS update_journal_notes_updated_at ON journal_notes;
CREATE TRIGGER update_journal_notes_updated_at 
  BEFORE UPDATE ON journal_notes
  FOR EACH ROW 
  EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 第八步: 启用 Row Level Security (RLS)
-- ============================================
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE journal_notes ENABLE ROW LEVEL SECURITY;

-- 创建允许所有操作的策略（适用于私人应用）
DROP POLICY IF EXISTS "Enable all for authenticated users" ON accounts;
CREATE POLICY "Enable all for authenticated users" ON accounts
  FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all for authenticated users" ON trades;
CREATE POLICY "Enable all for authenticated users" ON trades
  FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all for authenticated users" ON journal_notes;
CREATE POLICY "Enable all for authenticated users" ON journal_notes
  FOR ALL USING (true);

-- ============================================
-- 第九步: 插入示例数据
-- ============================================

-- 插入默认账户
-- 注意: initial_balance 是您的初始本金 (1084.8)
-- current_balance 将通过 API 自动计算（初始本金 + 所有交易盈亏）

-- 插入交易记录（最近30天的数据）
-- 注意: 这些是示例数据，盈亏金额已调整为小额账户规模
-- 您可以删除这些示例数据，使用CSV导入您的真实交易记录
INSERT INTO trades (
  account_id, symbol, side, entry_price, exit_price, quantity, 
  pnl_net, commission, status, created_at, external_order_id
) VALUES
  -- 今日交易（2025-11-21）
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'buy', 1.0850, 1.0880, 0.01, 3.00, 0.02, 'closed', '2025-11-21 08:30:00+00', 'ORD-20251121-001'),
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'sell', 1.2650, 1.2620, 0.01, 4.50, 0.03, 'closed', '2025-11-21 10:15:00+00', 'ORD-20251121-002'),
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'buy', 149.50, 149.80, 0.01, 2.00, 0.02, 'closed', '2025-11-21 14:20:00+00', 'ORD-20251121-003'),
  ('00000000-0000-0000-0000-000000000001', 'XAUUSD', 'buy', 2680.00, NULL, 0.01, 1.50, 0.01, 'open', '2025-11-21 16:00:00+00', 'ORD-20251121-004'),
  
  -- 昨日交易（2025-11-20）
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'sell', 1.0870, 1.0840, 1.0, 300.00, 2.00, 'closed', '2025-11-20 09:00:00+00', 'ORD-20251120-001'),
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'buy', 1.2600, 1.2580, 1.0, -200.00, 2.00, 'closed', '2025-11-20 13:30:00+00', 'ORD-20251120-002'),
  
  -- 2025-11-19
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'sell', 150.00, 149.50, 1.5, 500.00, 3.00, 'closed', '2025-11-19 10:00:00+00', 'ORD-20251119-001'),
  ('00000000-0000-0000-0000-000000000001', 'XAUUSD', 'buy', 2670.00, 2690.00, 0.3, 600.00, 2.00, 'closed', '2025-11-19 15:00:00+00', 'ORD-20251119-002'),
  
  -- 2025-11-18
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'buy', 1.0820, 1.0860, 2.0, 800.00, 4.00, 'closed', '2025-11-18 08:00:00+00', 'ORD-20251118-001'),
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'sell', 1.2700, 1.2650, 1.0, 500.00, 2.00, 'closed', '2025-11-18 12:00:00+00', 'ORD-20251118-002'),
  
  -- 2025-11-15
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'buy', 148.50, 149.00, 2.0, 670.00, 4.00, 'closed', '2025-11-15 09:30:00+00', 'ORD-20251115-001'),
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'sell', 1.0900, 1.0880, 1.5, 300.00, 3.00, 'closed', '2025-11-15 14:00:00+00', 'ORD-20251115-002'),
  
  -- 2025-11-14
  ('00000000-0000-0000-0000-000000000001', 'XAUUSD', 'buy', 2650.00, 2665.00, 0.5, 750.00, 2.00, 'closed', '2025-11-14 10:00:00+00', 'ORD-20251114-001'),
  
  -- 2025-11-13
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'buy', 1.2550, 1.2530, 1.0, -200.00, 2.00, 'closed', '2025-11-13 11:00:00+00', 'ORD-20251113-001'),
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'sell', 1.0850, 1.0820, 2.0, 600.00, 4.00, 'closed', '2025-11-13 15:30:00+00', 'ORD-20251113-002'),
  
  -- 2025-11-12
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'sell', 150.50, 149.80, 1.5, 700.00, 3.00, 'closed', '2025-11-12 09:00:00+00', 'ORD-20251112-001'),
  ('00000000-0000-0000-0000-000000000001', 'XAUUSD', 'buy', 2640.00, 2630.00, 0.4, -400.00, 2.00, 'closed', '2025-11-12 13:00:00+00', 'ORD-20251112-002'),
  
  -- 2025-11-11
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'buy', 1.0800, 1.0830, 1.5, 450.00, 3.00, 'closed', '2025-11-11 10:30:00+00', 'ORD-20251111-001'),
  
  -- 2025-11-08
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'sell', 1.2750, 1.2700, 1.0, 500.00, 2.00, 'closed', '2025-11-08 14:00:00+00', 'ORD-20251108-001'),
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'buy', 148.00, 148.50, 2.0, 670.00, 4.00, 'closed', '2025-11-08 16:00:00+00', 'ORD-20251108-002'),
  
  -- 2025-11-07
  ('00000000-0000-0000-0000-000000000001', 'XAUUSD', 'buy', 2660.00, 2680.00, 0.5, 1000.00, 2.00, 'closed', '2025-11-07 09:00:00+00', 'ORD-20251107-001'),
  
  -- 2025-11-06
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'sell', 1.0880, 1.0850, 1.5, 450.00, 3.00, 'closed', '2025-11-06 11:00:00+00', 'ORD-20251106-001'),
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'buy', 1.2600, 1.2580, 1.0, -200.00, 2.00, 'closed', '2025-11-06 15:00:00+00', 'ORD-20251106-002'),
  
  -- 2025-11-05
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'sell', 151.00, 150.30, 1.5, 700.00, 3.00, 'closed', '2025-11-05 10:00:00+00', 'ORD-20251105-001'),
  
  -- 2025-11-04
  ('00000000-0000-0000-0000-000000000001', 'XAUUSD', 'buy', 2650.00, 2640.00, 0.3, -300.00, 2.00, 'closed', '2025-11-04 12:00:00+00', 'ORD-20251104-001'),
  ('00000000-0000-0000-0000-000000000001', 'EURUSD', 'buy', 1.0810, 1.0840, 2.0, 600.00, 4.00, 'closed', '2025-11-04 16:00:00+00', 'ORD-20251104-002'),
  
  -- 2025-11-01
  ('00000000-0000-0000-0000-000000000001', 'GBPUSD', 'sell', 1.2800, 1.2750, 1.0, 500.00, 2.00, 'closed', '2025-11-01 09:00:00+00', 'ORD-20251101-001'),
  ('00000000-0000-0000-0000-000000000001', 'USDJPY', 'buy', 147.50, 148.00, 2.0, 670.00, 4.00, 'closed', '2025-11-01 13:00:00+00', 'ORD-20251101-002')
ON CONFLICT (external_order_id) DO NOTHING;

-- 插入交易笔记示例
INSERT INTO journal_notes (date, content, mood, tags) VALUES
  (
    '2025-11-21',
    '今日交易表现出色！严格执行了交易计划，3笔交易全部盈利。\n\n**交易总结:**\n- EURUSD: 抓住了欧洲时段的突破机会\n- GBPUSD: 英镑回调后的反弹很及时\n- USDJPY: 美元走强趋势延续\n\n**心得:**\n保持纪律性是关键，不要贪心。止盈位设置合理。',
    'confident',
    ARRAY['趋势交易', '突破', '盈利日']
  ),
  (
    '2025-11-20',
    '今日表现一般，有一笔止损。\n\nGBPUSD的假突破被止损，但这是正确的风险管理。需要提高对假突破的识别能力。',
    'calm',
    ARRAY['止损', '风险管理']
  ),
  (
    '2025-11-19',
    '非常好的交易日！两笔交易都抓住了主要趋势。\n\n黄金的突破非常强劲，提前布局收获丰厚。继续保持这种对市场的敏感度。',
    'confident',
    ARRAY['趋势交易', '黄金', '大赚']
  ),
  (
    '2025-11-15',
    '今日休息观望，市场波动较大。\n\n等待更好的入场机会，不要因为无聊而强行交易。',
    'calm',
    ARRAY['休息', '观望']
  )
ON CONFLICT (date) DO NOTHING;

-- ============================================
-- 第十步: 创建有用的视图（可选）
-- ============================================

-- 每日交易统计视图
CREATE OR REPLACE VIEW daily_trade_stats AS
SELECT 
  DATE(created_at) as trade_date,
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as winning_trades,
  SUM(CASE WHEN pnl_net < 0 THEN 1 ELSE 0 END) as losing_trades,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl,
  MAX(pnl_net) as best_trade,
  MIN(pnl_net) as worst_trade
FROM trades
WHERE status = 'closed'
GROUP BY DATE(created_at)
ORDER BY trade_date DESC;

-- 品种表现统计视图
CREATE OR REPLACE VIEW symbol_performance AS
SELECT 
  symbol,
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as winning_trades,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl
FROM trades
WHERE status = 'closed'
GROUP BY symbol
ORDER BY total_pnl DESC;

-- ============================================
-- 验证安装
-- ============================================

-- 检查表是否创建成功
DO $$
DECLARE
  accounts_count INTEGER;
  trades_count INTEGER;
  notes_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO accounts_count FROM accounts;
  SELECT COUNT(*) INTO trades_count FROM trades;
  SELECT COUNT(*) INTO notes_count FROM journal_notes;
  
  RAISE NOTICE '✅ 安装验证:';
  RAISE NOTICE '   - 账户数: %', accounts_count;
  RAISE NOTICE '   - 交易记录数: %', trades_count;
  RAISE NOTICE '   - 笔记数: %', notes_count;
  RAISE NOTICE '';
  RAISE NOTICE '🎉 数据库设置完成！';
END $$;

-- 显示今日交易统计
SELECT 
  '今日交易统计' as title,
  COUNT(*) as 交易笔数,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as 盈利笔数,
  ROUND(SUM(pnl_net), 2) as 净盈亏,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) || '%' as 胜率
FROM trades
WHERE DATE(created_at) = CURRENT_DATE AND status = 'closed';

-- 显示最近的笔记
SELECT 
  '最近笔记' as title,
  date as 日期,
  mood as 心情,
  LEFT(content, 50) || '...' as 内容预览,
  array_length(tags, 1) as 标签数
FROM journal_notes
ORDER BY date DESC
LIMIT 5;

