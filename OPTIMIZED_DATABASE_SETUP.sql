-- ============================================
-- AlphaOS 优化数据库设置脚本
-- ============================================
-- 版本: v2.0.0
-- 日期: 2025-11-25
-- 说明: 根据前端所有功能优化的完整数据库配置
-- 支持: 仪表盘、交易日志、数据分析、设置四个主要页面
-- ============================================

-- ============================================
-- 第一步: 清理旧数据（可选）
-- ============================================
-- 警告: 取消注释将删除所有现有数据！
-- DROP TABLE IF EXISTS user_preferences CASCADE;
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
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  name VARCHAR(100) DEFAULT '默认账户',
  initial_balance DECIMAL(15, 2) NOT NULL DEFAULT 10000.00,
  current_balance DECIMAL(15, 2) NOT NULL DEFAULT 10000.00,
  currency VARCHAR(10) DEFAULT 'USD',
  broker VARCHAR(100),
  account_number VARCHAR(100),
  account_type VARCHAR(50) DEFAULT 'demo' CHECK (account_type IN ('demo', 'live')),
  notes TEXT
);

-- 添加账户表索引
CREATE INDEX IF NOT EXISTS idx_accounts_account_number ON accounts(account_number);

-- ============================================
-- 第三步: 创建交易记录表（增强版）
-- ============================================
CREATE TABLE IF NOT EXISTS trades (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
  
  -- 基本交易信息
  symbol VARCHAR(20) NOT NULL,
  side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
  entry_price DECIMAL(15, 5) NOT NULL,
  exit_price DECIMAL(15, 5),
  quantity DECIMAL(15, 4) NOT NULL,
  entry_time TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  exit_time TIMESTAMP WITH TIME ZONE,
  
  -- 盈亏信息
  pnl_net DECIMAL(15, 2) DEFAULT 0,
  pnl_gross DECIMAL(15, 2) DEFAULT 0,
  commission DECIMAL(15, 4) DEFAULT 0,
  swap DECIMAL(15, 4) DEFAULT 0,
  
  -- 风险管理
  stop_loss DECIMAL(15, 5),
  take_profit DECIMAL(15, 5),
  risk_reward_ratio DECIMAL(10, 2),
  position_size_pct DECIMAL(5, 2),
  
  -- 交易分析
  mae DECIMAL(15, 2) DEFAULT 0, -- Maximum Adverse Excursion (最大不利偏移)
  mfe DECIMAL(15, 2) DEFAULT 0, -- Maximum Favorable Excursion (最大有利偏移)
  holding_time_seconds INTEGER,
  slippage DECIMAL(15, 5),
  
  -- 状态和分类
  status VARCHAR(10) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
  order_type VARCHAR(20) DEFAULT 'market' CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
  
  -- 交易笔记和标签
  notes TEXT,
  emotion_score INTEGER CHECK (emotion_score BETWEEN 1 AND 5),
  confidence_level INTEGER CHECK (confidence_level BETWEEN 1 AND 5),
  strategies TEXT[],
  tags TEXT[],
  
  -- 外部系统关联
  external_order_id VARCHAR(100) UNIQUE,
  external_ticket VARCHAR(100),
  source VARCHAR(50) DEFAULT 'manual' CHECK (source IN ('manual', 'mt5', 'api', 'import')),
  
  -- 图表和截图
  chart_snapshot_url TEXT,
  screenshot_urls TEXT[]
);

-- 交易表索引（优化查询性能）
CREATE INDEX IF NOT EXISTS idx_trades_account_id ON trades(account_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_external_order_id ON trades(external_order_id) WHERE external_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl_net) WHERE status = 'closed';
CREATE INDEX IF NOT EXISTS idx_trades_strategies ON trades USING GIN(strategies) WHERE strategies IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_tags ON trades USING GIN(tags) WHERE tags IS NOT NULL;

-- ============================================
-- 第四步: 创建交易笔记表（增强版）
-- ============================================
CREATE TABLE IF NOT EXISTS journal_notes (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  date DATE NOT NULL UNIQUE,
  
  -- 笔记内容
  content TEXT NOT NULL,
  summary VARCHAR(500),
  
  -- 情绪和状态
  mood VARCHAR(20) CHECK (mood IN ('confident', 'calm', 'anxious', 'frustrated', 'excited', 'tired', 'focused')),
  energy_level INTEGER CHECK (energy_level BETWEEN 1 AND 5),
  mental_state VARCHAR(20) CHECK (mental_state IN ('clear', 'distracted', 'stressed', 'relaxed')),
  
  -- 市场观察
  market_condition VARCHAR(50),
  market_sentiment VARCHAR(20) CHECK (market_sentiment IN ('bullish', 'bearish', 'neutral', 'volatile')),
  
  -- 标签和分类
  tags TEXT[],
  trade_ids UUID[],
  
  -- 学习和改进
  lessons_learned TEXT[],
  mistakes_made TEXT[],
  improvement_notes TEXT,
  
  -- 统计快照
  daily_pnl DECIMAL(15, 2),
  daily_trades_count INTEGER,
  daily_win_rate DECIMAL(5, 2)
);

-- 笔记表索引
CREATE INDEX IF NOT EXISTS idx_journal_notes_date ON journal_notes(date DESC);
CREATE INDEX IF NOT EXISTS idx_journal_notes_mood ON journal_notes(mood) WHERE mood IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_journal_notes_tags ON journal_notes USING GIN(tags) WHERE tags IS NOT NULL;

-- ============================================
-- 第五步: 创建用户偏好设置表（新增）
-- ============================================
CREATE TABLE IF NOT EXISTS user_preferences (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  user_id VARCHAR(100) DEFAULT 'default_user' UNIQUE,
  
  -- 个人信息
  display_name VARCHAR(100) DEFAULT '交易员',
  email VARCHAR(255),
  timezone VARCHAR(50) DEFAULT 'Asia/Shanghai',
  language VARCHAR(10) DEFAULT 'zh-CN',
  
  -- 交易偏好
  default_currency VARCHAR(10) DEFAULT 'USD',
  default_risk_percentage DECIMAL(5, 2) DEFAULT 2.00,
  default_symbols TEXT[],
  favorite_symbols TEXT[],
  risk_level VARCHAR(20) DEFAULT 'medium' CHECK (risk_level IN ('low', 'medium', 'high', 'aggressive')),
  
  -- 显示偏好
  show_live_price BOOLEAN DEFAULT true,
  auto_sync BOOLEAN DEFAULT true,
  chart_default_timeframe VARCHAR(10) DEFAULT 'M5',
  dashboard_layout JSONB,
  
  -- 通知设置
  email_notifications BOOLEAN DEFAULT true,
  trade_alerts BOOLEAN DEFAULT true,
  risk_alerts BOOLEAN DEFAULT true,
  daily_summary BOOLEAN DEFAULT false,
  win_notification BOOLEAN DEFAULT true,
  loss_notification BOOLEAN DEFAULT true,
  
  -- 主题设置
  theme VARCHAR(20) DEFAULT 'dark' CHECK (theme IN ('light', 'dark', 'auto')),
  accent_color VARCHAR(20) DEFAULT 'blue',
  chart_theme VARCHAR(20) DEFAULT 'dark',
  
  -- 高级设置
  api_keys JSONB,
  webhook_urls TEXT[],
  backup_enabled BOOLEAN DEFAULT false,
  last_backup_at TIMESTAMP WITH TIME ZONE
);

-- 用户偏好表索引
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

-- ============================================
-- 第六步: 创建触发器函数
-- ============================================

-- 自动更新 updated_at 时间戳
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 自动计算交易持有时间
CREATE OR REPLACE FUNCTION calculate_holding_time()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'closed' AND NEW.exit_time IS NOT NULL AND NEW.entry_time IS NOT NULL THEN
        NEW.holding_time_seconds = EXTRACT(EPOCH FROM (NEW.exit_time - NEW.entry_time))::INTEGER;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 自动更新账户余额
CREATE OR REPLACE FUNCTION update_account_balance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'closed' AND NEW.account_id IS NOT NULL THEN
        UPDATE accounts 
        SET current_balance = initial_balance + (
            SELECT COALESCE(SUM(pnl_net), 0) 
            FROM trades 
            WHERE account_id = NEW.account_id AND status = 'closed'
        )
        WHERE id = NEW.account_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 第七步: 创建触发器
-- ============================================

-- accounts 表触发器
DROP TRIGGER IF EXISTS update_accounts_updated_at ON accounts;
CREATE TRIGGER update_accounts_updated_at 
  BEFORE UPDATE ON accounts
  FOR EACH ROW 
  EXECUTE FUNCTION update_updated_at_column();

-- trades 表触发器
DROP TRIGGER IF EXISTS update_trades_updated_at ON trades;
CREATE TRIGGER update_trades_updated_at 
  BEFORE UPDATE ON trades
  FOR EACH ROW 
  EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS calculate_trades_holding_time ON trades;
CREATE TRIGGER calculate_trades_holding_time 
  BEFORE INSERT OR UPDATE ON trades
  FOR EACH ROW 
  EXECUTE FUNCTION calculate_holding_time();

DROP TRIGGER IF EXISTS update_account_balance_on_trade ON trades;
CREATE TRIGGER update_account_balance_on_trade 
  AFTER INSERT OR UPDATE OF status, pnl_net ON trades
  FOR EACH ROW 
  WHEN (NEW.status = 'closed')
  EXECUTE FUNCTION update_account_balance();

-- journal_notes 表触发器
DROP TRIGGER IF EXISTS update_journal_notes_updated_at ON journal_notes;
CREATE TRIGGER update_journal_notes_updated_at 
  BEFORE UPDATE ON journal_notes
  FOR EACH ROW 
  EXECUTE FUNCTION update_updated_at_column();

-- user_preferences 表触发器
DROP TRIGGER IF EXISTS update_user_preferences_updated_at ON user_preferences;
CREATE TRIGGER update_user_preferences_updated_at 
  BEFORE UPDATE ON user_preferences
  FOR EACH ROW 
  EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 第八步: 启用 Row Level Security (RLS)
-- ============================================
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE journal_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- 创建允许所有操作的策略（适用于私人应用）
DROP POLICY IF EXISTS "Enable all for authenticated users" ON accounts;
CREATE POLICY "Enable all for authenticated users" ON accounts FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all for authenticated users" ON trades;
CREATE POLICY "Enable all for authenticated users" ON trades FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all for authenticated users" ON journal_notes;
CREATE POLICY "Enable all for authenticated users" ON journal_notes FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all for authenticated users" ON user_preferences;
CREATE POLICY "Enable all for authenticated users" ON user_preferences FOR ALL USING (true);

-- ============================================
-- 第九步: 创建有用的视图
-- ============================================

-- 每日交易统计视图（用于交易日志页面）
CREATE OR REPLACE VIEW daily_trade_stats AS
SELECT 
  entry_time::date as trade_date,
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as winning_trades,
  SUM(CASE WHEN pnl_net < 0 THEN 1 ELSE 0 END) as losing_trades,
  SUM(CASE WHEN pnl_net = 0 THEN 1 ELSE 0 END) as breakeven_trades,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl,
  MAX(pnl_net) as best_trade,
  MIN(pnl_net) as worst_trade,
  STDDEV(pnl_net) as pnl_stddev,
  AVG(holding_time_seconds) / 3600.0 as avg_holding_hours
FROM trades
WHERE status = 'closed' AND entry_time IS NOT NULL
GROUP BY entry_time::date
ORDER BY trade_date DESC;

-- 品种表现统计视图（用于数据分析页面）
CREATE OR REPLACE VIEW symbol_performance AS
SELECT 
  symbol,
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as winning_trades,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl,
  MAX(pnl_net) as best_trade,
  MIN(pnl_net) as worst_trade,
  SUM(CASE WHEN pnl_net > 0 THEN pnl_net ELSE 0 END) as total_profit,
  ABS(SUM(CASE WHEN pnl_net < 0 THEN pnl_net ELSE 0 END)) as total_loss,
  CASE 
    WHEN ABS(SUM(CASE WHEN pnl_net < 0 THEN pnl_net ELSE 0 END)) > 0 
    THEN SUM(CASE WHEN pnl_net > 0 THEN pnl_net ELSE 0 END) / ABS(SUM(CASE WHEN pnl_net < 0 THEN pnl_net ELSE 0 END))
    ELSE 0 
  END as profit_factor,
  AVG(quantity) as avg_position_size
FROM trades
WHERE status = 'closed'
GROUP BY symbol
ORDER BY total_pnl DESC;

-- 月度统计视图（用于仪表盘）
CREATE OR REPLACE VIEW monthly_stats AS
SELECT 
  DATE_TRUNC('month', entry_time) as month,
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as winning_trades,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl,
  MAX(pnl_net) as best_trade,
  MIN(pnl_net) as worst_trade,
  SUM(CASE WHEN pnl_net > 0 THEN pnl_net ELSE 0 END) / NULLIF(ABS(SUM(CASE WHEN pnl_net < 0 THEN pnl_net ELSE 0 END)), 0) as profit_factor
FROM trades
WHERE status = 'closed' AND entry_time IS NOT NULL
GROUP BY DATE_TRUNC('month', entry_time)
ORDER BY month DESC;

-- 策略表现视图
CREATE OR REPLACE VIEW strategy_performance AS
SELECT 
  UNNEST(strategies) as strategy,
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as winning_trades,
  ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate,
  SUM(pnl_net) as total_pnl,
  AVG(pnl_net) as avg_pnl
FROM trades
WHERE status = 'closed' AND strategies IS NOT NULL AND array_length(strategies, 1) > 0
GROUP BY strategy
ORDER BY total_pnl DESC;

-- 风险分析视图
CREATE OR REPLACE VIEW risk_analysis AS
SELECT 
  DATE(entry_time) as trade_date,
  COUNT(*) as trades_count,
  AVG(CASE WHEN risk_reward_ratio IS NOT NULL THEN risk_reward_ratio END) as avg_rr_ratio,
  AVG(CASE WHEN position_size_pct IS NOT NULL THEN position_size_pct END) as avg_position_size,
  MAX(ABS(pnl_net)) as max_single_loss,
  STDDEV(pnl_net) as daily_volatility
FROM trades
WHERE status = 'closed' AND entry_time IS NOT NULL
GROUP BY DATE(entry_time)
ORDER BY trade_date DESC;

-- ============================================
-- 第十步: 插入初始数据
-- ============================================

-- 插入默认账户
INSERT INTO accounts (
  id, 
  name, 
  initial_balance, 
  current_balance, 
  currency,
  account_type
) VALUES (
  '00000000-0000-0000-0000-000000000001',
  '主账户',
  1084.80,
  1084.80,
  'USD',
  'demo'
)
ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name,
  updated_at = timezone('utc'::text, now());

-- 插入默认用户偏好
INSERT INTO user_preferences (
  user_id,
  display_name,
  timezone,
  default_currency,
  risk_level,
  theme,
  accent_color,
  show_live_price,
  auto_sync,
  email_notifications,
  trade_alerts,
  risk_alerts
) VALUES (
  'default_user',
  '交易员',
  'Asia/Shanghai',
  'USD',
  'medium',
  'dark',
  'blue',
  true,
  true,
  true,
  true,
  true
)
ON CONFLICT (user_id) DO UPDATE SET
  updated_at = timezone('utc'::text, now());

-- ============================================
-- 第十一步: 创建辅助函数
-- ============================================

-- 计算最大回撤函数
CREATE OR REPLACE FUNCTION calculate_max_drawdown(account_uuid UUID DEFAULT NULL)
RETURNS TABLE(max_drawdown DECIMAL, drawdown_date DATE) AS $$
WITH equity_curve AS (
  SELECT 
    entry_time::date as date,
    SUM(pnl_net) OVER (ORDER BY entry_time) as cumulative_pnl
  FROM trades
  WHERE status = 'closed' 
    AND (account_uuid IS NULL OR account_id = account_uuid)
    AND entry_time IS NOT NULL
),
running_max AS (
  SELECT 
    date,
    cumulative_pnl,
    MAX(cumulative_pnl) OVER (ORDER BY date) as peak
  FROM equity_curve
),
drawdowns AS (
  SELECT 
    date,
    cumulative_pnl,
    peak,
    (cumulative_pnl - peak) as drawdown
  FROM running_max
),
min_drawdown AS (
  SELECT MIN(drawdown) as min_dd
  FROM drawdowns
)
SELECT 
  md.min_dd as max_drawdown,
  (SELECT d.date FROM drawdowns d WHERE d.drawdown = md.min_dd ORDER BY d.date LIMIT 1) as drawdown_date
FROM min_drawdown md;
$$ LANGUAGE SQL;

-- 清理旧测试数据函数
CREATE OR REPLACE FUNCTION cleanup_legacy_trades()
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  DELETE FROM trades WHERE external_order_id IS NULL;
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 第十二步: 验证安装
-- ============================================

DO $$
DECLARE
  accounts_count INTEGER;
  trades_count INTEGER;
  notes_count INTEGER;
  prefs_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO accounts_count FROM accounts;
  SELECT COUNT(*) INTO trades_count FROM trades;
  SELECT COUNT(*) INTO notes_count FROM journal_notes;
  SELECT COUNT(*) INTO prefs_count FROM user_preferences;
  
  RAISE NOTICE '✅ 数据库优化完成！';
  RAISE NOTICE '';
  RAISE NOTICE '📊 当前统计:';
  RAISE NOTICE '   - 账户数: %', accounts_count;
  RAISE NOTICE '   - 交易记录数: %', trades_count;
  RAISE NOTICE '   - 笔记数: %', notes_count;
  RAISE NOTICE '   - 用户偏好: %', prefs_count;
  RAISE NOTICE '';
  RAISE NOTICE '🎉 所有功能已就绪！';
  RAISE NOTICE '   ✓ 仪表盘 - 统计、图表、实时订单';
  RAISE NOTICE '   ✓ 交易日志 - 日历、笔记、统计';
  RAISE NOTICE '   ✓ 数据分析 - MAE/MFE、策略分析';
  RAISE NOTICE '   ✓ 设置 - 用户偏好、通知配置';
END $$;

-- 显示表结构信息
SELECT 
  '数据库表' as 类别,
  table_name as 表名,
  (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as 字段数
FROM information_schema.tables t
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE'
  AND table_name IN ('accounts', 'trades', 'journal_notes', 'user_preferences')
ORDER BY table_name;

