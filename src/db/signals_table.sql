-- 创建信号表
CREATE TABLE IF NOT EXISTS signals (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  
  symbol VARCHAR(20) NOT NULL,
  action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
  price DECIMAL(15, 5),
  sl DECIMAL(15, 5),
  tp DECIMAL(15, 5),
  
  status VARCHAR(20) DEFAULT 'new' CHECK (status IN ('new', 'viewed', 'processed', 'ignored', 'expired')),
  source VARCHAR(50) DEFAULT 'mt5_indicator',
  raw_data JSONB,
  comment TEXT
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

-- RLS
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Enable all for authenticated users" ON signals;
CREATE POLICY "Enable all for authenticated users" ON signals FOR ALL USING (true);

-- 触发器
DROP TRIGGER IF EXISTS update_signals_updated_at ON signals;
CREATE TRIGGER update_signals_updated_at 
  BEFORE UPDATE ON signals
  FOR EACH ROW 
  EXECUTE FUNCTION update_updated_at_column();

