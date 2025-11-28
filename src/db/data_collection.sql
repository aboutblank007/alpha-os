-- Table for storing comprehensive signal data for offline training
-- Features are logged at the time of signal generation (snapshot)
-- Outcomes (profit, mae, mfe) are updated later by a separate process

-- Drop table if it exists to ensure clean slate with correct schema
DROP TABLE IF EXISTS training_signals;

CREATE TABLE IF NOT EXISTS training_signals (
    -- 主键标识
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id TEXT UNIQUE,          -- 唯一信号标识 (Symbol + Time)
    symbol TEXT NOT NULL,           -- 交易品种
    action TEXT NOT NULL,           -- 操作类型
    
    -- 时间信息
    timestamp TIMESTAMPTZ NOT NULL, -- 信号时间
    broker_time TIMESTAMPTZ,        -- 经纪商时间
    execution_time TIMESTAMPTZ,     -- 执行时间
    exit_time TIMESTAMPTZ,          -- 平仓时间
    
    -- 交易执行信息
    order_id TEXT,                  -- MT5订单号
    position_id TEXT,               -- MT5持仓号
    executed BOOLEAN DEFAULT FALSE, -- 是否执行
    execution_price DECIMAL,        -- 执行价格
    execution_spread DECIMAL,       -- 执行点差
    
    -- 价格信息
    signal_price DECIMAL NOT NULL,  -- 信号价格
    sl DECIMAL NOT NULL,            -- 止损价
    tp DECIMAL NOT NULL,            -- 止盈价
    exit_price DECIMAL,             -- 平仓价
    
    -- 技术指标特征
    ema_short DECIMAL,              -- EMA短期
    ema_long DECIMAL,               -- EMA长期
    atr DECIMAL,                    -- ATR值
    adx DECIMAL,                    -- ADX强度
    center DECIMAL,                 -- 中心线
    
    -- 过滤状态特征
    distance_ok BOOLEAN,            -- 距离过滤
    slope_ok BOOLEAN,               -- 斜率过滤
    trend_filter_ok BOOLEAN,        -- 趋势过滤
    htf_trend_ok BOOLEAN,           -- HTF趋势
    volatility_ok BOOLEAN,          -- 波动过滤
    chop_ok BOOLEAN,                -- 震荡过滤
    spread_ok BOOLEAN,              -- 价差过滤
    
    -- 扩展特征
    bars_since_last INTEGER,        -- 距离上次信号
    trend_direction INTEGER,        -- 趋势方向
    ema_cross_event INTEGER,        -- EMA交叉事件
    ema_spread DECIMAL,             -- EMA价差
    atr_percent DECIMAL,            -- ATR百分比
    reclaim_state INTEGER,          -- Reclaim状态
    is_reclaim_signal BOOLEAN,      -- 是否Reclaim
    price_vs_center DECIMAL,        -- 价格vs中心
    cloud_width DECIMAL,            -- 云层宽度
    
    -- 交易结果
    result_profit DECIMAL,          -- 盈亏结果
    result_mae DECIMAL,             -- 最大不利偏移
    result_mfe DECIMAL,             -- 最大有利偏移
    result_win BOOLEAN,             -- 是否盈利
    exit_reason TEXT,               -- 退出原因
    holding_period INTERVAL,        -- 持仓时间
    
    -- 系统字段
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_signals_symbol_time ON training_signals(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_training_signals_signal_id ON training_signals(signal_id);
CREATE INDEX IF NOT EXISTS idx_training_signals_position_id ON training_signals(position_id);
