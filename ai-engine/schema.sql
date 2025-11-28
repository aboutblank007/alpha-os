-- AlphaOS AI Trading System Schema
-- Based on ai-engine/alphaos.md

-- 1. Training Signals Table
CREATE TABLE IF NOT EXISTS training_signals (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id TEXT UNIQUE,          -- Unique Signal ID (Symbol_Timestamp)
    symbol TEXT NOT NULL,           -- Trading Symbol
    action TEXT NOT NULL,           -- Action (BUY/SELL/RECLAIM_BUY/RECLAIM_SELL)
    
    -- Time Information
    timestamp TIMESTAMPTZ NOT NULL, -- Signal Time
    broker_time TIMESTAMPTZ,        -- Broker Time
    execution_time TIMESTAMPTZ,     -- Execution Time
    exit_time TIMESTAMPTZ,          -- Exit Time
    
    -- Execution Information
    order_id TEXT,                  -- MT5 Order ID
    position_id TEXT,               -- MT5 Position ID
    executed BOOLEAN DEFAULT FALSE, -- Executed by AI?
    execution_price DECIMAL,        -- Execution Price
    execution_spread DECIMAL,       -- Execution Spread
    
    -- Price Information
    signal_price DECIMAL NOT NULL,  -- Signal Price
    sl DECIMAL NOT NULL,            -- Stop Loss
    tp DECIMAL NOT NULL,            -- Take Profit
    exit_price DECIMAL,             -- Exit Price
    
    -- Technical Indicators (Features)
    ema_short DECIMAL,              -- EMA Short
    ema_long DECIMAL,               -- EMA Long
    atr DECIMAL,                    -- ATR
    adx DECIMAL,                    -- ADX
    center DECIMAL,                 -- Center Line
    
    -- Filter States
    distance_ok BOOLEAN,            -- Distance Filter
    slope_ok BOOLEAN,               -- Slope Filter
    trend_filter_ok BOOLEAN,        -- Trend Filter
    htf_trend_ok BOOLEAN,           -- HTF Trend Filter
    volatility_ok BOOLEAN,          -- Volatility Filter
    chop_ok BOOLEAN,                -- Chop Filter
    spread_ok BOOLEAN,              -- Spread Filter
    
    -- Extended Features
    bars_since_last INTEGER,        -- Bars Since Last Signal
    trend_direction INTEGER,        -- Trend Direction (1=Up, 0=Down)
    ema_cross_event INTEGER,        -- EMA Cross Event
    ema_spread DECIMAL,             -- EMA Spread
    atr_percent DECIMAL,            -- ATR Percent
    reclaim_state INTEGER,          -- Reclaim State
    is_reclaim_signal BOOLEAN,      -- Is Reclaim Signal
    price_vs_center DECIMAL,        -- Price vs Center
    cloud_width DECIMAL,            -- Cloud Width
    
    -- Trade Results (Labels)
    result_profit DECIMAL,          -- Profit/Loss
    result_mae DECIMAL,             -- Max Adverse Excursion
    result_mfe DECIMAL,             -- Max Favorable Excursion
    result_win BOOLEAN,             -- Win/Loss
    exit_reason TEXT,               -- Exit Reason (TP/SL/Time/Manual)
    holding_period INTERVAL,        -- Holding Period
    
    -- System Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_training_signals_symbol ON training_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_training_signals_timestamp ON training_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_training_signals_executed ON training_signals(executed);
