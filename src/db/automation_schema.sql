-- Create automation_rules table
CREATE TABLE IF NOT EXISTS automation_rules (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    symbol TEXT NOT NULL, -- 'GLOBAL' or specific symbol like 'EURUSD'
    is_enabled BOOLEAN DEFAULT false NOT NULL,
    fixed_lot_size NUMERIC DEFAULT 0.01 NOT NULL,
    max_spread_points INTEGER DEFAULT 50 NOT NULL, -- Maximum allowed spread in points
    strategy_id TEXT, -- Optional, to link to specific strategy
    
    user_id UUID REFERENCES auth.users(id) -- Optional if multi-user
);

-- Add unique constraint to prevent duplicate rules for same symbol/strategy
CREATE UNIQUE INDEX IF NOT EXISTS idx_automation_rules_symbol ON automation_rules(symbol);

-- Enable RLS
ALTER TABLE automation_rules ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Enable read access for all users" ON automation_rules FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users only" ON automation_rules FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Enable update for authenticated users only" ON automation_rules FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Enable delete for authenticated users only" ON automation_rules FOR DELETE USING (auth.role() = 'authenticated');

