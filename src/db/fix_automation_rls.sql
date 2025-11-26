-- Fix RLS policies for automation_rules to allow anonymous access (for local/personal use)
-- Run this in your Supabase SQL Editor

ALTER TABLE automation_rules ENABLE ROW LEVEL SECURITY;

-- Drop restrictive policies
DROP POLICY IF EXISTS "Enable insert for authenticated users only" ON automation_rules;
DROP POLICY IF EXISTS "Enable update for authenticated users only" ON automation_rules;
DROP POLICY IF EXISTS "Enable delete for authenticated users only" ON automation_rules;

-- Create permissive policies for all operations
CREATE POLICY "Enable insert for all users" ON automation_rules FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON automation_rules FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON automation_rules FOR DELETE USING (true);

