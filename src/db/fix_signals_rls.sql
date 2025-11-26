-- Fix RLS policies for signals table to allow anonymous access (consistent with automation_rules)
-- Run this in your Supabase SQL Editor

ALTER TABLE signals ENABLE ROW LEVEL SECURITY;

-- Drop restrictive policies
DROP POLICY IF EXISTS "Enable all for authenticated users" ON signals;
DROP POLICY IF EXISTS "Enable insert for all users" ON signals;
DROP POLICY IF EXISTS "Enable select for all users" ON signals;
DROP POLICY IF EXISTS "Enable update for all users" ON signals;

-- Create permissive policies for all operations
CREATE POLICY "Enable insert for all users" ON signals FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable select for all users" ON signals FOR SELECT USING (true);
CREATE POLICY "Enable update for all users" ON signals FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON signals FOR DELETE USING (true);

