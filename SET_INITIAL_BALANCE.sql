-- ============================================
-- 设置初始本金
-- ============================================
-- 说明: 此脚本用于快速设置您的初始账户本金
-- 使用方法: 修改下面的本金金额，然后在 Supabase SQL Editor 中执行
-- ============================================

-- 方式1: 如果账户已存在，更新初始本金
UPDATE accounts 
SET 
  initial_balance = 1084.8,    -- 修改为您的实际本金
  current_balance = 1084.8     -- 初始时，当前余额等于初始本金
WHERE id = '00000000-0000-0000-0000-000000000001';

-- 方式2: 如果账户不存在，创建新账户
INSERT INTO accounts (id, name, initial_balance, current_balance, currency)
VALUES (
  '00000000-0000-0000-0000-000000000001',
  '主账户',
  1084.8,    -- 修改为您的实际本金
  1084.8,
  'USD'
)
ON CONFLICT (id) DO UPDATE SET
  initial_balance = EXCLUDED.initial_balance,
  current_balance = EXCLUDED.current_balance,
  name = EXCLUDED.name;

-- 验证设置
SELECT 
  name as 账户名称,
  initial_balance as 初始本金,
  current_balance as 当前余额,
  currency as 货币,
  created_at as 创建时间
FROM accounts
WHERE id = '00000000-0000-0000-0000-000000000001';

-- ============================================
-- 常见本金设置示例
-- ============================================

-- 示例1: 小额账户 ($200)
-- UPDATE accounts SET initial_balance = 200.00, current_balance = 200.00 WHERE id = '00000000-0000-0000-0000-000000000001';

-- 示例2: 中等账户 ($1000)
-- UPDATE accounts SET initial_balance = 1000.00, current_balance = 1000.00 WHERE id = '00000000-0000-0000-0000-000000000001';

-- 示例3: 标准账户 ($5000)
-- UPDATE accounts SET initial_balance = 5000.00, current_balance = 5000.00 WHERE id = '00000000-0000-0000-0000-000000000001';

-- 示例4: 大额账户 ($10000)
-- UPDATE accounts SET initial_balance = 10000.00, current_balance = 10000.00 WHERE id = '00000000-0000-0000-0000-000000000001';

-- ============================================
-- 重要提醒
-- ============================================
-- 1. initial_balance 应该是您开始交易时的本金
-- 2. 不要手动修改 current_balance，它会通过 API 自动计算
-- 3. 如果您已有交易记录，净资产 = initial_balance + 所有交易盈亏
-- 4. 修改 initial_balance 后，刷新页面查看左上角的净资产显示

