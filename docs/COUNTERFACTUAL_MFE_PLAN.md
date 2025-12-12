# 反事实 MFE 计算 - 数据库优化计划

## 📋 背景

**问题**: 训练数据类别严重不平衡（7.4% 真实交易 vs 92.6% 负样本）  
**解决方案**: 使用时间戳数据模拟"如果交易会怎样"，为负样本计算真实的 MFE

**当前状态**: 
- ✅ 已实现 `calculate_counterfactual_mfe()` 函数
- ❌ 遇到时间戳数据类型问题（timestamp 是 datetime 不是 float）
- ❌ 所有查询返回 400 Bad Request，0% 成功率

---

## 🎯 优化目标

1. **修复时间戳类型不匹配**
2. **提高查询性能**（11,000+ 样本 × 每个查询后续数据）
3. **确保数据质量**（至少 80% 样本能计算出 MFE）

---

## 🔧 方案 1：修复当前实现（推荐）

### 问题诊断

```sql
-- training_signals 表结构
Column         | Type                     
---------------+--------------------------
timestamp      | timestamp with time zone  -- ❌ 不是 float!
created_at     | timestamp with time zone
```

当前代码错误：
```python
timestamp = float(row['timestamp'])  # ❌ 实际是 datetime 字符串
future_end = timestamp + (12 * 60)   # ❌ datetime 不能加 int
```

### 修复步骤

#### Step 1: 修改时间戳处理

```python
# 在 calculate_counterfactual_mfe() 函数中
from datetime import datetime, timedelta
import pandas as pd

# 替换 Line 61-62
if pd.isna(row.get('timestamp')):
    continue

# 改为：
timestamp_val = row.get('timestamp')
if pd.isna(timestamp_val):
    continue

# 转换为 datetime
if isinstance(timestamp_val, (int, float)):
    # Unix timestamp
    ts_dt = datetime.fromtimestamp(timestamp_val)
else:
    # ISO 字符串或 pandas Timestamp
    ts_dt = pd.to_datetime(timestamp_val)

future_end_dt = ts_dt + timedelta(minutes=lookforward_bars)
```

#### Step 2: 修改 Supabase 查询

```python
# 替换 Line 67-75
response = supabase.table("training_signals") \
    .select("timestamp, ai_features") \
    .eq("symbol", symbol) \
    .gt("timestamp", timestamp) \          # ❌ float
    .lte("timestamp", future_end) \        # ❌ float
    .not_.is_("ai_features", "null") \
    .order("timestamp") \
    .limit(lookforward_bars) \
    .execute()

# 改为：
response = supabase.table("training_signals") \
    .select("timestamp, ai_features") \
    .eq("symbol", symbol) \
    .gte("timestamp", ts_dt.isoformat()) \     # ✅ ISO 8601 字符串
    .lte("timestamp", future_end_dt.isoformat()) \  # ✅ ISO 8601 字符串
    .not_.is_("ai_features", "null") \
    .order("timestamp") \
    .limit(lookforward_bars) \
    .execute()
```

#### Step 3: 测试验证

```bash
# 部署修复
./deploy_orb.sh --ai

# 测试 100 个样本
ssh macOS "docker exec ai-engine bash -c 'cd /app && python -c \"
import pandas as pd
from enhance_features import calculate_counterfactual_mfe

df = pd.read_csv(\"training_data_enhanced.csv\")
# 只测试前 100 个负样本
test_df = df[df[\"is_negative_sample\"] == True].head(100)
result = calculate_counterfactual_mfe(test_df, lookforward_bars=12)
print(f\"Success rate: {(result[\"result_mfe\"] != 0).sum()} / {len(test_df)}\")
\"'"
```

---

## 🚀 方案 2：数据库优化（高性能）

### 创建物化视图

为了避免 11,000+ 次查询，创建预计算的价格序列视图：

```sql
-- 创建 price_timeseries 物化视图
CREATE MATERIALIZED VIEW price_timeseries AS
SELECT 
    symbol,
    timestamp,
    (ai_features->>'price')::float as price,
    (ai_features->>'close')::float as close,
    LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp
FROM training_signals
WHERE ai_features IS NOT NULL
  AND (ai_features->>'price' IS NOT NULL OR ai_features->>'close' IS NOT NULL)
ORDER BY symbol, timestamp;

-- 创建索引
CREATE INDEX idx_price_ts_symbol_time ON price_timeseries(symbol, timestamp);
REFRESH MATERIALIZED VIEW price_timeseries;
```

### 批量查询优化

```python
def calculate_counterfactual_mfe_optimized(df, lookforward_bars=12):
    """使用批量查询优化性能"""
    from supabase import create_client
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # 按品种分组
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        neg_indices = symbol_df[symbol_df['is_negative_sample'] == True].index
        
        if len(neg_indices) == 0:
            continue
        
        # 获取该品种的所有历史价格（一次性查询）
        all_prices = supabase.table("training_signals") \
            .select("timestamp, ai_features") \
            .eq("symbol", symbol) \
            .not_.is_("ai_features", "null") \
            .order("timestamp") \
            .execute()
        
        # 构建价格字典
        price_dict = {}
        for rec in all_prices.data:
            ts = pd.to_datetime(rec['timestamp'])
            ai_feats = json.loads(rec['ai_features']) if isinstance(rec['ai_features'], str) else rec['ai_features']
            price = ai_feats.get('price') or ai_feats.get('close')
            if price:
                price_dict[ts] = float(price)
        
        # 本地计算 MFE（无需额外查询）
        for idx in neg_indices:
            row = symbol_df.loc[idx]
            ts = pd.to_datetime(row['timestamp'])
            future_end = ts + timedelta(minutes=lookforward_bars)
            
            # 找到时间范围内的价格
            future_prices = [
                price for t, price in price_dict.items()
                if ts < t <= future_end
            ]
            
            if len(future_prices) < 3:
                continue
            
            # 计算 MFE（同现有逻辑）
            # ...
```

**优化效果**:
- 查询次数: 11,000+ → 5 次（每品种一次）
- 预计时间: 30+ 分钟 → 5-10 秒

---

## 🎨 方案 3：简化方案（备选）

如果上述方案太复杂，使用更简单的启发式方法：

### 基于波动率的反事实 MFE

```python
def calculate_simple_counterfactual_mfe(df):
    """
    简化版：不查询实际价格，使用波动率估算
    
    假设：如果交易，MFE ≈ ATR * 随机因子
    """
    import numpy as np
    
    neg_mask = df['is_negative_sample'] == True
    
    for idx in df[neg_mask].index:
        row = df.loc[idx]
        atr = float(row.get('atr', 0.0001))
        
        # 根据市场条件估算 MFE
        # 强趋势 = 高 ADX → 可能错失大机会
        # 弱趋势 = 低 ADX → 正确避免
        adx = float(row.get('adx', 20))
        
        if adx > 25:  # 强趋势
            # 可能错失机会，分配正 MFE
            random_factor = np.random.normal(0.5, 0.3)  # 均值 0.5 ATR
        else:  # 弱趋势
            # 正确避免，分配小或负 MFE  
            random_factor = np.random.normal(-0.1, 0.2)
        
        mfe = random_factor
        df.loc[idx, 'result_mfe'] = np.clip(mfe, -2, 10)
    
    return df
```

**优点**:
- 无需数据库查询
- 瞬间完成
- 引入合理的随机性

**缺点**:
- 不是真实的反事实
- 依赖启发式假设

---

## 📊 实施优先级

### 立即（本周）
- [ ] **方案 1 修复**: 修改 timestamp 处理（2小时）
- [ ] 测试 100 个样本验证（30分钟）
- [ ] 全量运行并重新训练（1小时）

### 短期（下周）
- [ ] **方案 2 优化**: 创建物化视图（1小时）
- [ ] 实现批量查询版本（2小时）
- [ ] 性能对比测试（30分钟）

### 中期（下月）
- [ ] 对比三种方案的实战效果
- [ ] A/B 测试：真实MFE vs 估算MFE
- [ ] 选择最优方案作为标准

---

## 📝 预期效果

### 数据分布改善

**之前**:
```
result_mfe 分布：
- 92.6%: 固定 0.0
- 7.4%: 真实值（-2 到 +10）
```

**之后** (方案 1/2):
```
result_mfe 分布：
- ~60-70%: 小的正值或负值（0.5 到 -0.5）
- ~20-30%: 中等值（0.5 到 2.0）
- ~7.4%: 真实交易值
- ~2-3%: 大的正值（2.0+，错失机会）
```

### 模型训练改善

- **R² Score**: 预计从 1.0 降至 0.6-0.8（更真实）
- **MAE**: 预计从 0.0 升至 0.3-0.5
- **类别平衡**: 从 7.4:92.6 改善至近似正态分布

---

## 🔍 验收标准

成功指标：
- [ ] 至少 80% 的负样本成功计算 MFE
- [ ] 负样本 MFE 均值接近 0（±0.3）
- [ ] 负样本 MFE 标准差 > 0.5（有变化）
- [ ] 模型验证指标不再完美（R² < 0.95）
- [ ] 实战胜率提升至 35%+（当前 28%）

---

## 📚 相关文档

- [当前实现](file:///Users/hanjianglin/github/alpha-os/ai-engine/enhance_features.py#L10-L149) (calculate_counterfactual_mfe)
- [训练质量报告](file:///Users/hanjianglin/github/alpha-os/ai-engine/TRAINING_QUALITY_REPORT.md)
- [数据库 Schema](file:///Users/hanjianglin/github/alpha-os/src/db/cloud_schema.sql)

---

**下次实施**: 按方案 1 修复 → 方案 2 优化 → 方案 3 作为 fallback
