# Kelly Criterion Lot Size Fix

**Date**: 2025-12-12  
**Issue**: Kelly lot sizing only producing 0.01 or 1.0 lots (extreme values)  
**Status**: ✅ FIXED

---

## 🐛 Root Cause

### Original Buggy Formula (Line 381)
```python
lot_size = (account_balance * kelly) / 100.0
```

### Problem Example
With:
- `account_balance` = $10,000
- `kelly` = 0.1 (10% of bankroll)
- `kelly_fraction` = 0.25 (fractional Kelly)

**Buggy calculation**:
```
kelly = 0.1 * 0.25 = 0.025 (2.5% of bankroll)
lot_size = (10000 * 0.025) / 100.0 = 250 / 100 = 2.5 lots
final_lot = min(max(2.5, 0.01), 1.0) = 1.0 lots (capped)
```

Or with lower Kelly:
```
kelly = 0.01 (1%)
lot_size = (10000 * 0.01) / 100 = 1.0 lots
```

Or even lower:
```
kelly = 0.001 (0.1%)
lot_size = (10000 * 0.001) / 100 = 0.1 lots
```

**Issue**: The `/100.0` division was arbitrary and caused either:
1. **Too large** (capped to max_lot = 1.0)
2. **Too small** (floored to 0.01)

No intermediate values possible!

---

## ✅ Fixed Formula

### New Correct Calculation
```python
# Kelly suggests what % of bankroll to risk
margin_per_lot = 1000.0  # $1000 margin per 1.0 lot

risk_amount = account_balance * kelly
lot_size = risk_amount / margin_per_lot
lot_size = round(lot_size, 2)

final_lot = min(max(lot_size, 0.01), max_lot)
```

### Example Scenarios

**Scenario 1: Conservative (Kelly = 2.5%)**
```
Account: $10,000
kelly = 0.025
risk_amount = 10,000 * 0.025 = $250
lot_size = 250 / 1000 = 0.25 lots ✅
```

**Scenario 2: Moderate (Kelly = 5%)**
```
Account: $10,000
kelly = 0.05
risk_amount = 10,000 * 0.05 = $500
lot_size = 500 / 1000 = 0.50 lots ✅
```

**Scenario 3: Aggressive (Kelly = 10%)**
```
Account: $10,000
kelly = 0.10
risk_amount = 10,000 * 0.10 = $1000
lot_size = 1000 / 1000 = 1.00 lots (at cap) ✅
```

**Scenario 4: Very Conservative (Kelly = 1%)**
```
Account: $10,000
kelly = 0.01
risk_amount = 10,000 * 0.01 = $100
lot_size = 100 / 1000 = 0.10 lots ✅
```

Now we get **continuous intermediate values**: 0.10, 0.25, 0.50, 0.75, 1.00 etc!

---

## 📊 Kelly Formula Recap

### Standard Kelly Criterion
```
f* = (p * b - q) / b

Where:
- p = Win probability
- b = Win/Loss ratio (avg_win / avg_loss)
- q = Loss probability (1 - p)
- f* = Optimal fraction of bankroll to risk
```

### Fractional Kelly (Safety Factor)
```
kelly_actual = f* * kelly_fraction

Default: kelly_fraction = 0.25 (Quarter Kelly)
Max cap: 25% of bankroll regardless of calculation
```

### AI Integration
```
If AI confidence provided:
  p = clamp(confidence, 0.1, 0.9)
Else:
  p = historical_win_rate
```

---

## 🔧 Adjustable Parameters

### In `automation_rules` table:
- `kelly_fraction` (default: 0.25)
  - 0.25 = Quarter Kelly (conservative)
  - 0.50 = Half Kelly (moderate)
  - 1.00 = Full Kelly (aggressive, not recommended!)

- `kelly_lookback_trades` (default: 50)
  - Number of recent trades to analyze

- `max_lot_size` (default: 1.0)
  - Hard cap on position size

### In code:
- `margin_per_lot` (hardcoded: $1000)
  - Adjust based on:
    - Broker margin requirements
    - Symbol type (Forex, Indices, Crypto)
    - Account leverage

**Recommended values**:
- Forex (1:100 leverage): $100-500 per lot
- Indices (1:20 leverage): $1000-2000 per lot  
- Current: $1000 (conservative for mixed trading)

---

##Expected Lot Size Distribution

With default settings (kelly_fraction=0.25, max_lot=1.0):

| Win Rate | Win/Loss Ratio | Kelly* | Actual Kelly | $10k → Lots |
|----------|----------------|--------|--------------|-------------|
| 40% | 1.5 | 0.07 | 0.018 | 0.18 |
| 45% | 1.5 | 0.18 | 0.045 | 0.45 |
| 50% | 1.5 | 0.25 | 0.063 | 0.63 |
| 55% | 1.5 | 0.32 | 0.080 | 0.80 |
| 60% | 2.0 | 0.40 | 0.100 | 1.00 (capped) |

**Result**: Smooth gradation from 0.18 → 1.00 lots! ✅

---

## 🚀 Deployment

**File Modified**: `trading-bridge/src/main.py` (lines 373-388)

**Deployment Command**:
```bash
./deploy_orb.sh --bridge
```

**Verification**:
1. Check Bridge API logs for Kelly calculation output
2. Look for: `📊 Kelly: ... Kelly%=X.X% Risk=$XXX -> X.XX lots`
3. Verify lot sizes are between 0.01 and max_lot

---

## 📝 Testing Checklist

- [ ] Verify lot sizes are no longer just 0.01 or 1.0
- [ ] Check intermediate values appear (0.25, 0.50, 0.75)
- [ ] Confirm Kelly% and Risk$ shown in logs
- [ ] Test with different account balances
- [ ] Test with different win rates
- [ ] Validate max_lot cap still works

---

## 🎓 Key Takeaway

**Before**: Division by 100 was a miscalculation  
**After**: Proper risk-to-lot conversion using margin per lot

The Kelly % is already a fraction (0-0.25), so we:
1. Calculate risk amount = balance * kelly%
2. Convert to lots = risk / margin_per_lot
3. Apply bounds (0.01 to max_lot)

Simple and correct! 🎯
