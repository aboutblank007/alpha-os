# AlphaOS AI交易系统 - 完整技术文档

## 📋 文档概述

本文档详细描述了AlphaOS AI交易系统的完整架构、数据流程、特征工程和模型训练方案，专门针对5-15分钟剥头皮交易策略优化。

---

## 🏗️ 系统架构

### 整体架构图
```
[MT5终端] ←→ [Python Bridge] ←→ [前端界面]
     ↓              ↓              ↓
[信号生成]      [数据处理]      [监控展示]
     ↓              ↓              ↓
[DOM数据] → [AI引擎(M2 Pro)] → [交易执行]
     ↓              ↓              ↓
[Supabase] ← [特征存储] ← [结果反馈]
```

### 核心组件
1. **MT5指标** - PivotTrendSignals.mq5
2. **Python Bridge** - 信号中转与处理
3. **AI引擎** - LightGBM模型推理
4. **前端界面** - 交易监控与配置
5. **数据库** - Supabase数据存储

---

## 📊 数据模型设计

### 训练信号表 (training_signals)

```sql
CREATE TABLE IF NOT EXISTS training_signals (
    -- 主键标识
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id TEXT UNIQUE,          -- 唯一信号标识
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
```

---

## 🎯 特征工程方案

### 精简剥头皮特征集 (8个核心特征)

```python
SCALPING_FEATURE_SET = {
    # 趋势特征
    'ema_spread_ratio': '(EMA短期 - EMA长期) / ATR',
    'trend_direction': '1=上升, 0=下降',
    'price_vs_cloud': '价格在云层位置',
    
    # 波动特征
    'atr_percent': 'ATR/价格 (标准化波动)',
    'session_volatility': '时段波动特性',
    
    # 过滤特征
    'critical_filters_ok': '核心过滤综合状态',
    'htf_aligned': '高 timeframe 对齐',
    
    # 市场特征
    'signal_density': '近期信号频率'
}
```

### 完整特征集 (用于数据收集)

```python
FULL_FEATURE_SET = {
    # 基础指标
    'ema_short', 'ema_long', 'atr', 'adx', 'center',
    
    # 衍生特征
    'ema_spread', 'atr_percent', 'price_vs_center', 'cloud_width',
    
    # 过滤状态
    'distance_ok', 'slope_ok', 'trend_filter_ok', 'htf_trend_ok',
    'volatility_ok', 'chop_ok', 'spread_ok',
    
    # 上下文特征
    'trend_direction', 'ema_cross_event', 'reclaim_state',
    'is_reclaim_signal', 'bars_since_last', 'session_hour'
}
```

---

## 🔄 数据流程

### 1. 信号生成流程
```mql5
// PivotTrendSignals.mq5 增强输出
void WriteSignalToFile(string action, double price, double sl, double tp, string comment)
{
    string json = StringFormat(
        "{\"signal_id\":\"%s_%d\",\"symbol\":\"%s\",\"action\":\"%s\",\"price\":%.5f,"
        "\"sl\":%.5f,\"tp\":%.5f,\"comment\":\"%s\","
        "\"ema_short\":%.5f,\"ema_long\":%.5f,\"atr\":%.5f,\"adx\":%.5f,\"center\":%.5f,"
        "\"distance_ok\":%d,\"slope_ok\":%d,\"trend_filter_ok\":%d,\"htf_trend_ok\":%d,"
        "\"volatility_ok\":%d,\"chop_ok\":%d,\"spread_ok\":%d,"
        "\"bars_since_last\":%d,\"trend_direction\":%d,\"ema_cross_event\":%d,"
        "\"ema_spread\":%.5f,\"atr_percent\":%.4f,\"reclaim_state\":%d,"
        "\"is_reclaim_signal\":%d,\"price_vs_center\":%.5f,\"cloud_width\":%.5f,"
        "\"timestamp\":%d}",
        // ... 参数列表
    );
}
```

### 2. AI处理流程
```python
# ai-engine/src/client.py
class AIScalpingEngine:
    def process_signal(self, signal_request):
        # 特征提取
        features = self.feature_engineer.extract_scalping_features(signal_request)
        
        # 模型推理
        confidence, should_execute = self.lightgbm_model.predict(features)
        
        # 返回决策
        return SignalResponse(
            should_execute=should_execute,
            confidence=confidence,
            adjusted_sl=self.calculate_adjusted_sl(signal_request),
            adjusted_tp=self.calculate_adjusted_tp(signal_request)
        )
```

### 3. 训练数据流程
```python
# ai-engine/src/train.py
class TrainingPipeline:
    def prepare_training_data(self):
        # 从数据库获取历史信号
        signals = self.fetch_historical_signals()
        
        # 计算标签 (三重屏障法)
        labels = self.calculate_triple_barrier_labels(signals)
        
        # 特征工程
        features = self.feature_engineer.batch_extract(signals)
        
        # 训练模型
        model = self.train_lightgbm(features, labels)
        
        # 保存模型
        self.save_model(model, 'lgbm_scalping_v1.txt')
```

---

## ⚙️ 模型配置

### LightGBM参数
```python
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction_seed': 42,
    'bagging_seed': 42
}
```

### 剥头皮专用参数
```python
SCALPING_CONFIG = {
    'time_window': '15-25分钟',      # 3-5根M5 K线
    'profit_target': 0.001,          # 0.1% 止盈
    'stop_loss': 0.0008,             # 0.08% 止损
    'max_holding_bars': 5,           # 最大持仓K线数
    'confidence_threshold': 0.75,    # 执行置信度
    'daily_trade_limit': 15,         # 每日交易限制
    'cooldown_period': 3             # 交易冷却期(分钟)
}
```

---

## 🚀 实施路线图

### 阶段1: 数据基础设施 (1-2周)
- [ ] 更新MT5指标输出完整特征
- [ ] 设计并创建Supabase数据表
- [ ] 实现Python Bridge数据收集
- [ ] 建立历史数据回填流程

### 阶段2: AI引擎开发 (2-3周)
- [ ] 实现特征工程模块
- [ ] 开发模型训练流水线
- [ ] 集成LightGBM推理服务
- [ ] 实现gRPC通信接口

### 阶段3: 系统集成 (1-2周)
- [ ] 更新前端AI模式配置
- [ ] 实现实时信号处理
- [ ] 添加性能监控面板
- [ ] 部署回退机制

### 阶段4: 优化迭代 (持续)
- [ ] 特征重要性分析
- [ ] 模型参数调优
- [ ] 风险控制增强
- [ ] 多品种扩展

---

## 📈 成功指标

### 技术指标
- 特征计算延迟: < 10ms
- 模型推理延迟: < 5ms
- 系统可用性: > 99.5%
- 数据一致性: > 99.9%

### 交易指标
- 模型准确率: > 60%
- 交易胜率: > 55%
- 夏普比率: > 1.5
- 最大回撤: < 5%

---

## 🔒 风险控制

### 系统风险
```python
SYSTEM_SAFETY = {
    'fallback_mechanism': 'AI失败时回退原逻辑',
    'circuit_breaker': '连续亏损时暂停交易',
    'performance_monitor': '实时监控模型表现',
    'manual_override': '支持人工干预'
}
```

### 交易风险
```python
TRADING_SAFETY = {
    'position_sizing': '固定比例仓位管理',
    'daily_loss_limit': '每日最大亏损限制',
    'correlation_control': '相关品种风险控制',
    'session_filtering': '高波动时段过滤'
}
```

---

## 📞 维护与支持

### 监控项目
- 模型预测准确率趋势
- 特征重要性变化
- 交易执行质量
- 系统资源使用

### 更新策略
- 每周模型重训练
- 每月特征工程优化
- 季度架构评审
- 根据市场变化调整参数

---

## ✅ 验收标准

1. [ ] AI过滤模式可正确配置和切换
2. [ ] 完整特征数据正确存储到Supabase
3. [ ] LightGBM模型可实时推理并返回置信度
4. [ ] 剥头皮交易符合风险参数要求
5. [ ] 系统在实盘环境中稳定运行48小时

---

*文档版本: v1.0*  
*最后更新: 2024-11-26*  
*维护团队: AlphaOS AI开发组*