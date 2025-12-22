# QuantumNet 数据采集

## 基于 MetaTrader 5 回测架构的高保真金融数据集工程报告

---

## 执行摘要

随着量子机器学习（Quantum Machine Learning, QML）与算法交易的深度融合，金融市场分析正在经历一场范式转移。传统的深度学习模型依赖于海量数据集来逼近连续函数，而以 KOSASIH/QuantumNet-Core 为代表的量子神经网络（QNN）框架，利用叠加和纠缠原理在多维希尔伯特空间中处理信息。

然而，这些量子模型的有效性与其输入数据的质量和结构有着本质的联系。不同于可以通过暴力扩展规模来容忍噪声的经典模型，参数化量子电路（PQCs）和变分量子分类器（VQCs）需要高度精选的、"干净"的数据，以减轻退相干和有限量子比特连接性的影响。

本报告旨在提供一份详尽的架构与技术分析，阐述如何设计一个专用的 MetaTrader 5（MT5）智能交易系统（Expert Advisor, EA），用于提取、清洗并格式化符合 QuantumNet 训练标准的金融时间序列数据。

分析重点在于通过回测阶段捕获高级微观结构特征——如成交量冲击（Volume Shock）、订单不平衡代理（Order Imbalance Proxy）和影线比率（Wick Ratios）。利用 MT5 策略测试器作为高保真数据生成器，研究人员可以模拟多年的市场行为，不仅捕获开高低收（OHLC）数据，还能捕获量子态编码所需的衍生"AI 特征"。

报告详细阐述了这些指标的数学公式、MQL5 实现逻辑，以及将原始 CSV 导出转换为适用于旋转角度嵌入或振幅编码的量子张量所需的管道要求。

---

## 1. 量子-经典数据接口架构

### 1.1 量子计算语境下的"干净数据"定义

在 QuantumNet-Core 框架的语境中，"干净数据"这一概念超越了传统数据科学中缺失值填补或离群值移除的定义。对于量子算法而言，数据的"清洁度"直接关系到特征集映射到量子态时的可分离性与表达能力。

量子特征映射 $\phi(x)$ 将经典数据向量 $x$ 转换为量子态 $|\phi(x)\rangle$。如果输入数据包含大量噪声、高度共线性或未经过良好的缩放，生成的量子态可能会坍缩到希尔伯特空间的一个狭窄区域，导致量子核函数与平凡的经典核函数无异，从而丧失量子优势。

研究表明，当输入特征在统计上相互独立或正交时，量子神经网络（QNN）的表现最佳，这使得旋转门（如 $R_x(\theta), R_y(\theta)$）能够充分利用布洛赫球（Bloch Sphere）的整个状态空间。

因此，从 MT5 导出的数据采集管道必须在原位执行严格的特征工程，计算归一化的指标，而不是简单地导出原始的、无界的数值（如原始价格或成交量）。这种原位处理确保了进入量子电路的数据已经具备了某种物理意义上的"纯度"，减少了后续量子纠错（SH-QEC）模块的负担。

### 1.2 QuantumNet 数据规格与全息编码

QuantumNet 的架构，特别是其"自愈量子纠错"（SH-QEC）和"自主量子网络编排器"（AQNO）模块，暗示了对捕捉系统动力学的结构化、高维数据的依赖。虽然具体的输入形状因架构而异（例如，用于轻量级分类器的 4 量子比特电路与用于深度 QNN 的大型寄存器），但标准的数据加载器格式通常预期以下特征：

- **平稳性（Stationarity）**：时间序列数据必须经过差分或对数变换，以确保统计特性不随时间漂移。对于价格数据，通常使用对数收益率或相对于中心点（Center）的偏移量。

- **归一化（Normalization）**：特征必须被缩放到特定的区间（通常是 $[0, 1]$ 用于振幅编码，或 $[-\pi, \pi]$ 用于角度编码），以便直接映射到量子比特的旋转角度。

- **维度匹配（Dimensionality）**：理想情况下，特征的数量应与量子比特的数量或电路的编码能力（如密集编码）相一致，以最小化电路深度和退相干噪声。

用户提供的数据集结构（包含 ai_features JSON 字符串的 CSV）揭示了一种混合方法：首先提取丰富的经典特征集，然后在量子编码之前进行降维（例如通过主成分分析 PCA）。这种"全息数据"（Holographic Data）的概念在 KOSASIH 的文档中也有所体现，暗示了将 $N$ 个特征编码进 $\log(N)$ 个量子比特的高效压缩策略。

### 1.3 MT5 策略测试器作为数据生成器的优势

使用 MT5 策略测试器进行数据采集，相较于下载静态的历史 CSV 文件，具有显著的方法论优势：

1. **指标一致性（Indicator Consistency）**：它确保了指标（如 EMA, RSI, ATR）的计算逻辑与实盘交易中完全一致，消除了"前瞻偏差"和计算库差异带来的误差。

2. **事件重现（Event Reproduction）**：它允许计算基于事件的度量，例如 `bars_since_last`（自上次信号以来的 K 线数）或 `reclaim_state`（价格重回某一水平的状态），这些状态依赖于交易逻辑的运行时状态，而不仅仅是静态的价格历史。

3. **微观结构模拟（Microstructure Simulation）**：在"Every Tick"（每一笔报价）模式下，策略测试器生成的高精度 tick 数据允许近似计算订单流不平衡（Order Flow Imbalance）和攻击性成交量（Aggressor Volume），这些微观结构指标在聚合的 OHLC 数据中是完全丢失的。

---

## 2. 量子金融模型的特征工程理论

为了有效地训练 QuantumNet 模型，EA 必须提取能够捕捉市场潜在状态的特征。根据用户数据中识别出的 `ai_features` 字典，我们需要实现以下类别的指标，并深入理解其物理意义和计算逻辑。

### 2.1 波动率与市场体制度量 (Volatility and Regime Metrics)

在金融 QNN 中，波动率是一个关键输入，因为它定义了市场系统的"温度"或能量水平。

#### 平均真实波幅 (ATR)

衡量市场波动性的标准指标。

**公式**：

$$ATR_n = \frac{1}{n} \sum_{i=1}^{n} \max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|)$$

**量子意义**：ATR 提供了价格变动的尺度因子。在量子数据加载中，ATR 常用于归一化价格偏移量（`price_vs_center`），使其成为无量纲的物理量。

#### 波动率冲击 (Volatility Shock)

该指标识别从低波动到高波动的突然转变，这通常预示着相变（Phase Transition）或趋势的改变。

**MQL5 逻辑**：通常计算为短期波动率与长期波动率的比率。

**公式**：

$$V_{shock} = \frac{\sigma_{short}}{\sigma_{long}}$$

**深度洞察**：突然的波动率冲击可能对应于量子系统中的波函数坍缩，标志着市场从叠加的不确定状态通过测量（交易行为）确立了新的价格状态。

#### 波动率偏度 (Volatility Skew)

衡量波动率分布的不对称性，可能指示方向性的恐惧或贪婪。

### 2.2 成交量与订单流代理 (Volume and Order Flow Proxies)

量子模型擅长在复杂、嘈杂的数据中进行模式识别。成交量动力学提供了价格变动背后的"质量"或动量。

#### 成交量密度 (Volume Density)

通过价格范围对成交量进行归一化，区分高成交量的震荡（犹豫不决）和高成交量的突破（信念坚定）。

**公式**：

$$\rho_{vol} = \frac{Volume}{High - Low}$$

**实现细节**：在 MQL5 中，需要处理 $High - Low = 0$ 的除零异常。通常加上一个最小的 $\epsilon$（如 `_Point`）。

**信号意义**：极高的成交量密度通常意味着在狭窄的价格带内发生了大量的换手，这在量子网络中可以模拟为高纠缠度的节点区域。

#### 成交量冲击 (Volume Shock)

检测相对于基线的异常成交量峰值。

**公式**：

$$S_{vol} = \frac{\ln(V_t) - \ln(MA(V, n))}{\sigma_{vol}}$$

**逻辑**：使用对数成交量是为了适应交易量的幂律分布特性，使其更符合 QNN 线性代数运算的假设。

#### 订单不平衡代理 (Order Imbalance Proxy)

由于 MT5 在外汇市场中通常只提供 tick volume 而非真实的中心化交易所成交量，我们需要通过分析 tick 的方向或 K 线几何形状来近似订单流不平衡。

**公式（近似）**：

$$Imbalance \approx V_{tick} \times \frac{Close - Open}{High - Low}$$

**进阶实现**：更细粒度的方法是在 `OnTick` 函数中设置缓冲区，统计向上跳动（Up-ticks）与向下跳动（Down-ticks）的数量差异，从而推断买方或卖方的攻击性（Aggressor Volume）。

### 2.3 几何与分形特征 (Geometric and Fractal Features)

这些特征描述了价格分布的形态，可以编码到量子态中以检测形态学模式。

#### 影线比率 (Wick Ratio)

量化拒绝压力。高上影线比率暗示卖方主导。

**公式**：

$$R_{wick} = \frac{\max(High - \max(Open, Close), \min(Open, Close) - Low)}{|High - Low|}$$

**量子视角**：在量子金融模型中，影线可以被视为"量子噪声"或"不确定性"的直接度量。高影线比率意味着买卖双方意图的叠加未能解析为明确的方向性收盘，这是一种高熵状态。

#### 云带宽度与距离 (Cloud Width & Distance)

衍生自 Ichimoku 云图或布林带，衡量趋势的势能。

**度量**：`cloud_dist_atr` 将价格到云带均值的距离用当前波动率（ATR）进行归一化，确保特征具有尺度不变性（Scale Invariance）。

### 2.4 目标变量与监督学习

为了进行监督学习（Supervised Learning），EA 不仅需要导出当前的特征 $X_t$，还需要导出未来的结果 $Y_{t+k}$。在回测模式下，我们拥有"上帝视角"，可以预先计算下一根 K 线的涨跌。

**目标定义**：`target_next_close`，即 $Close_{t+1}$ 相对于 $Close_t$ 的变化方向或幅度。这将作为变分量子分类器（VQC）的训练标签。

---

## 3. MQL5 智能交易系统的数据挖掘架构设计

开发用于数据挖掘的 EA 需要与传统交易 EA 截然不同的架构设计。这里的首要任务不是执行速度或低延迟，而是数据的完整性、同步性和文件 I/O 的效率。

### 3.1 数据挖掘 EA 的生命周期

EA 必须在策略测试器中运行以遍历历史数据。其生命周期包括：

1. **OnInit()**：初始化指标句柄（`iMA`, `iATR`, `iRSI` 等），打开 CSV 文件句柄，并写入 CSV 表头（Header Row）。表头的正确性至关重要，它必须与 Python 数据加载器（Data Loader）的预期 Schema 严格匹配。

2. **OnTick()**：检测新 K 线的形成。数据提取通常在 K 线收盘时进行，以确保特征的稳定性。对于 tick 级指标，需要在 K 线形成过程中累积数据，并在收盘时重置。

3. **OnDeinit()**：刷新文件缓冲区并关闭 CSV 文件，防止数据损坏或丢失。

### 3.2 动态指标句柄管理

为了捕获描述的丰富特征集（如 `ema_long`, `ema_short`, `adx`），EA 必须实例化多个指标句柄。MQL5 高效地管理这些句柄，但必须严格在 `OnInit` 中创建，以避免内存泄漏。

**句柄示例**：

```cpp
h_EMA_Fast = iMA(_Symbol, _Period, FastPeriod, 0, MODE_EMA, PRICE_CLOSE);
h_ATR = iATR(_Symbol, _Period, ATRPeriod);
h_Volume = iVolumes(_Symbol, _Period, VOLUME_TICK);
```

### 3.3 "新 K 线"事件处理程序

标准的 MQL5 `OnTick` 在测试期间每毫秒触发一次。为了避免写入数百万行冗余数据，EA 必须检测 K 线收盘的确切时刻。

```cpp
bool IsNewBar() {
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(last_time != current_time) {
      last_time = current_time;
      return true;
   }
   return false;
}
```

这种逻辑确保了诸如 `wick_ratio` 之类的特征是基于完全形成的 `bar[1]` 计算的，而不是正在变动的 bar。

### 3.4 复杂衍生指标的 MQL5 实现逻辑

#### 3.4.1 实现影线比率 (Wick Ratio)

影线比率的计算需要访问 OHLC 数据。这里 `CopyRates` 函数是必不可少的。我们需要计算上影线长度、下影线长度，并将其与 K 线总长度进行比对。

**核心逻辑**：

```
UpperWick = High - Max(Open, Close)
LowerWick = Min(Open, Close) - Low
TotalRange = High - Low
Ratio = (UpperWick + LowerWick) / TotalRange
```

该公式允许 QuantumNet 模型学习独立于绝对价格水平的拒绝模式。

#### 3.4.2 实现成交量密度 (Volume Density)

此指标需要同步访问价格数组和成交量数组。

**核心逻辑**：

```
Range = High - Low
Density = TickVolume / (Range / _Point)
```

> **注意**：除以 `_Point` 是为了将价格范围转换为点数（Points），使密度具有每点成交量的物理意义。高成交量密度伴随低价格变动，是经典的吸筹或派发信号，线性模型常忽略这一细微差别，但 QLSTM 等量子架构有望利用这一点。

#### 3.4.3 实现订单不平衡代理 (Order Imbalance Proxy)

由于 MetaTrader 5 的 `MqlTick` 结构包含 `COPY_TICKS_TRADE` 标志（如果经纪商支持），理论上可以聚合买入发起的交易与卖出发起的交易。但在使用标准 OHLCV 数据的回测环境中，必须使用启发式代理。

**启发式算法**：

利用收盘价在 K 线高低范围内的位置来分配成交量。

$$BuyingVolume \approx Volume \times \frac{Close - Low}{High - Low}$$

$$SellingVolume \approx Volume \times \frac{High - Close}{High - Low}$$

$$Imbalance = BuyingVolume - SellingVolume$$

---

## 4. MQL5 专家顾问（EA）完整代码实现

以下部分将上述研究需求综合为一个连贯的 MQL5 专家顾问结构。该代码旨在生成与用户提供的 `supabase_xauusd_actions.csv` 模式匹配的 CSV 文件。代码包含详细的中文注释，以解释每个模块的功能，特别是针对 QuantumNet 数据要求的特定处理。

```cpp
//+------------------------------------------------------------------+
//| QuantumNet_Data_Miner.mq5 |
//| Copyright 2025, QuantumNet Research Team |
//| https://github.com/KOSASIH |
//+------------------------------------------------------------------+
#property copyright "QuantumNet Research"
#property version   "1.00"
#property strict
#property description "?? QuantumNet ?????????????? EA"

//--- ?????????
input group "Indicator Settings"
input int      InpEMAFast        = 9;          // EMA ????
input int      InpEMASlow        = 21;         // EMA ????
input int      InpRSI            = 14;         // RSI ??
input int      InpATR            = 14;         // ATR ??
input int      InpADX            = 14;         // ADX ??
input int      InpVolMaPeriod    = 20;         // ??????? (?? Shock ??)

//--- ?????????
input group "File Settings"
input string   InpFileName       = "QuantumNet_Training_Data.csv"; // ?????
input bool     InpWriteHeaders   = true;       // ??????

//--- ????
int h_ma_fast, h_ma_slow, h_rsi, h_atr, h_adx;
int file_handle;
int g_bar_count = 0; // ?????????

//--- ????? (????????)
// double vol_buffer; // Unused in this version

//+------------------------------------------------------------------+
//| Expert initialization function |
//+------------------------------------------------------------------+
int OnInit()
{
   // 1. ?????????
   h_ma_fast = iMA(_Symbol, _Period, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
   h_ma_slow = iMA(_Symbol, _Period, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);
   h_rsi     = iRSI(_Symbol, _Period, InpRSI, PRICE_CLOSE);
   h_atr     = iATR(_Symbol, _Period, InpATR);
   h_adx     = iADX(_Symbol, _Period, InpADX);

   // ???????
   if(h_ma_fast == INVALID_HANDLE ||
      h_ma_slow == INVALID_HANDLE || 
      h_rsi == INVALID_HANDLE ||
      h_atr == INVALID_HANDLE ||
      h_adx == INVALID_HANDLE) {
      Print("??: ???????????????");
      return(INIT_FAILED);
   }

   // 2. ?? CSV ??
   // ?? FILE_COMMON ??????????????????????
   Print("====== QuantumNet EA ?? ======");
   Print("????????: ", InpFileName);
   
   // ?????????
   string common_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH);
   Print("??????: ", common_path);
   Print("CSV ????: ", common_path, "\\Files\\", InpFileName);
   
   file_handle = FileOpen(InpFileName, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ",");
   
   if(file_handle == INVALID_HANDLE) {
      int err = GetLastError();
      Print("??: ??????! ????: ", err);
      return(INIT_FAILED);
   }
   
   Print("??????! ??: ", file_handle);

   // 3. ?? CSV ??
   if(InpWriteHeaders) {
      string header = "timestamp,symbol,open,high,low,close,tick_volume," +
                      "ema_fast,ema_slow,ema_spread,rsi,atr,adx," +
                      "wick_upper,wick_lower,wick_ratio,candle_size," +
                      "volume_density,volume_shock,dom_pressure_proxy," +
                      "target_next_close_change";
      FileWrite(file_handle, header);
      Print("?????");
   }

   Print("????????????...");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // ???????????????
   if(file_handle != INVALID_HANDLE) {
      FileClose(file_handle);
      Print("??????????? ", g_bar_count, " ????");
   }
   
   // ??????
   IndicatorRelease(h_ma_fast);
   IndicatorRelease(h_ma_slow);
   IndicatorRelease(h_rsi);
   IndicatorRelease(h_atr);
   IndicatorRelease(h_adx);
}

//+------------------------------------------------------------------+
//| Expert tick function |
//+------------------------------------------------------------------+
void OnTick()
{
   // ??? K ?????????????? K ??? (Bar 1)
   if(!IsNewBar()) return;

   //--- 1. ?????? (OHLCV)
   MqlRates rates[]; // FIXED: Added brackets for array
   ArraySetAsSeries(rates, true);
   // ????? 100 ? K ?????????????? Shock ??
   int copied = CopyRates(_Symbol, _Period, 1, 100, rates); 
   if(copied < InpVolMaPeriod + 1) return;

   //--- 2. ?????????
   double ema_f[], ema_s[], rsi[], atr[], adx[]; // FIXED: Added brackets for arrays
   ArraySetAsSeries(ema_f, true); 
   ArraySetAsSeries(ema_s, true);
   ArraySetAsSeries(rsi, true);   
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(adx, true);

   // ?? Bar 1 ??? (?????? K ?)
   if(CopyBuffer(h_ma_fast, 0, 1, 1, ema_f) <= 0) return;
   if(CopyBuffer(h_ma_slow, 0, 1, 1, ema_s) <= 0) return;
   if(CopyBuffer(h_rsi, 0, 1, 1, rsi) <= 0) return;
   if(CopyBuffer(h_atr, 0, 1, 1, atr) <= 0) return;
   if(CopyBuffer(h_adx, 0, 1, 1, adx) <= 0) return;

   //--- 3. ?? AI ???? (Feature Engineering)
   // Use rates[0] (which is Bar 1 from the chart because start_pos=1)
   
   // A. ?????? (Candle Geometry)
   // ??????????????????????
   double candle_range = rates[0].high - rates[0].low;
   double body_size    = MathAbs(rates[0].close - rates[0].open);
   double wick_upper   = rates[0].high - MathMax(rates[0].open, rates[0].close);
   double wick_lower   = MathMin(rates[0].open, rates[0].close) - rates[0].low;
   
   // Wick Ratio: ???????????? range ? 0 (???), ??? 0
   double wick_ratio = (candle_range > _Point) ? (wick_upper + wick_lower) / candle_range : 0.0;

   // B. ????? (Volume Density)
   // ??: TickVolume / (Range in Points)
   // ??: ???????????????????/????
   double vol_density = 0.0;
   if(candle_range > _Point) {
      vol_density = (double)rates[0].tick_volume / (candle_range / _Point);
   }

   // C. ????? (Volume Shock)
   // ??: ????? / ?? N ????????
   double vol_sum = 0;
   for(int i=1; i<=InpVolMaPeriod; i++) {
      vol_sum += (double)rates[i].tick_volume;
   }
   double vol_ma = vol_sum / (double)InpVolMaPeriod;
   double vol_shock = (vol_ma > 0) ? (double)rates[0].tick_volume / vol_ma : 1.0;

   // D. ??????? (DOM Pressure Proxy)
   // ?????: ??????? K ??????????????????
   // Imbalance = (Close???? - 0.5) * ???
   double range_position = 0.5;
   if(candle_range > _Point) {
      range_position = (rates[0].close - rates[0].low) / candle_range;
   }
   double dom_pressure = (range_position - 0.5) * (double)rates[0].tick_volume; 

   //--- 4. ?????? (Target Variable)
   // ???????????"??"????
   // ? Bar 1 ?????Bar 0 ????????? Bar 0 ? Close?
   // ??????? Bar 1 ??????? Bar 1 ??? Bar 2 ??????
   // ???????????????????
   // ?????????Python ????? Target ??? (Shift -1)?
   // ?????? Bar 1 ??? Bar 2 ?????"??"??????
   double price_change = rates[0].close - rates[1].close; // FIXED: syntax rates.close -> rates[0].close

   //--- 5. ????????
   string time_str = TimeToString(rates[0].time, TIME_DATE|TIME_MINUTES);
   
   // ???? CSV ???
   // FIXED: Accessing array elements [0]
   string csv_row = StringFormat("%s,%s,%.5f,%.5f,%.5f,%.5f,%d," + 
                                 "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f," + 
                                 "%.5f,%.5f,%.5f,%.5f," + 
                                 "%.5f,%.5f,%.2f," + 
                                 "%.5f",
                                 time_str, _Symbol, 
                                 rates[0].open, rates[0].high, rates[0].low, rates[0].close, rates[0].tick_volume,
                                 ema_f[0], ema_s[0], (ema_f[0] - ema_s[0]), // EMA Spread
                                 rsi[0], atr[0], adx[0],
                                 wick_upper, wick_lower, wick_ratio, candle_range,
                                 vol_density, vol_shock, dom_pressure,
                                 price_change
                                 );
                                 
   FileWrite(file_handle, csv_row);
   g_bar_count++;
   
   // ?1000???????
   if(g_bar_count % 1000 == 0) {
      Print("??? ", g_bar_count, " ???...");
   }
   
   // ?????????????????????
   FileFlush(file_handle);
}

//+------------------------------------------------------------------+
//| ??? K ??? |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(last_time != current_time) {
      last_time = current_time;
      return true;
   }
   return false;
}@
```

### 4.4 关键代码逻辑解析

1. **扁平化 CSV 结构**：尽管 QuantumNet 可能在内部使用复杂的 JSON 结构，但在数据采集阶段，扁平化 CSV (Flat CSV) 是最佳实践。Python 的 Pandas 库可以极快地读取扁平文件，并在内存中将其转换为所需的 `holographic_data.json` 格式。尝试在 MQL5 中构建复杂的 JSON 字符串不仅计算昂贵，而且容易出错。

2. **IsNewBar 检查**：这是确保数据不重复的关键。我们只在 K 线收盘确认后（即新 K 线刚开始时）采集上一根 K 线的数据。这意味着记录的数据是确定性的，不会发生重绘（Repainting）。

3. **微观结构特征计算**：
   - `dom_pressure` 的计算使用了 `range_position`（收盘价在 K 线内的相对位置）。这是一个基于价格行为的订单流近似值。如果价格收在最高点（Position = 1.0），则 `dom_pressure` 为正的最大值（$0.5 \times Volume$），表示强烈的买入压力吸收了所有卖单。
   - `vol_density` 解决了"大成交量小涨幅"的异常检测问题。如果 `vol_density` 极高，说明市场遇到了巨大的阻力或支撑，这是量子分类器识别反转点的重要特征。

---

## 5. 集成到 QuantumNet 工作流

### 5.1 数据预处理与量子嵌入 (Quantum Embedding)

一旦 EA 生成了 `QuantumNet_Training_Data.csv`，这些数据虽然在缺失值和计算一致性方面是"干净"的，但还不是"量子就绪"的（Quantum-Ready）。经典数据必须映射到量子态。

标准的管道涉及以下步骤：

#### 标准化 (Standardization/Z-Score)

量子旋转门 $R(\theta)$ 通常接受 $[0, 2\pi]$ 或 $[-\pi, \pi]$ 范围内的参数。原始价格（例如黄金的 2000.00）必须被归一化。

**技术**：使用 Python (Pandas/Scikit-Learn) 进行 MinMax Scaling 或 Z-Score Normalization。

**代码逻辑**：

```python
X_scaled = (X - X.mean()) / X.std()
```

#### 降维 (Dimensionality Reduction)

如果 QuantumNet 模型使用 4 量子比特电路（常见于 NISQ 设备，如 KOSASIH/QuantumNet-Core 文档中提到的 4-qubit dressed circuit），输入特征向量必须被压缩到 4 个维度。

**策略**：EA 导出了约 20 个特征。应在 Python 中应用 **主成分分析 (PCA)** 将 `rsi`, `adx`, `wick_ratio` 等压缩为 4 个正交的主成分。这确保了输入到量子电路的信息是统计独立的，最大化希尔伯特空间的利用率。

### 5.2 "全息"数据结构转换

KOSASIH/QuantumNet-Core 文档提到了 `holographic_data.json`。这意味着框架可能利用一种特定的 JSON 模式进行数据摄取，模拟全息编码（将 $N$ 个特征编码为 $\log(N)$ 个量子比特）。

**推荐工作流**：

1. **MQL5**：运行上述 EA，导出扁平 CSV。

2. **Python Bridge**：编写一个简单的 Python 脚本 `csv_to_holographic.py`。

3. **转换逻辑**：
   - 读取 CSV 到 Pandas DataFrame。
   - 应用 `StandardScaler`。
   - 应用 `PCA(n_components=4)`。
   - 将结果行转换为 JSON 对象：`{"features": [pc1, pc2, pc3, pc4], "target": label}`。
   - 保存为 `holographic_data.json` 供 QuantumNet 加载器使用。

---

## 6. 高级洞察与二阶效应分析

### 6.1 "成交量冲击"作为量子相变的代理

EA 衍生的 `volume_shock` 指标不仅用于确认趋势，它具有更深层的物理意义。在量子金融模型中，市场常被建模为经历相变（Phase Transitions）的复杂系统（例如，从层流/趋势到湍流/崩盘）。

突然的 `volume_shock` 飙升（例如 $> 2\sigma$）作为一个临界信号，可能与量子决策模型中波函数的坍缩相关联。在变分量子分类器（QVC）中，该特征往往具有极高的权重。

### 6.2 影线比率与"量子噪声"

在经典交易中，影线通常被视为噪音。但在 QuantumNet 中，它们代表不确定性。高 `wick_ratio` 意味着买卖意图的叠加未能解析为清晰的方向性收盘。

通过显式地将这一特征输入网络，我们允许 QuantumNet 将"市场犹豫"建模为一个独立的状态，而不仅仅是"弱趋势"。这与量子测量结果的概率性质高度一致。

### 6.3 Tick 数据的局限性与修正

虽然 EA 使用 tick volume 计算 `dom_pressure_proxy`，用户必须意识到其"盲点"。在外汇中，tick volume 与活动相关，但并不总是与方向性意图相关（例如冰山订单是被动吸收流动性）。

`volume_density` 特征通过引入空间维度（价格范围）来修正这一点，有效地创建了一个 2D 特征图（成交量 x 空间），为量子电路提供了比单纯成交量更丰富、更具拓扑结构的输入。

---

## 7. 结论与技术建议

为了成功使用 MT5 数据训练 QuantumNet 模型，数据的"清洁度"不仅由无缺失值定义，更由其特征的丰富度和数学稳定性定义。

本报告提供的 EA 代码超越了简单的 OHLC 提取，工程化了描述市场物理属性的特征——动量（RSI/EMA）、质量（Volume/Density）和能量（Volatility/ATR）。

### 战略建议

1. **使用"Every Tick"模式回测**：为了确保 `dom_pressure` 和 `tick_count` 指标的精确性，必须使用真实 tick 数据（Real Ticks）或基于真实 tick 的生成模式进行回测，而非仅使用"开盘价"模式。

2. **后处理（Post-Processing）在 Python 中完成**：不要试图在 MQL5 EA 内部进行归一化。导出原始浮点值以保留精度。在 QuantumNet Python 管道中执行 PCA 和缩放，以将特征映射到角度嵌入（Angle Embedding）所需的 $[0, 2\pi]$ 区间。

3. **目标变量的标记**：为了进行监督学习，建议在 Python 脚本中进一步处理 CSV，生成分类标签（例如：如果未来 5 根 K 线涨幅 > ATR，则 Label=1，否则 Label=0），从而创建一个可以直接用于训练变分量子电路的标记数据集。

本报告确立了连接经典 MQL5 算法交易与量子金融机器学习前沿的理论和实践框架。提供的代码基础设施确保了 QuantumNet 所需的高保真数据能够被高效、一致地生成。

---

## 附录：QuantumNet 输入特征对照表

| 特征名称 | 符号 | 量子态映射建议 | 物理/市场意义 | MQL5 数据源 |
|----------|------|----------------|---------------|-------------|
| 相对强弱指数 | $RSI$ | 旋转角度 $\theta_1$ | 动量/超买超卖 | `iRSI` 缓冲区 |
| 影线比率 | $R_{wick}$ | 旋转角度 $\theta_2$ | 市场熵/不确定性 | High/Low 几何计算 |
| 成交量密度 | $\rho_{vol}$ | 振幅 $|\psi|$ | 流动性/支撑阻力 | TickVolume / Range |
| 波动率冲击 | $V_{shock}$ | 相位 $\phi$ | 相变/状态突变 | TickVolume 均线偏离度 |
| 订单流代理 | $P_{dom}$ | 辅助量子比特 | 买卖压力不平衡 | 收盘价位置 × 成交量 |

---

## 参考文献

1. KOSASIH/QuantumNet-Core Repository: Framework architecture, Self-Healing QEC, and data requirements.
2. Quantum Machine Learning in Finance: Feature engineering, PCA dimensionality reduction, and quantum encoding strategies (Angle/Amplitude).
3. MQL5 File Operations: Documentation and forum best practices for CSV export, header writing, and file handling.
4. MQL5 Indicators: Technical implementation of handle creation and CopyBuffer for iMA, iATR, iRSI.
5. User Data Analysis: Analysis of ai_features keys including wick_ratio, volume_density, and dom_pressure.
6. Wick Ratio Logic: Mathematical formulas for calculating candle geometry and rejection signals.
7. Volume Density Logic: Derivation of volume per unit of price movement as a measure of liquidity support/resistance.
8. Volume/Volatility Shock: Statistical methods for detecting anomalous market activity.
9. Order Flow Proxies: Techniques for approximating DOM and order imbalance using tick volume and price action.
