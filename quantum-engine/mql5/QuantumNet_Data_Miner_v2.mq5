//+------------------------------------------------------------------+
//| QuantumNet_Data_Miner_v2.mq5                                     |
//| Copyright 2025, QuantumNet Research Team                          |
//| https://github.com/KOSASIH                                        |
//+------------------------------------------------------------------+
#property copyright "QuantumNet Research"
#property version   "2.00"
#property strict
#property description "🔬 QuantumNet 量子交易训练数据采集 EA v2 (含微观状态)"

//--- 指标参数设置
input group "Indicator Settings"
input int      InpEMAFast        = 9;          // EMA 快线周期
input int      InpEMASlow        = 21;         // EMA 慢线周期
input int      InpRSI            = 14;         // RSI 周期
input int      InpATR            = 14;         // ATR 周期
input int      InpADX            = 14;         // ADX 周期
input int      InpVolMaPeriod    = 20;         // 成交量均线周期 (用于 Shock 计算)

//--- 文件输出设置
input group "File Settings"
input string   InpFileName       = "QuantumNet_Training_Data_v2.csv"; // 输出文件名
input bool     InpWriteHeaders   = true;       // 是否写入表头

//--- 指标句柄
int h_ma_fast, h_ma_slow, h_rsi, h_atr, h_adx;
int file_handle;
int g_bar_count = 0; // 已处理的K线数

//--- 用于计算 Tick Rate
int g_tick_count_in_bar = 0;       // 当前K线内的tick计数
datetime g_current_bar_time = 0;   // 当前K线时间

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // 1. 初始化技术指标
   h_ma_fast = iMA(_Symbol, _Period, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
   h_ma_slow = iMA(_Symbol, _Period, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);
   h_rsi     = iRSI(_Symbol, _Period, InpRSI, PRICE_CLOSE);
   h_atr     = iATR(_Symbol, _Period, InpATR);
   h_adx     = iADX(_Symbol, _Period, InpADX);

   // 检查指标句柄
   if(h_ma_fast == INVALID_HANDLE ||
      h_ma_slow == INVALID_HANDLE || 
      h_rsi == INVALID_HANDLE ||
      h_atr == INVALID_HANDLE ||
      h_adx == INVALID_HANDLE) {
      Print("错误: 无法创建指标句柄！");
      return(INIT_FAILED);
   }

   // 2. 打开 CSV 文件
   Print("====== QuantumNet EA v2 启动 ======");
   Print("输出文件名: ", InpFileName);
   
   // 获取公共文件夹路径
   string common_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH);
   Print("公共文件夹: ", common_path);
   Print("CSV 路径: ", common_path, "\\Files\\", InpFileName);
   
   file_handle = FileOpen(InpFileName, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ",");
   
   if(file_handle == INVALID_HANDLE) {
      int err = GetLastError();
      Print("错误: 无法打开文件! 错误码: ", err);
      return(INIT_FAILED);
   }
   
   Print("文件打开成功! 句柄: ", file_handle);

   // 3. 写入 CSV 表头 (新增 spread, tick_rate, bid_ask_imbalance)
   if(InpWriteHeaders) {
      string header = "timestamp,symbol,open,high,low,close,tick_volume," +
                      "ema_fast,ema_slow,ema_spread,rsi,atr,adx," +
                      "wick_upper,wick_lower,wick_ratio,candle_size," +
                      "volume_density,volume_shock,dom_pressure_proxy," +
                      "spread,tick_rate,bid_ask_imbalance," +
                      "target_next_close_change";
      FileWrite(file_handle, header);
      Print("表头写入完成");
   }

   Print("开始采集交易数据...");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // 关闭文件并报告统计
   if(file_handle != INVALID_HANDLE) {
      FileClose(file_handle);
      Print("数据采集结束，共采集 ", g_bar_count, " 条K线数据");
   }
   
   // 释放指标
   IndicatorRelease(h_ma_fast);
   IndicatorRelease(h_ma_slow);
   IndicatorRelease(h_rsi);
   IndicatorRelease(h_atr);
   IndicatorRelease(h_adx);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 检测新K线
   datetime current_time = iTime(_Symbol, _Period, 0);
   
   // 累计当前K线的tick数
   if(current_time == g_current_bar_time) {
      g_tick_count_in_bar++;
   } else {
      // 新K线开始时，处理前一根K线并重置计数
      if(g_current_bar_time != 0) {
         // 保存前一根K线的tick_rate供写入使用
         ProcessBar(1, g_tick_count_in_bar);
      }
      g_current_bar_time = current_time;
      g_tick_count_in_bar = 1;  // 新K线的第一个tick
   }
}

//+------------------------------------------------------------------+
//| 处理完成的K线数据                                                 |
//+------------------------------------------------------------------+
void ProcessBar(int bar_index, int tick_rate)
{
   //--- 1. 获取K线数据 (OHLCV)
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   // 复制最近 100 根K线（用于计算 Shock 和均线）
   int copied = CopyRates(_Symbol, _Period, bar_index, 100, rates); 
   if(copied < InpVolMaPeriod + 1) return;

   //--- 2. 获取技术指标值
   double ema_f[], ema_s[], rsi[], atr[], adx[];
   ArraySetAsSeries(ema_f, true); 
   ArraySetAsSeries(ema_s, true);
   ArraySetAsSeries(rsi, true);   
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(adx, true);

   // 获取 Bar 1 的指标值 (已完成的K线)
   if(CopyBuffer(h_ma_fast, 0, bar_index, 1, ema_f) <= 0) return;
   if(CopyBuffer(h_ma_slow, 0, bar_index, 1, ema_s) <= 0) return;
   if(CopyBuffer(h_rsi, 0, bar_index, 1, rsi) <= 0) return;
   if(CopyBuffer(h_atr, 0, bar_index, 1, atr) <= 0) return;
   if(CopyBuffer(h_adx, 0, bar_index, 1, adx) <= 0) return;

   //--- 3. 计算 AI 工程特征 (Feature Engineering)
   
   // A. 蜡烛图几何 (Candle Geometry)
   double candle_range = rates[0].high - rates[0].low;
   double body_size    = MathAbs(rates[0].close - rates[0].open);
   double wick_upper   = rates[0].high - MathMax(rates[0].open, rates[0].close);
   double wick_lower   = MathMin(rates[0].open, rates[0].close) - rates[0].low;
   
   // Wick Ratio: 影线总长度占比
   double wick_ratio = (candle_range > _Point) ? (wick_upper + wick_lower) / candle_range : 0.0;

   // B. 成交量密度 (Volume Density)
   double vol_density = 0.0;
   if(candle_range > _Point) {
      vol_density = (double)rates[0].tick_volume / (candle_range / _Point);
   }

   // C. 成交量冲击 (Volume Shock)
   double vol_sum = 0;
   for(int i=1; i<=InpVolMaPeriod; i++) {
      vol_sum += (double)rates[i].tick_volume;
   }
   double vol_ma = vol_sum / (double)InpVolMaPeriod;
   double vol_shock = (vol_ma > 0) ? (double)rates[0].tick_volume / vol_ma : 1.0;

   // D. 盘口压力代理 (DOM Pressure Proxy)
   double range_position = 0.5;
   if(candle_range > _Point) {
      range_position = (rates[0].close - rates[0].low) / candle_range;
   }
   double dom_pressure = (range_position - 0.5) * (double)rates[0].tick_volume; 

   //--- 4. 微观状态字段 (新增 v2)
   
   // E. Spread (点数) - 使用该K线时间点的Spread
   int spread_points = iSpread(_Symbol, _Period, bar_index);
   
   // F. Tick Rate - 该K线期间的tick次数（已作为参数传入）
   // tick_rate 已经作为参数传入
   
   // G. Bid/Ask Imbalance (Alpha101 修正版：位置 * 能量)
   // 公式: ((Close - Low) / Range - 0.5) * VolumeShock
   double bid_ask_imbalance = 0.0;
   if(candle_range > _Point) {
      double position_factor = ((rates[0].close - rates[0].low) / candle_range) - 0.5;
      bid_ask_imbalance = position_factor * vol_shock;
   }

   //--- 5. 计算目标变量 (Target Variable)
   // 当前K线(Bar 1)到前一根K线(Bar 2)的价格变化
   double price_change = rates[0].close - rates[1].close;

   //--- 6. 格式化并写入 (Float64 适配: %.7f 精度)
   string time_str = TimeToString(rates[0].time, TIME_DATE|TIME_MINUTES);
   
   // 构建 CSV 行（新增 3 个微观状态字段）
   string csv_row = StringFormat("%s,%s,%.7f,%.7f,%.7f,%.7f,%d," + 
                                 "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f," + 
                                 "%.7f,%.7f,%.7f,%.7f," + 
                                 "%.7f,%.7f,%.7f," + 
                                 "%d,%d,%.7f," +
                                 "%.7f",
                                 time_str, _Symbol, 
                                 rates[0].open, rates[0].high, rates[0].low, rates[0].close, rates[0].tick_volume,
                                 ema_f[0], ema_s[0], (ema_f[0] - ema_s[0]), // EMA Spread
                                 rsi[0], atr[0], adx[0],
                                 wick_upper, wick_lower, wick_ratio, candle_range,
                                 vol_density, vol_shock, dom_pressure,
                                 spread_points, tick_rate, bid_ask_imbalance,
                                 price_change
                                 );
                                 
   FileWrite(file_handle, csv_row);
   g_bar_count++;
   
   // 每1000条打印一次进度
   if(g_bar_count % 1000 == 0) {
      Print("已采集 ", g_bar_count, " 条数据...");
   }
   
   // 立即刷新到磁盘，防止数据丢失
   FileFlush(file_handle);
}
//+------------------------------------------------------------------+
