//+------------------------------------------------------------------+
//|                                       PivotTrend_DataCollector.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                      Adapted from Pine Script V3 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "3.00"
#property description "Expert Advisor: Collects Signals AND Outcomes for ML Training"

//--- Inputs
input group "=== Core Settings ==="
input int      InpPivotPeriod = 2;     // Pivot Point Period
input int      InpATRPeriod   = 14;    // ATR Period
input int      InpEMALength1  = 6;     // EMA Short Period
input int      InpEMALength2  = 24;    // EMA Long Period

input group "=== Filter Settings ==="
enum ENUM_FILTER_MODE {
   FILTER_NONE,   // No Filter
   FILTER_BASIC,  // Basic Filter (+Trend)
   FILTER_STRICT  // Strict Filter (+Vol+Chop)
};
input ENUM_FILTER_MODE InpFilterMode = FILTER_BASIC; // Filter Mode
input double   InpMinDistance    = 0.4;   // Min Distance (ATR Multiplier)
input double   InpTrendBuffer    = 0.1;   // Trend Buffer (ATR Multiplier)
input bool     InpUseSlope       = true;  // Enable Slope Filter
input double   InpSlopeThreshold = 0.5;   // Min Slope Threshold
input bool     InpStrictSlope    = true;  // Strict Slope (2 bars)
input double   InpMinEMASpread   = 0.1;   // Min EMA Spread (ATR Multiplier)
input int      InpMinBarsBetween = 3;     // Min Bars Between Signals
input bool     InpUseHTFFilter   = true;  // Use HTF Trend Filter
input ENUM_TIMEFRAMES InpHTFPeriod = PERIOD_H1; // HTF Period

input group "=== Reclaim Settings ==="
input bool     InpUseReclaim     = true;  // Enable Reclaim Signals
input bool     InpReclaimStrict  = true;  // Reclaim Strict (ADX>20)

input group "=== Risk Management ==="
input double   InpTPMultiplier   = 1.5;   // TP ATR Multiplier
input double   InpSLMultiplier   = 1.0;   // SL ATR Multiplier

//--- Handles
int            hEMA1;
int            hEMA2;
int            hATR;
int            hADX;
int            hHTF_EMA1;
int            hHTF_EMA2;

//--- Structures
struct SignalFeatures {
   double ema_short;
   double ema_long;
   double atr;
   double adx;
   double center;
   bool distance_ok;
   bool slope_ok;
   bool trend_filter_ok;
   bool htf_trend_ok;
   bool volatility_ok;
   bool chop_ok;
   bool spread_ok;
   int bars_since_last;
   int trend_direction;
   int ema_cross_event;
   double ema_spread;
   double atr_percent;
   int reclaim_state;
   bool is_reclaim_signal;
   double price_vs_center;
   double cloud_width;
};

struct VirtualTrade {
    string   signal_id;
    string   symbol;
    string   action; 
    double   entry_price;
    double   sl;
    double   tp;
    datetime open_time;
    bool     active;
    SignalFeatures features;
};

//--- State Variables
VirtualTrade active_trades[];
int          trade_counter = 0;
datetime     last_bar_time = 0;

// Buffers for Pivot Calculation (Simulated inside EA)
double       BufferHigh[];
double       BufferLow[];
double       BufferCenter[];
double       BufferReclaimState[];
double       BufferReclaimPrice[];
double       BufferLastBuyBar[];
double       BufferLastSellBar[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   hEMA1 = iMA(_Symbol, _Period, InpEMALength1, 0, MODE_EMA, PRICE_CLOSE);
   hEMA2 = iMA(_Symbol, _Period, InpEMALength2, 0, MODE_EMA, PRICE_CLOSE);
   hATR  = iATR(_Symbol, _Period, InpATRPeriod);
   hADX  = iADX(_Symbol, _Period, 14); 
   
   if(InpUseHTFFilter)
     {
      hHTF_EMA1 = iMA(_Symbol, InpHTFPeriod, InpEMALength1, 0, MODE_EMA, PRICE_CLOSE);
      hHTF_EMA2 = iMA(_Symbol, InpHTFPeriod, InpEMALength2, 0, MODE_EMA, PRICE_CLOSE);
     }

   if(hEMA1 == INVALID_HANDLE || hEMA2 == INVALID_HANDLE || hATR == INVALID_HANDLE)
     {
      Print("Failed to create indicator handles");
      return(INIT_FAILED);
     }

   // Test File Write Permission
   string test_file = "test_permission_" + IntegerToString(TimeCurrent()) + ".txt";
   int h = FileOpen(test_file, FILE_WRITE|FILE_TXT);
   if(h != INVALID_HANDLE) {
       FileWriteString(h, "OK");
       FileClose(h);
       Print("✅ Permission Check: OK (Written to ", test_file, ")");
       FileDelete(test_file);
   } else {
       Print("❌ Permission Check: FAILED (Error ", GetLastError(), ")");
   }

   Print("PivotTrend Data Collector EA V3.00 Initialized");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Close all handles
   IndicatorRelease(hEMA1);
   IndicatorRelease(hEMA2);
   IndicatorRelease(hATR);
   IndicatorRelease(hADX);
   if(InpUseHTFFilter) {
       IndicatorRelease(hHTF_EMA1);
       IndicatorRelease(hHTF_EMA2);
   }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Only run on new bar
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(current_time == last_bar_time) return;
   last_bar_time = current_time;

   // Calculate for PREVIOUS bar (Bar 1) because Bar 0 is forming
   int i = 1; 
   
   // Need enough data
   if(iBars(_Symbol, _Period) < InpEMALength2 + 50) return;

   // 1. Fetch Data
   int count = 300; // Increase buffer size for lookback calculations
   
   double ema1[], ema2[], atr[], adx[], high[], low[], close[];
   ArraySetAsSeries(ema1, true);
   ArraySetAsSeries(ema2, true);
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(adx, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);

   if(CopyBuffer(hEMA1, 0, 0, count, ema1) < count) return;
   if(CopyBuffer(hEMA2, 0, 0, count, ema2) < count) return;
   if(CopyBuffer(hATR, 0, 0, count, atr) < count) return;
   if(CopyBuffer(hADX, 0, 0, count, adx) < count) return;
   
   if(CopyHigh(_Symbol, _Period, 0, count, high) < count) return;
   if(CopyLow(_Symbol, _Period, 0, count, low) < count) return;
   if(CopyClose(_Symbol, _Period, 0, count, close) < count) return;

   // Resize state buffers if needed (simulate indicator buffers)
   // For EA, we just need variables for current state and maybe a small history array if logic depends on history
   // Pivot logic needs history.
   // Simplification: We re-calculate Pivot for recent bars to determine 'Center'. 
   // Since we only trade on Bar 1, we need Center for Bar 1.
   
   // 2. Calculate Pivot Center (Simplified for EA)
   // To get Center[i], we need to look back.
   // We can run a mini-loop.
   
   double center = CalculatePivotCenter(high, low, i); 

   // 3. Trend Logic
   bool trend_up = ema1[i] > ema2[i];
   bool cross_up = (ema1[i] > ema2[i]) && (ema1[i+1] <= ema2[i+1]);
   bool cross_dn = (ema1[i] < ema2[i]) && (ema1[i+1] >= ema2[i+1]);

   // 4. Features
   double atr_val = atr[i];
   double dist = (atr_val != 0) ? MathAbs(close[i] - center) / atr_val : 0;
   
   double slope = MathAbs(ema1[i] - ema1[i+1]) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   bool slope_val_ok = slope > InpSlopeThreshold;
   double prev_slope = MathAbs(ema1[i+1] - ema1[i+2]) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   bool prev_slope_ok = prev_slope > InpSlopeThreshold;
   bool slope_strict_ok = InpStrictSlope ? (slope_val_ok && prev_slope_ok) : slope_val_ok;
   bool slope_ok = InpUseSlope ? slope_strict_ok : false;
   
   bool distance_ok = (dist >= InpMinDistance) || slope_ok;
   
   double t_buf = atr_val * InpTrendBuffer;
   bool price_above = close[i] > (center + t_buf);
   bool price_below = close[i] < (center - t_buf);
   bool trend_filter_ok = (InpFilterMode == FILTER_NONE) ? true :
                          ((trend_up && price_above) || (!trend_up && price_below));

   bool htf_trend_ok = true;
   if(InpUseHTFFilter)
     {
      double h_ema1[], h_ema2[];
      ArraySetAsSeries(h_ema1, true);
      ArraySetAsSeries(h_ema2, true);
      
      // Get HTF Bar Index corresponding to current time
      datetime bar_time = iTime(_Symbol, _Period, i);
      int htf_idx = iBarShift(_Symbol, InpHTFPeriod, bar_time);
      
      CopyBuffer(hHTF_EMA1, 0, 0, htf_idx+2, h_ema1); 
      CopyBuffer(hHTF_EMA2, 0, 0, htf_idx+2, h_ema2);
      
      // Check logic carefully - Series array means index 0 is latest
      // But we copied 0..htf_idx+2. 
      // Actually simplest is CopyBuffer with time.
      CopyBuffer(hHTF_EMA1, 0, bar_time, 1, h_ema1);
      CopyBuffer(hHTF_EMA2, 0, bar_time, 1, h_ema2);
      
      if(ArraySize(h_ema1)>0 && ArraySize(h_ema2)>0)
        {
         bool htf_up = h_ema1[0] > h_ema2[0];
         htf_trend_ok = (trend_up && htf_up) || (!trend_up && !htf_up);
        }
     }

   double atr_pct = (close[i] != 0) ? (atr_val / close[i] * 100) : 0;
   bool volatility_ok = true;
   bool not_chop = true;
   
   if(InpFilterMode == FILTER_STRICT)
     {
      volatility_ok = (atr_pct >= 0.3 && atr_pct <= 2.0);
      
      int highest_idx = iHighest(_Symbol, _Period, MODE_HIGH, InpEMALength2, i);
      int lowest_idx  = iLowest(_Symbol, _Period, MODE_LOW, InpEMALength2, i);
      double h_val = (highest_idx != -1) ? iHigh(_Symbol, _Period, highest_idx) : high[i];
      double l_val = (lowest_idx != -1) ? iLow(_Symbol, _Period, lowest_idx) : low[i];
      
      double rng_pct = (close[i]!=0) ? ((h_val - l_val)/close[i] * 100) : 0;
      not_chop = (rng_pct >= 0.5);
     }

   double spread_thresh = atr_val * InpMinEMASpread;
   bool spread_ok = MathAbs(ema1[i] - ema2[i]) > spread_thresh;

   // Note: We don't have global buffers for 'last_buy' in EA easily without recreating indicator logic.
   // Simplification: Rely on 'bars_since_last' roughly or skip time filter for data collection if acceptable.
   // Or use a static variable.
   static int last_buy_bar = -999;
   static int last_sell_bar = -999;
   // Current bar absolute index
   int current_bar_idx = iBars(_Symbol, _Period) - i; 
   
   bool time_ok_buy = (current_bar_idx - last_buy_bar >= InpMinBarsBetween);
   bool time_ok_sell = (current_bar_idx - last_sell_bar >= InpMinBarsBetween);
   
   bool valid_buy = cross_up && trend_filter_ok && htf_trend_ok && distance_ok && volatility_ok && not_chop && time_ok_buy && spread_ok;
   bool valid_sell = cross_dn && trend_filter_ok && htf_trend_ok && distance_ok && volatility_ok && not_chop && time_ok_sell && spread_ok;

   // Reclaim Logic (Simplified State Machine)
   static int rec_state = 0;
   static double rec_ref = 0;
   
   if(valid_buy) { rec_state = 0; rec_ref = 0; }
   else if(cross_up) { rec_state = 1; rec_ref = high[i]; }
   
   if(valid_sell) { rec_state = 0; rec_ref = 0; }
   else if(cross_dn) { rec_state = -1; rec_ref = low[i]; }
   
   bool reclaim_buy_sig = false;
   bool reclaim_sell_sig = false;
   
   if(InpUseReclaim)
     {
      if(rec_state == 1 && trend_up && !valid_buy)
        {
         bool adx_ok = !InpReclaimStrict || (adx[i] > 20);
         bool breakout = (close[i] > rec_ref);
         if(rec_ref != 0) rec_ref = MathMax(rec_ref, high[i]); 
         if(breakout && adx_ok && (current_bar_idx - last_buy_bar >= InpMinBarsBetween * 2) && htf_trend_ok && spread_ok)
           {
            reclaim_buy_sig = true;
            rec_ref = high[i]; 
           }
        }
      else if(rec_state == -1 && !trend_up && !valid_sell)
        {
         bool adx_ok = !InpReclaimStrict || (adx[i] > 20);
         bool breakout = (close[i] < rec_ref);
         if(rec_ref != 0) rec_ref = MathMin(rec_ref, low[i]);
         if(breakout && adx_ok && (current_bar_idx - last_sell_bar >= InpMinBarsBetween * 2) && htf_trend_ok && spread_ok)
           {
            reclaim_sell_sig = true;
            rec_ref = low[i];
           }
        }
     }

   // 5. Execution
   if(valid_buy || reclaim_buy_sig)
     {
      last_buy_bar = current_bar_idx;
      double sl = close[i] - (atr_val * InpSLMultiplier);
      double tp = close[i] + (atr_val * InpTPMultiplier);
      
      SignalFeatures features;
      features.ema_short = ema1[i];
      features.ema_long = ema2[i];
      features.atr = atr_val;
      features.adx = adx[i];
      features.center = center;
      features.distance_ok = distance_ok;
      features.slope_ok = slope_ok;
      features.trend_filter_ok = trend_filter_ok;
      features.htf_trend_ok = htf_trend_ok;
      features.volatility_ok = volatility_ok;
      features.chop_ok = not_chop;
      features.spread_ok = spread_ok;
      features.bars_since_last = current_bar_idx - last_buy_bar; 
      features.trend_direction = trend_up ? 1 : 0;
      features.ema_cross_event = cross_up ? 1 : 0;
      features.ema_spread = ema1[i] - ema2[i];
      features.atr_percent = atr_pct;
      features.reclaim_state = rec_state;
      features.is_reclaim_signal = reclaim_buy_sig;
      features.price_vs_center = close[i] - center;
      features.cloud_width = MathAbs(ema1[i] - ema2[i]);

      RegisterVirtualTrade(valid_buy ? "BUY" : "RECLAIM_BUY", close[i], sl, tp, iTime(_Symbol, _Period, i), features);
     }
   else if(valid_sell || reclaim_sell_sig)
     {
      last_sell_bar = current_bar_idx;
      double sl = close[i] + (atr_val * InpSLMultiplier);
      double tp = close[i] - (atr_val * InpTPMultiplier);
      
      SignalFeatures features;
      features.ema_short = ema1[i];
      features.ema_long = ema2[i];
      features.atr = atr_val;
      features.adx = adx[i];
      features.center = center;
      features.distance_ok = distance_ok;
      features.slope_ok = slope_ok;
      features.trend_filter_ok = trend_filter_ok;
      features.htf_trend_ok = htf_trend_ok;
      features.volatility_ok = volatility_ok;
      features.chop_ok = not_chop;
      features.spread_ok = spread_ok;
      features.bars_since_last = current_bar_idx - last_sell_bar;
      features.trend_direction = trend_up ? 1 : 0;
      features.ema_cross_event = cross_dn ? -1 : 0;
      features.ema_spread = ema1[i] - ema2[i];
      features.atr_percent = atr_pct;
      features.reclaim_state = rec_state;
      features.is_reclaim_signal = reclaim_sell_sig;
      features.price_vs_center = close[i] - center;
      features.cloud_width = MathAbs(ema1[i] - ema2[i]);

      RegisterVirtualTrade(valid_sell ? "SELL" : "RECLAIM_SELL", close[i], sl, tp, iTime(_Symbol, _Period, i), features);
     }

   // 6. Manage Virtual Trades
   ManageVirtualTrades(high[i], low[i], iTime(_Symbol, _Period, i));
  }

//+------------------------------------------------------------------+
//| Helper: Calculate Pivot Center (Recursive approximation)         |
//+------------------------------------------------------------------+
double CalculatePivotCenter(double &high[], double &low[], int start_idx)
{
   int arr_size = ArraySize(high);
   int lookback = 50; 
   
   // Safety check for array bounds
   if(start_idx + lookback + InpPivotPeriod >= arr_size) {
       lookback = arr_size - start_idx - InpPivotPeriod - 1;
       if(lookback < 1) return (high[start_idx] + low[start_idx]) / 2.0;
   }

   double center = high[start_idx+lookback]; // Initial guess
   
   for(int k=lookback; k>=0; k--)
   {
      int idx = start_idx + k;
      
      // Check bounds for pivot neighbors
      if(idx - InpPivotPeriod < 0 || idx + InpPivotPeriod >= arr_size) continue;

      // Check Pivot High
      bool isPh = true;
      for(int p=1; p<=InpPivotPeriod; p++) if(high[idx+p] > high[idx] || high[idx-p] > high[idx]) isPh = false;
      
      // Check Pivot Low
      bool isPl = true;
      for(int p=1; p<=InpPivotPeriod; p++) if(low[idx+p] < low[idx] || low[idx-p] < low[idx]) isPl = false;
      
      double pp = EMPTY_VALUE;
      if(isPh) pp = high[idx];
      else if(isPl) pp = low[idx];
      
      if(pp != EMPTY_VALUE) center = (center * 2 + pp) / 3.0;
   }
   return center;
}

//+------------------------------------------------------------------+
//| Virtual Trade Logic                                              |
//+------------------------------------------------------------------+
void RegisterVirtualTrade(string action, double price, double sl, double tp, datetime time, SignalFeatures &features)
{
    int size = ArraySize(active_trades);
    ArrayResize(active_trades, size+1);
    
    trade_counter++;
    string sig_id = _Symbol + "_" + IntegerToString((long)time) + "_" + IntegerToString(trade_counter);
    
    active_trades[size].signal_id = sig_id;
    active_trades[size].symbol = _Symbol;
    active_trades[size].action = action;
    active_trades[size].entry_price = price;
    active_trades[size].sl = sl;
    active_trades[size].tp = tp;
    active_trades[size].open_time = time;
    active_trades[size].active = true;
    active_trades[size].features = features;
    
    WriteSignalEvent(active_trades[size]);
}

void ManageVirtualTrades(double high, double low, datetime current_time)
{
    for(int i=0; i<ArraySize(active_trades); i++)
    {
        if(!active_trades[i].active) continue;
        if(active_trades[i].open_time >= current_time) continue;
        
        bool closed = false;
        int outcome = 0;
        double exit_price = 0;
        
        if(StringFind(active_trades[i].action, "BUY") >= 0) {
            if(low <= active_trades[i].sl) { outcome=0; exit_price=active_trades[i].sl; closed=true; }
            else if(high >= active_trades[i].tp) { outcome=1; exit_price=active_trades[i].tp; closed=true; }
        } else {
            if(high >= active_trades[i].sl) { outcome=0; exit_price=active_trades[i].sl; closed=true; }
            else if(low <= active_trades[i].tp) { outcome=1; exit_price=active_trades[i].tp; closed=true; }
        }
        
        if(closed) {
            active_trades[i].active = false;
            WriteTradeOutcome(active_trades[i], outcome, exit_price, current_time);
        }
    }
}

void WriteSignalEvent(VirtualTrade &trade)
{
   string filename = "AlphaOS\\Signals\\signals_" + _Symbol + ".json"; 
   int file_handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(file_handle != INVALID_HANDLE) {
       FileSeek(file_handle, 0, SEEK_END);
       string json = StringFormat("{\"type\":\"SIGNAL\",\"signal_id\":\"%s\",\"action\":\"%s\",\"price\":%.5f,\"timestamp\":%d}\n", 
                                  trade.signal_id, trade.action, trade.entry_price, (int)trade.open_time);
       // Note: Full JSON construction omitted for brevity in this EA conversion example, 
       // but logically you should include ALL features here as in the indicator version.
       // I will assume the full JSON format is used.
       
       // Re-constructing full JSON for safety:
       json = StringFormat(
          "{\"type\":\"SIGNAL\",\"signal_id\":\"%s\",\"symbol\":\"%s\",\"action\":\"%s\",\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,"
          "\"ema_short\":%.5f,\"ema_long\":%.5f,\"atr\":%.5f,\"adx\":%.5f,\"center\":%.5f,"
          "\"distance_ok\":%d,\"slope_ok\":%d,\"trend_filter_ok\":%d,\"htf_trend_ok\":%d,"
          "\"volatility_ok\":%d,\"chop_ok\":%d,\"spread_ok\":%d,"
          "\"bars_since_last\":%d,\"trend_direction\":%d,\"ema_cross_event\":%d,"
          "\"ema_spread\":%.5f,\"atr_percent\":%.4f,\"reclaim_state\":%d,"
          "\"is_reclaim_signal\":%d,\"price_vs_center\":%.5f,\"cloud_width\":%.5f,"
          "\"timestamp\":%d}\n",
          trade.signal_id, trade.symbol, trade.action, trade.entry_price, trade.sl, trade.tp,
          trade.features.ema_short, trade.features.ema_long, trade.features.atr, trade.features.adx, trade.features.center,
          trade.features.distance_ok, trade.features.slope_ok, trade.features.trend_filter_ok, trade.features.htf_trend_ok,
          trade.features.volatility_ok, trade.features.chop_ok, trade.features.spread_ok,
          trade.features.bars_since_last, trade.features.trend_direction, trade.features.ema_cross_event,
          trade.features.ema_spread, trade.features.atr_percent, trade.features.reclaim_state,
          trade.features.is_reclaim_signal, trade.features.price_vs_center, trade.features.cloud_width,
          (int)trade.open_time
       );

       FileWriteString(file_handle, json);
       FileClose(file_handle);
       Print("✅ Signal: ", trade.action, " @ ", trade.entry_price);
   }
}

void WriteTradeOutcome(VirtualTrade &trade, int outcome, double exit_price, datetime close_time)
{
   string filename = "AlphaOS\\Signals\\outcomes_" + _Symbol + ".json"; 
   int file_handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(file_handle != INVALID_HANDLE) {
       FileSeek(file_handle, 0, SEEK_END);
       string json = StringFormat("{\"type\":\"OUTCOME\",\"signal_id\":\"%s\",\"outcome\":%d,\"exit_price\":%.5f,\"close_time\":%d}\n",
          trade.signal_id, outcome, exit_price, (int)close_time);
       FileWriteString(file_handle, json);
       FileClose(file_handle);
       Print("💰 Outcome: ", outcome);
   }
}
