//+------------------------------------------------------------------+
//|                                       PivotTrend_DataCollector.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                      Adapted from Pine Script V3 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.10"
#property description "Collects Signals AND Outcomes for ML Training"
#property indicator_chart_window
#property indicator_buffers 18 
#property indicator_plots   6

//--- Plot settings (Same as before)
#property indicator_label1  "Cloud Fill"
#property indicator_type1   DRAW_FILLING
#property indicator_color1  clrWhite, clrYellow 
#property indicator_width1  1

#property indicator_label2  "EMA Short"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrWhite
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

#property indicator_label3  "EMA Long"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrYellow
#property indicator_style3  STYLE_SOLID
#property indicator_width3  2

#property indicator_label4  "Center Line"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrAqua 
#property indicator_style4  STYLE_DOT
#property indicator_width4  1

#property indicator_label5  "Buy Arrow"
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrSpringGreen
#property indicator_width5  2

#property indicator_label6  "Sell Arrow"
#property indicator_type6   DRAW_ARROW
#property indicator_color6  clrRed
#property indicator_width6  2

//--- Enums
enum ENUM_FILTER_MODE {
   FILTER_NONE,   // No Filter
   FILTER_BASIC,  // Basic Filter (+Trend)
   FILTER_STRICT  // Strict Filter (+Vol+Chop)
};

enum ENUM_LABEL_SIZE {
   SIZE_SMALL,
   SIZE_NORMAL,
   SIZE_LARGE
};

//--- Inputs
input group "=== Core Settings ==="
input int      InpPivotPeriod = 2;     // Pivot Point Period
input int      InpATRPeriod   = 14;    // ATR Period
input int      InpEMALength1  = 6;     // EMA Short Period
input int      InpEMALength2  = 24;    // EMA Long Period

input group "=== Filter Settings ==="
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
input string   InpTPBase         = "SignalClose"; 

input group "=== Appearance ==="
input bool     InpShowCenter     = true;  // Show Center Line (Default True)
input bool     InpShowHistory    = true;  // Show History Labels
input color    InpColorBuy       = clrSpringGreen; // Buy Color
input color    InpColorSell      = clrRed;         // Sell Color
input double   InpLabelOffset    = 1.0;   // Label Offset (ATR Multiplier)
input ENUM_LABEL_SIZE InpLabelSize = SIZE_NORMAL; // Label Size
input int      InpCloudAlpha     = 70;    // Cloud Transparency (0-255)

//--- Buffers
double         BufferFill1[]; 
double         BufferFill2[]; 
double         BufferEMA1[];  
double         BufferEMA2[];  
double         BufferCenter[];
double         BufferBuy[];
double         BufferSell[];

double         CalcCenter[];
double         CalcReclaimState[]; 
double         CalcReclaimPrice[];
double         CalcLastBuyBar[];
double         CalcLastSellBar[];
double         CalcATR[];
double         CalcADX[];
double         CalcSlopeOk[];
double         CalcSpreadOk[];
double         CalcHigh[]; 
double         CalcLow[];

//--- Handles
int            hEMA1;
int            hEMA2;
int            hATR;
int            hADX;
int            hHTF_EMA1;
int            hHTF_EMA2;

//--- Globals
string         Prefix = "AlphaOS_DC_"; 
string         LastLabelName = "";
int            last_signal_index = 0;

//--- Virtual Trade Tracking
struct VirtualTrade {
    int      id;
    string   symbol;
    string   action; // "BUY", "SELL", "RECLAIM_BUY", "RECLAIM_SELL"
    double   entry_price;
    double   sl;
    double   tp;
    datetime open_time;
    
    // Snapshot Features at Entry
    double   ema_short;
    double   ema_long;
    double   atr;
    double   adx;
    double   center;
    bool     distance_ok;
    bool     slope_ok;
    bool     trend_filter_ok;
    bool     htf_trend_ok;
    int      trend_direction; // 1=UP, -1=DN
    bool     volatility_ok;
    bool     spread_ok;
    double   cloud_width;
    
    bool     active;
};

VirtualTrade active_trades[];
int          trade_counter = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, BufferFill1, INDICATOR_DATA);
   SetIndexBuffer(1, BufferFill2, INDICATOR_DATA);
   SetIndexBuffer(2, BufferEMA1, INDICATOR_DATA);
   SetIndexBuffer(3, BufferEMA2, INDICATOR_DATA);
   SetIndexBuffer(4, BufferCenter, INDICATOR_DATA);
   SetIndexBuffer(5, BufferBuy, INDICATOR_DATA);
   SetIndexBuffer(6, BufferSell, INDICATOR_DATA);
   
   SetIndexBuffer(7, CalcCenter, INDICATOR_CALCULATIONS);
   SetIndexBuffer(8, CalcReclaimState, INDICATOR_CALCULATIONS);
   SetIndexBuffer(9, CalcReclaimPrice, INDICATOR_CALCULATIONS);
   SetIndexBuffer(10, CalcLastBuyBar, INDICATOR_CALCULATIONS);
   SetIndexBuffer(11, CalcLastSellBar, INDICATOR_CALCULATIONS);
   SetIndexBuffer(12, CalcATR, INDICATOR_CALCULATIONS);
   SetIndexBuffer(13, CalcADX, INDICATOR_CALCULATIONS);
   SetIndexBuffer(14, CalcSlopeOk, INDICATOR_CALCULATIONS);
   SetIndexBuffer(15, CalcSpreadOk, INDICATOR_CALCULATIONS);
   SetIndexBuffer(16, CalcHigh, INDICATOR_CALCULATIONS);
   SetIndexBuffer(17, CalcLow, INDICATOR_CALCULATIONS);

   PlotIndexSetInteger(0, PLOT_SHOW_DATA, false); 
   PlotIndexSetInteger(0, PLOT_COLOR_INDEXES, 2);
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, 0, C'0,64,0'); 
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, 1, C'64,0,0'); 
   
   PlotIndexSetInteger(4, PLOT_ARROW, 233); 
   PlotIndexSetInteger(5, PLOT_ARROW, 234); 
   
   if(!InpShowCenter) PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_NONE);

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

   ObjectsDeleteAll(0, Prefix);
   Print("PivotTrend Data Collector V2 (With Outcomes) Initialized");
   
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, Prefix);
  }

//--- Helper Functions
bool IsPivotHigh(const double &high[], int i, int prd, int total) {
   if (i < prd || i + prd >= total) return false; 
   double centerVal = high[i];
   for(int k = 1; k <= prd; k++) {
      if(high[i-k] > centerVal) return false; 
      if(high[i+k] >= centerVal) return false; 
   }
   return true;
}

bool IsPivotLow(const double &low[], int i, int prd, int total) {
   if (i < prd || i + prd >= total) return false;
   double centerVal = low[i];
   for(int k = 1; k <= prd; k++) {
      if(low[i-k] < centerVal) return false;
      if(low[i+k] <= centerVal) return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(rates_total < InpEMALength2 + 50) return(0);

   double ema1[], ema2[], atr[], adx[];
   CopyBuffer(hEMA1, 0, 0, rates_total, ema1);
   CopyBuffer(hEMA2, 0, 0, rates_total, ema2);
   CopyBuffer(hATR, 0, 0, rates_total, atr);
   CopyBuffer(hADX, 0, 0, rates_total, adx); 

   int start = prev_calculated - 1;
   if(start < InpEMALength2) start = InpEMALength2;
   if(start < InpPivotPeriod * 2) start = InpPivotPeriod * 2;
   
   // --- Manage Virtual Trades (Check outcomes for existing trades) ---
   // We only check trades once per bar close to avoid noise, or on every tick if needed.
   // For indicators, it's safer to check on every call but only process the 'latest' data if we are real-time.
   // For historical calculation, we iterate 'i'.
   
   // Note: Managing a dynamic list of trades inside OnCalculate history loop is tricky because 
   // OnCalculate might re-run for the same bars. 
   // SIMPLIFICATION: We only process trade logic for the *current* bar 'i' being calculated.
   // But we need to persist the trades. The `active_trades` array is global.
   // ISSUE: If we re-calculate history, we might duplicate trades.
   // FIX: Clear trades if prev_calculated == 0.
   
   if(prev_calculated == 0) {
       ArrayResize(active_trades, 0);
       trade_counter = 0;
   }

   for(int i = start; i < rates_total; i++)
     {
      // 0. Update Active Trades (Check TP/SL) based on CURRENT High/Low
      int total_trades = ArraySize(active_trades);
      for(int t=total_trades-1; t>=0; t--) {
          if(!active_trades[t].active) continue;
          
          // We can only check outcome if the trade was opened BEFORE this bar 'i'
          // or if we want to simulate intra-bar (risky).
          // Let's assume we check outcome against the current bar 'i' price action.
          // But trade must have been opened at 'i-1' or earlier.
          
          if(active_trades[t].open_time >= time[i]) continue; // Opened on this bar, skip
          
          bool closed = false;
          int outcome = 0; // 0=Loss, 1=Win
          double exit_price = 0;
          
          // Check Buy
          if(StringFind(active_trades[t].action, "BUY") >= 0) {
              if(low[i] <= active_trades[t].sl) {
                  outcome = 0;
                  exit_price = active_trades[t].sl;
                  closed = true;
              }
              else if(high[i] >= active_trades[t].tp) {
                  outcome = 1;
                  exit_price = active_trades[t].tp;
                  closed = true;
              }
          }
          // Check Sell
          else {
              if(high[i] >= active_trades[t].sl) {
                  outcome = 0; // Hit Stop
                  exit_price = active_trades[t].sl;
                  closed = true;
              }
              else if(low[i] <= active_trades[t].tp) {
                  outcome = 1; // Hit TP
                  exit_price = active_trades[t].tp;
                  closed = true;
              }
          }
          
          if(closed) {
              active_trades[t].active = false;
              // Write to File
              WriteTradeResult(active_trades[t], outcome, exit_price, time[i]);
          }
      }

      // 1. Store Data & Fill Visuals
      BufferEMA1[i] = ema1[i];
      BufferEMA2[i] = ema2[i];
      BufferFill1[i] = ema1[i];
      BufferFill2[i] = ema2[i];
      
      CalcATR[i]    = atr[i];
      CalcADX[i]    = adx[i];
      CalcHigh[i]   = high[i];
      CalcLow[i]    = low[i];

      // 2. Pivot & Center
      double last_center = (i > 0) ? CalcCenter[i-1] : 0;
      double new_center = last_center;
      
      int check_idx = i - InpPivotPeriod;
      if(check_idx >= InpPivotPeriod) 
        {
         bool isPh = IsPivotHigh(high, check_idx, InpPivotPeriod, rates_total); 
         bool isPl = IsPivotLow(low, check_idx, InpPivotPeriod, rates_total);
         
         double found_pp = EMPTY_VALUE;
         if(isPh) found_pp = high[check_idx];
         else if(isPl) found_pp = low[check_idx];
         
         if(found_pp != EMPTY_VALUE)
           {
            if(last_center == 0 || last_center == EMPTY_VALUE) new_center = found_pp;
            else new_center = (last_center * 2 + found_pp) / 3.0;
           }
        }
      CalcCenter[i] = new_center;
      BufferCenter[i] = new_center;

      // 3. Trend Logic
      bool trend_up = ema1[i] > ema2[i];
      bool cross_up = (ema1[i] > ema2[i]) && (ema1[i-1] <= ema2[i-1]);
      bool cross_dn = (ema1[i] < ema2[i]) && (ema1[i-1] >= ema2[i-1]);

      // 4. Filters
      double center = CalcCenter[i];
      double atr_val = atr[i];
      double dist = (atr_val != 0) ? MathAbs(close[i] - center) / atr_val : 0;
      
      double slope = (i>0) ? MathAbs(ema1[i] - ema1[i-1]) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) : 0;
      bool slope_val_ok = slope > InpSlopeThreshold;
      bool prev_slope = (i>1) ? MathAbs(ema1[i-1] - ema1[i-2]) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) : 0;
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
         int htf_idx = iBarShift(_Symbol, InpHTFPeriod, time[i], false);
         if(htf_idx >= 0)
           {
            double h_ema1[], h_ema2[];
            CopyBuffer(hHTF_EMA1, 0, htf_idx, 1, h_ema1); 
            CopyBuffer(hHTF_EMA2, 0, htf_idx, 1, h_ema2);
            if(ArraySize(h_ema1)>0 && ArraySize(h_ema2)>0)
              {
               bool htf_up = h_ema1[0] > h_ema2[0];
               htf_trend_ok = (trend_up && htf_up) || (!trend_up && !htf_up);
              }
           }
        }

      double atr_pct = (close[i] != 0) ? (atr_val / close[i] * 100) : 0;
      bool volatility_ok = true;
      if(InpFilterMode == FILTER_STRICT) volatility_ok = (atr_pct >= 0.3 && atr_pct <= 2.0);

      double spread_thresh = atr_val * InpMinEMASpread;
      bool spread_ok = MathAbs(ema1[i] - ema2[i]) > spread_thresh;
      CalcSpreadOk[i] = spread_ok;

      // 5. Signal Generation
      int last_buy = (i>0) ? (int)CalcLastBuyBar[i-1] : -100;
      int last_sell = (i>0) ? (int)CalcLastSellBar[i-1] : -100;
      
      bool time_ok_buy = (i - last_buy >= InpMinBarsBetween);
      bool time_ok_sell = (i - last_sell >= InpMinBarsBetween);
      
      bool valid_buy = cross_up && trend_filter_ok && htf_trend_ok && distance_ok && volatility_ok && time_ok_buy && spread_ok;
      bool valid_sell = cross_dn && trend_filter_ok && htf_trend_ok && distance_ok && volatility_ok && time_ok_sell && spread_ok;

      // 6. Reclaim Logic (Simplified for Brevity)
      int rec_state = (i>0) ? (int)CalcReclaimState[i-1] : 0; 
      double rec_ref = (i>0) ? CalcReclaimPrice[i-1] : 0;
      
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
            if(breakout && adx_ok && (i - last_buy >= InpMinBarsBetween * 2) && htf_trend_ok && spread_ok) {
               reclaim_buy_sig = true;
               rec_ref = high[i]; 
            }
           }
         else if(rec_state == -1 && !trend_up && !valid_sell)
           {
            bool adx_ok = !InpReclaimStrict || (adx[i] > 20);
            bool breakout = (close[i] < rec_ref);
            if(rec_ref != 0) rec_ref = MathMin(rec_ref, low[i]);
            if(breakout && adx_ok && (i - last_sell >= InpMinBarsBetween * 2) && htf_trend_ok && spread_ok) {
               reclaim_sell_sig = true;
               rec_ref = low[i];
            }
           }
        }

      CalcReclaimState[i] = rec_state;
      CalcReclaimPrice[i] = rec_ref;

      // 7. Register Signals and Open Virtual Trades
      BufferBuy[i] = EMPTY_VALUE;
      BufferSell[i] = EMPTY_VALUE;
      
      string signal_action = "";
      bool is_buy = false;
      
      if(valid_buy || reclaim_buy_sig)
        {
         BufferBuy[i] = low[i];
         CalcLastBuyBar[i] = i;
         CalcLastSellBar[i] = last_sell; 
         signal_action = valid_buy ? "BUY" : "RECLAIM_BUY";
         is_buy = true;
         
         // Open Virtual Trade
         double sl = close[i] - (atr_val * InpSLMultiplier);
         double tp = close[i] + (atr_val * InpTPMultiplier);
         RegisterVirtualTrade(i, signal_action, close[i], sl, tp, time[i], 
                              ema1[i], ema2[i], atr_val, adx[i], center, 
                              distance_ok, slope_ok, trend_filter_ok, htf_trend_ok, 1, volatility_ok, spread_ok);
        }
      else if(valid_sell || reclaim_sell_sig)
        {
         BufferSell[i] = high[i];
         CalcLastSellBar[i] = i;
         CalcLastBuyBar[i] = last_buy; 
         signal_action = valid_sell ? "SELL" : "RECLAIM_SELL";
         is_buy = false;
         
         double sl = close[i] + (atr_val * InpSLMultiplier);
         double tp = close[i] - (atr_val * InpTPMultiplier);
         RegisterVirtualTrade(i, signal_action, close[i], sl, tp, time[i], 
                              ema1[i], ema2[i], atr_val, adx[i], center, 
                              distance_ok, slope_ok, trend_filter_ok, htf_trend_ok, -1, volatility_ok, spread_ok);
        }
      else
        {
         CalcLastBuyBar[i] = last_buy;
         CalcLastSellBar[i] = last_sell;
        }
        
      if(signal_action != "")
        {
         CreateLabel(time[i], is_buy ? low[i] : high[i], signal_action, is_buy ? InpColorBuy : InpColorSell, is_buy);
        }
     }
     
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Register Virtual Trade                                           |
//+------------------------------------------------------------------+
void RegisterVirtualTrade(int index, string action, double price, double sl, double tp, datetime time,
                          double ema_s, double ema_l, double atr, double adx, double center,
                          bool dist_ok, bool slope_ok, bool trend_ok, bool htf_ok, int direction, bool vol_ok, bool spr_ok)
{
    int size = ArraySize(active_trades);
    ArrayResize(active_trades, size+1);
    
    active_trades[size].id = ++trade_counter;
    active_trades[size].symbol = _Symbol;
    active_trades[size].action = action;
    active_trades[size].entry_price = price;
    active_trades[size].sl = sl;
    active_trades[size].tp = tp;
    active_trades[size].open_time = time;
    active_trades[size].active = true;
    
    // Features
    active_trades[size].ema_short = ema_s;
    active_trades[size].ema_long = ema_l;
    active_trades[size].atr = atr;
    active_trades[size].adx = adx;
    active_trades[size].center = center;
    active_trades[size].distance_ok = dist_ok;
    active_trades[size].slope_ok = slope_ok;
    active_trades[size].trend_filter_ok = trend_ok;
    active_trades[size].htf_trend_ok = htf_ok;
    active_trades[size].trend_direction = direction;
    active_trades[size].volatility_ok = vol_ok;
    active_trades[size].spread_ok = spr_ok;
    active_trades[size].cloud_width = MathAbs(ema_s - ema_l);
}

//+------------------------------------------------------------------+
//| Write Result to JSON                                             |
//+------------------------------------------------------------------+
void WriteTradeResult(VirtualTrade &trade, int outcome, double exit_price, datetime close_time)
{
   string filename = "AlphaOS\\Signals\\training_data_" + _Symbol + ".json";
   int file_handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI);
   
   if(file_handle != INVALID_HANDLE)
     {
      FileSeek(file_handle, 0, SEEK_END); // Append
      
      string json = StringFormat(
         "{\"symbol\":\"%s\",\"action\":\"%s\",\"outcome\":%d,\"entry_price\":%.5f,\"exit_price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,"
         "\"ema_short\":%.5f,\"ema_long\":%.5f,\"atr\":%.5f,\"adx\":%.5f,\"center\":%.5f,"
         "\"distance_ok\":%d,\"slope_ok\":%d,\"trend_filter_ok\":%d,\"htf_trend_ok\":%d,"
         "\"trend_direction\":%d,\"volatility_ok\":%d,\"spread_ok\":%d,\"cloud_width\":%.5f,"
         "\"open_time\":%d,\"close_time\":%d}\n", // Add newline for line-by-line JSON reading
         trade.symbol, trade.action, outcome, trade.entry_price, exit_price, trade.sl, trade.tp,
         trade.ema_short, trade.ema_long, trade.atr, trade.adx, trade.center,
         trade.distance_ok ? 1 : 0, trade.slope_ok ? 1 : 0, trade.trend_filter_ok ? 1 : 0, trade.htf_trend_ok ? 1 : 0,
         trade.trend_direction, trade.volatility_ok ? 1 : 0, trade.spread_ok ? 1 : 0, trade.cloud_width,
         (int)trade.open_time, (int)close_time
      );
      
      FileWriteString(file_handle, json);
      FileClose(file_handle);
     }
}

//+------------------------------------------------------------------+
//| Draw Label Object                                                |
//+------------------------------------------------------------------+
void CreateLabel(datetime time, double price, string text, color clr, bool up)
  {
   string name = Prefix + TimeToString(time);
   
   if(!InpShowHistory)
     {
      if(LastLabelName != "" && LastLabelName != name) ObjectDelete(0, LastLabelName);
     }
     
   if(ObjectFind(0, name) >= 0) return;
   
   ObjectCreate(0, name, OBJ_TEXT, 0, time, price);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpLabelSize == SIZE_SMALL ? 8 : 10);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, up ? ANCHOR_TOP : ANCHOR_BOTTOM);
   
   LastLabelName = name;
  }
