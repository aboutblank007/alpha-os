//+------------------------------------------------------------------+
//|                                            PivotTrendSignals.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                      Adapted from Pine Script V3 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "2.10"
#property indicator_chart_window
#property indicator_buffers 18 
#property indicator_plots   6

//--- Plot settings

// Plot 1: Cloud Filling (Must be first for correct layering usually, but index matters)
// MQL5 DRAW_FILLING uses 2 buffers.
#property indicator_label1  "Cloud Fill"
#property indicator_type1   DRAW_FILLING
#property indicator_color1  clrWhite, clrYellow // Colors for Up/Down fills? Actually filling takes 2 colors usually
// Note: MQL5 DRAW_FILLING takes 2 buffers. Color logic:
// If Buffer1 > Buffer2 -> Color1
// If Buffer1 < Buffer2 -> Color2
#property indicator_width1  1

// Plot 2: EMA Short Line
#property indicator_label2  "EMA Short"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrWhite
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

// Plot 3: EMA Long Line
#property indicator_label3  "EMA Long"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrYellow
#property indicator_style3  STYLE_SOLID
#property indicator_width3  2

// Plot 4: Center Line
#property indicator_label4  "Center Line"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrAqua // Changed to visible color
#property indicator_style4  STYLE_DOT
#property indicator_width4  1

// Plot 5: Buy Arrow
#property indicator_label5  "Buy Arrow"
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrSpringGreen
#property indicator_width5  2

// Plot 6: Sell Arrow
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
input string   InpTPBase         = "SignalClose"; // TP/SL Base (Simplified to SignalClose)

input group "=== Appearance ==="
input bool     InpShowCenter     = true;  // Show Center Line (Default True)
input bool     InpShowHistory    = true;  // Show History Labels
input color    InpColorBuy       = clrSpringGreen; // Buy Color
input color    InpColorSell      = clrRed;         // Sell Color
input double   InpLabelOffset    = 1.0;   // Label Offset (ATR Multiplier)
input ENUM_LABEL_SIZE InpLabelSize = SIZE_NORMAL; // Label Size
input int      InpCloudAlpha     = 70;    // Cloud Transparency (0-255) - Not directly supported in MQL5 standard, handled via Color

//--- Buffers
// Visual Buffers
double         BufferFill1[]; // For Cloud
double         BufferFill2[]; // For Cloud
double         BufferEMA1[];  // Line
double         BufferEMA2[];  // Line
double         BufferCenter[];
double         BufferBuy[];
double         BufferSell[];

// Calculation Buffers (Hidden)
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
string         Prefix = "AlphaOS_Label_";
string         LastLabelName = "";

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   //--- Indicator Buffers Mapping
   // Plot 1 (Filling): Buffers 0, 1
   SetIndexBuffer(0, BufferFill1, INDICATOR_DATA);
   SetIndexBuffer(1, BufferFill2, INDICATOR_DATA);
   
   // Plot 2 (EMA1): Buffer 2
   SetIndexBuffer(2, BufferEMA1, INDICATOR_DATA);
   
   // Plot 3 (EMA2): Buffer 3
   SetIndexBuffer(3, BufferEMA2, INDICATOR_DATA);
   
   // Plot 4 (Center): Buffer 4
   SetIndexBuffer(4, BufferCenter, INDICATOR_DATA);
   
   // Plot 5 (Buy): Buffer 5
   SetIndexBuffer(5, BufferBuy, INDICATOR_DATA);
   
   // Plot 6 (Sell): Buffer 6
   SetIndexBuffer(6, BufferSell, INDICATOR_DATA);
   
   // Calculation Buffers
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

   //--- Plot Settings
   
   // Filling Colors (Transparency simulated via color selection if possible, but MQL5 standard colors are solid usually)
   // Using dim colors for filling
   PlotIndexSetInteger(0, PLOT_SHOW_DATA, false); // Don't show filling values in Data Window
   // Setting fill colors: Color1 for Buf1 > Buf2 (Trend UP), Color2 for Buf1 < Buf2 (Trend DOWN)
   // EMA1 (Short) > EMA2 (Long) -> Trend UP.
   // So if BufferFill1=EMA1, BufferFill2=EMA2:
   // 1 > 2 -> Color1. We want Greenish for Up.
   // 1 < 2 -> Color2. We want Reddish for Down.
   PlotIndexSetInteger(0, PLOT_COLOR_INDEXES, 2);
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, 0, C'0,64,0'); // Dark Green for Up Fill
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, 1, C'64,0,0'); // Dark Red for Down Fill
   
   PlotIndexSetInteger(4, PLOT_ARROW, 233); // Wingdings Up
   PlotIndexSetInteger(5, PLOT_ARROW, 234); // Wingdings Down
   
   if(!InpShowCenter) PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_NONE); // Plot 4 is index 3? No.
   // Plot indices start at 0.
   // Plot 0: Fill
   // Plot 1: EMA1
   // Plot 2: EMA2
   // Plot 3: Center
   // Plot 4: Buy
   // Plot 5: Sell
   if(!InpShowCenter) PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_NONE);

   //--- Initialize Handles
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
   
   Print("PivotTrendSignals initialized on ", _Symbol, " ", EnumToString(_Period));
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Deinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, Prefix);
  }

//+------------------------------------------------------------------+
//| Helper: Check Pivot                                              |
//+------------------------------------------------------------------+
bool IsPivotHigh(const double &high[], int i, int prd, int total)
  {
   if (i < prd || i + prd >= total) return false; 
   double centerVal = high[i];
   for(int k = 1; k <= prd; k++)
     {
      if(i - k < 0 || i + k >= total) return false;
      if(high[i-k] > centerVal) return false; 
      if(high[i+k] >= centerVal) return false; 
     }
   return true;
  }

bool IsPivotLow(const double &low[], int i, int prd, int total)
  {
   if (i < prd || i + prd >= total) return false;
   double centerVal = low[i];
   for(int k = 1; k <= prd; k++)
     {
      if(i - k < 0 || i + k >= total) return false;
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

   for(int i = start; i < rates_total; i++)
     {
      // 1. Store Data & Fill Visuals
      BufferEMA1[i] = ema1[i];
      BufferEMA2[i] = ema2[i];
      
      // Fill Buffers (Same as EMAs)
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
      double prev_slope = (i>1) ? MathAbs(ema1[i-1] - ema1[i-2]) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) : 0;
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

      bool volatility_ok = true;
      bool not_chop = true;
      if(InpFilterMode == FILTER_STRICT)
        {
         double atr_pct = (close[i] != 0) ? (atr_val / close[i] * 100) : 0;
         volatility_ok = (atr_pct >= 0.3 && atr_pct <= 2.0);
         
         int highest_idx = iHighest(_Symbol, _Period, MODE_HIGH, InpEMALength2, i - InpEMALength2 + 1);
         int lowest_idx  = iLowest(_Symbol, _Period, MODE_LOW, InpEMALength2, i - InpEMALength2 + 1);
         double h_val = (highest_idx != -1) ? high[highest_idx] : high[i];
         double l_val = (lowest_idx != -1) ? low[lowest_idx] : low[i];
         
         double rng_pct = (close[i]!=0) ? ((h_val - l_val)/close[i] * 100) : 0;
         not_chop = (rng_pct >= 0.5);
        }

      double spread_thresh = atr_val * InpMinEMASpread;
      bool spread_ok = MathAbs(ema1[i] - ema2[i]) > spread_thresh;
      CalcSpreadOk[i] = spread_ok;

      // 5. Signal Generation
      int last_buy = (i>0) ? (int)CalcLastBuyBar[i-1] : -100;
      int last_sell = (i>0) ? (int)CalcLastSellBar[i-1] : -100;
      
      bool time_ok_buy = (i - last_buy >= InpMinBarsBetween);
      bool time_ok_sell = (i - last_sell >= InpMinBarsBetween);
      
      bool valid_buy = cross_up && trend_filter_ok && htf_trend_ok && distance_ok && volatility_ok && not_chop && time_ok_buy && spread_ok;
      bool valid_sell = cross_dn && trend_filter_ok && htf_trend_ok && distance_ok && volatility_ok && not_chop && time_ok_sell && spread_ok;

      // 6. Reclaim Logic
      int rec_state = (i>0) ? (int)CalcReclaimState[i-1] : 0; 
      double rec_ref = (i>0) ? CalcReclaimPrice[i-1] : 0;
      
      if(valid_buy)
        {
         rec_state = 0; 
         rec_ref = 0;
        }
      else if(cross_up)
        {
         rec_state = 1; 
         rec_ref = high[i];
        }
        
      if(valid_sell)
        {
         rec_state = 0;
         rec_ref = 0;
        }
      else if(cross_dn)
        {
         rec_state = -1; 
         rec_ref = low[i];
        }
        
      bool reclaim_buy_sig = false;
      bool reclaim_sell_sig = false;
      
      if(InpUseReclaim)
        {
         if(rec_state == 1 && trend_up && !valid_buy)
           {
            bool adx_ok = !InpReclaimStrict || (adx[i] > 20);
            bool breakout = (close[i] > rec_ref);
            
            if(rec_ref != 0) rec_ref = MathMax(rec_ref, high[i]); 
            
            bool rec_time_ok = (i - last_buy >= InpMinBarsBetween * 2);
            
            if(breakout && adx_ok && rec_time_ok && htf_trend_ok && spread_ok)
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
            
            bool rec_time_ok = (i - last_sell >= InpMinBarsBetween * 2);
            
            if(breakout && adx_ok && rec_time_ok && htf_trend_ok && spread_ok)
              {
               reclaim_sell_sig = true;
               rec_ref = low[i];
              }
           }
        }

      CalcReclaimState[i] = rec_state;
      CalcReclaimPrice[i] = rec_ref;

      // 7. Finalize Output & Draw
      BufferBuy[i] = EMPTY_VALUE;
      BufferSell[i] = EMPTY_VALUE;
      
      string signal_txt = "";
      color signal_clr = clrNONE;
      bool is_buy = false;
      
      // Offset for Arrow (closer to candle)
      double arrow_offset = atr_val * 0.5; 
      // Offset for Text (further from candle)
      double text_offset = atr_val * InpLabelOffset;
      
      if(valid_buy || reclaim_buy_sig)
        {
         BufferBuy[i] = low[i] - arrow_offset;
         CalcLastBuyBar[i] = i;
         CalcLastSellBar[i] = last_sell; 
         
         signal_txt = valid_buy ? "买入" : "Reclaim";
         signal_clr = InpColorBuy;
         is_buy = true;
         
         // Write signal only for CONFIRMED bar (rates_total - 2, not the current forming bar)
         // This matches Pine Script behavior where signals confirm on bar close
         if(i == rates_total - 2)
           {
            double sl = close[i] - (atr_val * InpSLMultiplier);
            double tp = close[i] + (atr_val * InpTPMultiplier);
            WriteSignalToFile(valid_buy ? "BUY" : "RECLAIM_BUY", close[i], sl, tp, signal_txt);
           }
        }
      else if(valid_sell || reclaim_sell_sig)
        {
         BufferSell[i] = high[i] + arrow_offset;
         CalcLastSellBar[i] = i;
         CalcLastBuyBar[i] = last_buy; 
         
         signal_txt = valid_sell ? "卖出" : "Reclaim";
         signal_clr = InpColorSell;
         is_buy = false;
         
         // Write signal only for CONFIRMED bar (rates_total - 2, not the current forming bar)
         if(i == rates_total - 2)
           {
            double sl = close[i] + (atr_val * InpSLMultiplier);
            double tp = close[i] - (atr_val * InpTPMultiplier);
            WriteSignalToFile(valid_sell ? "SELL" : "RECLAIM_SELL", close[i], sl, tp, signal_txt);
           }
        }
      else
        {
         CalcLastBuyBar[i] = last_buy;
         CalcLastSellBar[i] = last_sell;
        }
        
      // Draw Labels (Object)
      if(signal_txt != "")
        {
         // Text Position:
         // Buy: Low - TextOffset. Anchor Top (Text hangs down).
         // Sell: High + TextOffset. Anchor Bottom (Text stands up).
         CreateLabel(time[i], is_buy ? low[i] - text_offset : high[i] + text_offset, 
                     signal_txt + " @ " + DoubleToString(close[i], _Digits), 
                     signal_clr, is_buy);
        }
     }
     
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Draw Label Object                                                |
//+------------------------------------------------------------------+
void CreateLabel(datetime time, double price, string text, color clr, bool up)
  {
   string name = Prefix + TimeToString(time);
   
   if(!InpShowHistory)
     {
      if(LastLabelName != "" && LastLabelName != name)
        {
         ObjectDelete(0, LastLabelName);
        }
     }
     
   if(ObjectFind(0, name) >= 0) 
     {
      ObjectSetString(0, name, OBJPROP_TEXT, text);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
      return;
     }
   
   ObjectCreate(0, name, OBJ_TEXT, 0, time, price);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpLabelSize == SIZE_SMALL ? 8 : (InpLabelSize == SIZE_LARGE ? 12 : 10));
   
   // Anchor:
   // TV style_label_up (Buy): Arrow points UP, Label is BELOW price.
   // My Code: Price = Low - Offset. This is correct.
   // Anchor: If I use ANCHOR_TOP, the top of the text is at 'Price'. Text extends downwards.
   // Correct for Buy.
   
   // TV style_label_down (Sell): Arrow points DOWN, Label is ABOVE price.
   // My Code: Price = High + Offset.
   // Anchor: If I use ANCHOR_BOTTOM, the bottom of the text is at 'Price'. Text extends upwards.
   // Correct for Sell.
   
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, up ? ANCHOR_TOP : ANCHOR_BOTTOM);
   
   LastLabelName = name;
  }

//+------------------------------------------------------------------+
//| Helper: Write Signal to JSON File (with duplicate prevention)    |
//+------------------------------------------------------------------+
// Static variables to prevent duplicate signals
static datetime g_lastSignalTime = 0;
static string   g_lastSignalAction = "";
static string   g_lastSignalSymbol = "";

void WriteSignalToFile(string action, double price, double sl, double tp, string comment)
  {
   // Get current bar time (not tick time) to prevent duplicates within same bar
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   
   // Check if this is a duplicate signal (same symbol, action, and bar)
   if(g_lastSignalTime == currentBarTime && 
      g_lastSignalAction == action && 
      g_lastSignalSymbol == _Symbol)
     {
      // Skip duplicate signal
      return;
     }
   
   // Update last signal info
   g_lastSignalTime = currentBarTime;
   g_lastSignalAction = action;
   g_lastSignalSymbol = _Symbol;
   
   // Write signal file
   string filename = "AlphaOS\\Signals\\signal_" + _Symbol + "_" + IntegerToString(TimeCurrent()) + ".json";
   int file_handle = FileOpen(filename, FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(file_handle != INVALID_HANDLE)
     {
      string json = StringFormat(
         "{\"symbol\":\"%s\",\"action\":\"%s\",\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"comment\":\"%s\"}",
         _Symbol, action, price, sl, tp, comment
      );
      FileWriteString(file_handle, json);
      FileClose(file_handle);
      
      Print("📤 Signal written: ", action, " ", _Symbol, " @ ", price);
     }
   else
     {
      Print("❌ Failed to write signal file: ", GetLastError());
     }
  }
