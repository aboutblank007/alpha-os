//+------------------------------------------------------------------+
//|                                                 DataExporter.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                       Fast Data Generation EA    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.00"

// Include standard libraries if needed, but we'll keep it self-contained
// Reusing Logic from PivotTrendSignals.mq5

input group "=== Indicator Settings ==="
input int      InpPivotPeriod = 2;
input int      InpATRPeriod   = 14;
input int      InpEMALength1  = 6;
input int      InpEMALength2  = 24;
input int      InpRSIPeriod   = 14;

input group "=== Export Settings ==="
input string   InpFileName    = "training_data_mt5.csv";

// Handles
int hEMA1, hEMA2, hATR, hADX, hRSI;

// File Handle
int file_handle = INVALID_HANDLE;

// Globals
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Create Indicators
   hEMA1 = iMA(_Symbol, _Period, InpEMALength1, 0, MODE_EMA, PRICE_CLOSE);
   hEMA2 = iMA(_Symbol, _Period, InpEMALength2, 0, MODE_EMA, PRICE_CLOSE);
   hATR  = iATR(_Symbol, _Period, InpATRPeriod);
   hADX  = iADX(_Symbol, _Period, 14);
   hRSI  = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);

   if(hEMA1 == INVALID_HANDLE || hEMA2 == INVALID_HANDLE || hATR == INVALID_HANDLE || hRSI == INVALID_HANDLE)
     {
      Print("Failed to create indicator handles");
      return(INIT_FAILED);
     }

   // Open File
   // FILE_COMMON creates it in the shared Terminal/Common/Files folder
   // Regular FILE_WRITE creates it in MQL5/Files of the specific instance
   // We use MQL5/Files so it's instance specific
   file_handle = FileOpen(InpFileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   
   if(file_handle == INVALID_HANDLE)
     {
      Print("Failed to open file: ", InpFileName, " Error: ", GetLastError());
      return(INIT_FAILED);
     }

   // Write Header (Must match Python expectations)
   // time, open, high, low, close, tick_volume, ema_short, ema_long, atr, adx, rsi, center, price_vs_center, cloud_width, ema_spread, atr_percent, candle_size, wick_upper, wick_lower
   FileWrite(file_handle, 
      "time", "open", "high", "low", "close", "tick_volume", 
      "ema_short", "ema_long", "atr", "adx", "rsi", 
      "center", "price_vs_center", "cloud_width", "ema_spread", "atr_percent", 
      "candle_size", "wick_upper", "wick_lower"
   );

   Print("DataExporter Initialized. Writing to ", InpFileName);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(file_handle != INVALID_HANDLE) FileClose(file_handle);
   IndicatorRelease(hEMA1);
   IndicatorRelease(hEMA2);
   IndicatorRelease(hATR);
   IndicatorRelease(hADX);
   IndicatorRelease(hRSI);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime current_time = iTime(_Symbol, _Period, 0);
   
   // Detect New Bar
   if(current_time != last_bar_time)
     {
      // If this is not the very first run
      if(last_bar_time != 0)
        {
         // Process the COMPLETED bar (index 1)
         ProcessBar(1);
        }
      last_bar_time = current_time;
     }
  }

//+------------------------------------------------------------------+
//| Process Bar and Write to CSV                                     |
//+------------------------------------------------------------------+
void ProcessBar(int i)
  {
   // Get Price Data
   double open = iOpen(_Symbol, _Period, i);
   double high = iHigh(_Symbol, _Period, i);
   double low = iLow(_Symbol, _Period, i);
   double close = iClose(_Symbol, _Period, i);
   long tick_vol = iTickVolume(_Symbol, _Period, i);
   datetime time = iTime(_Symbol, _Period, i);

   // Get Indicator Data
   double ema1[], ema2[], atr[], adx[], rsi[];
   
   // Use CopyBuffer to get specific index
   // Note: CopyBuffer gets data in array, careful with indexing
   // CopyBuffer(handle, buffer_num, start_index, count, buffer)
   
   double b_ema1[1], b_ema2[1], b_atr[1], b_adx[1], b_rsi[1];
   
   if(CopyBuffer(hEMA1, 0, i, 1, b_ema1) <= 0) return;
   if(CopyBuffer(hEMA2, 0, i, 1, b_ema2) <= 0) return;
   if(CopyBuffer(hATR, 0, i, 1, b_atr) <= 0) return;
   if(CopyBuffer(hADX, 0, i, 1, b_adx) <= 0) return;
   if(CopyBuffer(hRSI, 0, i, 1, b_rsi) <= 0) return;

   double v_ema1 = b_ema1[0];
   double v_ema2 = b_ema2[0];
   double v_atr  = b_atr[0];
   double v_adx  = b_adx[0];
   double v_rsi  = b_rsi[0];

   // Derived Features
   double center = (v_ema1 + v_ema2) / 2.0;
   // FIX: Export RAW difference to match DataCollector and Client logic
   double price_vs_center = (close - center); 
   double cloud_width = MathAbs(v_ema1 - v_ema2);
   double ema_spread = cloud_width; // Alias
   double atr_percent = (close != 0) ? (v_atr / close * 100.0) : 0;
   
   double candle_size = MathAbs(high - low);
   double wick_upper = high - MathMax(open, close);
   double wick_lower = MathMin(open, close) - low;

   // Write to CSV
   // Time format: Unix Timestamp (Integer)
   FileWrite(file_handle, 
      (long)time, open, high, low, close, tick_vol,
      v_ema1, v_ema2, v_atr, v_adx, v_rsi,
      center, price_vs_center, cloud_width, ema_spread, atr_percent,
      candle_size, wick_upper, wick_lower
   );
  }

