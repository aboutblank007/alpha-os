//+------------------------------------------------------------------+
//|                             AlphaOS_DataCollector_HighFreq.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                  High-Frequency Data Collector (1-5m Optimized)  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.00"
#property description "Collects OHLCV + Microstructure features (DOM, Aggressor Flow) for AI training"

input string FileName = "training_data_highfreq.csv"; // Output filename
input bool   CollectDOM = true;     // Whether to subscribe to Level 2 data
input int    FlushInterval = 10;    // Flush to disk every N bars

//--- Global Variables
int file_handle = INVALID_HANDLE;
MqlTick last_tick;
datetime last_bar_time = 0;
long flush_counter = 0;

// Accumulators for current bar
long acc_tick_count = 0;
double acc_buy_vol = 0;   // Aggressor Buy Volume
double acc_sell_vol = 0;  // Aggressor Sell Volume
double acc_spread_sum = 0;
double acc_dom_imbalance_sum = 0; // Sum of DOM imbalances
long acc_dom_ticks = 0;   // Number of DOM updates

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Open file for writing (append mode if exists, else create)
   // File is located in MQL5/Files/
   string path = FileName;
   
   // Check if file exists to write header
   bool file_exists = FileIsExist(path);
   
   file_handle = FileOpen(path, FILE_CSV|FILE_WRITE|FILE_ANSI, ",");
   if(file_handle == INVALID_HANDLE)
     {
      Print("❌ Failed to open file: ", path, " Error: ", GetLastError());
      return(INIT_FAILED);
     }
     
   // Move to end of file to append
   if(file_exists)
      FileSeek(file_handle, 0, SEEK_END);
   else
     {
      // Write Header
      string header = "timestamp,symbol,open,high,low,close,tick_volume,real_volume,spread,tick_count,aggressor_buy_vol,aggressor_sell_vol,avg_dom_imbalance,volatility_skew_proxy";
      FileWrite(file_handle, header);
     }

   // Subscribe to Market Book (DOM)
   if(CollectDOM)
     {
      if(MarketBookAdd(Symbol()))
         Print("✅ Subscribed to MarketBook for ", Symbol());
      else
         Print("⚠️ Failed to subscribe to MarketBook. DOM features will be 0.");
     }

   // Initialize last_tick
   if(!SymbolInfoTick(Symbol(), last_tick))
      Print("⚠️ Failed to get initial tick");

   last_bar_time = iTime(Symbol(), PERIOD_CURRENT, 0);

   Print("🚀 High-Frequency Data Collector Started on ", Symbol());
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(CollectDOM) MarketBookRelease(Symbol());
   
   if(file_handle != INVALID_HANDLE)
     {
      FileClose(file_handle);
      Print("💾 File closed.");
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   MqlTick current_tick;
   if(!SymbolInfoTick(Symbol(), current_tick)) return;

   // 1. Detect New Bar
   datetime current_time = iTime(Symbol(), PERIOD_CURRENT, 0);
   if(last_bar_time != current_time)
     {
      if(last_bar_time != 0)
        {
         WriteBarData(last_bar_time);
        }
      
      // Reset accumulators
      acc_tick_count = 0;
      acc_buy_vol = 0;
      acc_sell_vol = 0;
      acc_spread_sum = 0;
      acc_dom_imbalance_sum = 0;
      acc_dom_ticks = 0;
      
      last_bar_time = current_time;
     }

   // 2. Accumulate Tick Data
   acc_tick_count++;
   
   // Spread
   double spread = (current_tick.ask - current_tick.bid);
   acc_spread_sum += spread;
   
   // Aggressor Flow Logic
   // Try to use flags if available
   bool is_buy = ((current_tick.flags & TICK_FLAG_BUY) == TICK_FLAG_BUY);
   bool is_sell = ((current_tick.flags & TICK_FLAG_SELL) == TICK_FLAG_SELL);
   
   // Fallback to price change if flags missing
   if (!is_buy && !is_sell) {
       if (current_tick.last > last_tick.last) is_buy = true;
       else if (current_tick.last < last_tick.last) is_sell = true;
   }
   
   double vol = (current_tick.volume_real > 0) ? current_tick.volume_real : (double)current_tick.volume;
   if(vol <= 0) vol = 1.0; // Fallback
   
   if(is_buy) acc_buy_vol += vol;
   if(is_sell) acc_sell_vol += vol;
   
   last_tick = current_tick;
  }

//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string& symbol)
  {
   if(symbol != Symbol() || !CollectDOM) return;
   
   MqlBookInfo book[];
   if(MarketBookGet(symbol, book))
     {
      long total_bid_vol = 0;
      long total_ask_vol = 0;
      int size = ArraySize(book);
      
      for(int i=0; i<size; i++)
        {
         if(book[i].type == BOOK_TYPE_BUY || book[i].type == BOOK_TYPE_BUY_MARKET)
            total_bid_vol += book[i].volume;
         else if(book[i].type == BOOK_TYPE_SELL || book[i].type == BOOK_TYPE_SELL_MARKET)
            total_ask_vol += book[i].volume;
        }
        
      // Calculate Imbalance: (Bid - Ask) / (Bid + Ask)
      // Range: -1 (All Ask) to +1 (All Bid)
      double imbalance = 0;
      if(total_bid_vol + total_ask_vol > 0)
         imbalance = (double)(total_bid_vol - total_ask_vol) / (double)(total_bid_vol + total_ask_vol);
         
      acc_dom_imbalance_sum += imbalance;
      acc_dom_ticks++;
     }
  }

//+------------------------------------------------------------------+
//| Helper: Write Bar to CSV                                         |
//+------------------------------------------------------------------+
void WriteBarData(datetime time)
  {
   // Get Bar OHLCV (Shift 1 because 0 is the new forming bar, and we want the just closed one)
   // Wait, OnTick detects new bar time, so shift 1 is the bar that just finished.
   
   double open[], high[], low[], close[];
   long tick_vol[], real_vol[];
   
   // CopyRates is easier
   MqlRates rates[];
   if(CopyRates(Symbol(), PERIOD_CURRENT, 1, 1, rates) < 1) return;
   
   MqlRates bar = rates[0]; // The closed bar
   
   // Averages
   double avg_spread = (acc_tick_count > 0) ? acc_spread_sum / acc_tick_count : 0;
   double avg_dom_imb = (acc_dom_ticks > 0) ? acc_dom_imbalance_sum / acc_dom_ticks : 0;
   
   // Simple Volatility Skew Proxy: (High-Close) vs (Close-Low)
   // Measures if selling pressure (upper wick) was stronger than buying pressure (lower wick)
   double upper_wick = bar.high - MathMax(bar.open, bar.close);
   double lower_wick = MathMin(bar.open, bar.close) - bar.low;
   double vol_skew = (upper_wick - lower_wick) / (bar.high - bar.low + Point()); 
   
   // Format: timestamp,symbol,open,high,low,close,tick_volume,real_volume,spread,tick_count,aggressor_buy_vol,aggressor_sell_vol,avg_dom_imbalance,volatility_skew_proxy
   string line = StringFormat("%I64d,%s,%.5f,%.5f,%.5f,%.5f,%I64d,%I64d,%.5f,%I64d,%.2f,%.2f,%.4f,%.4f",
      bar.time,
      Symbol(),
      bar.open,
      bar.high,
      bar.low,
      bar.close,
      (long)bar.tick_volume,
      (long)bar.real_volume,
      avg_spread,
      acc_tick_count,
      acc_buy_vol,
      acc_sell_vol,
      avg_dom_imb,
      vol_skew
   );
   
   FileWrite(file_handle, line);
   FileFlush(file_handle); // Ensure data is written
   
   flush_counter++;
   if(flush_counter % 10 == 0) Print("📊 Collected ", flush_counter, " bars of high-freq data.");
  }

