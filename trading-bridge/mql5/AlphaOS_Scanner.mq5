//+------------------------------------------------------------------+
//|                                             AlphaOS_Scanner.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                     Data Push & AI Trigger Only  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.00"

input string ApiUrl = "http://192.168.3.8:8000"; 
input int HistoryBars = 120; // For AI Inference
input int StatusInterval = 1000; // Send status every 1s

// Global Variables
datetime last_bar_time = 0;
MqlTick last_tick;

// Accumulators for current bar (Microstructure)
long acc_tick_count = 0;
double acc_buy_vol = 0;   // Aggressor Buy Volume
double acc_sell_vol = 0;  // Aggressor Sell Volume

int OnInit()
  {
   Print("AlphaOS Scanner: Initializing on ", _Symbol);
   Print("Target API URL: ", ApiUrl);
   
   if(MarketBookAdd(Symbol()))
      Print("Subscribed to MarketBook for ", Symbol());
   else
      Print("Failed to subscribe to MarketBook for ", Symbol());

   // Initialize last_bar_time
   last_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   // Init tick for delta calc
   if(!SymbolInfoTick(Symbol(), last_tick))
      Print("⚠️ Failed to get initial tick");
   
   EventSetMillisecondTimer(StatusInterval);
   
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   MarketBookRelease(Symbol());
   EventKillTimer();
  }

void OnTimer()
  {
   SendStatusUpdate();
  }

void OnTick()
  {
   MqlTick current_tick;
   if(!SymbolInfoTick(Symbol(), current_tick)) return;

   // 1. Accumulate Aggressor Flow
   // (Same logic as DataCollector)
   acc_tick_count++;
   
   bool is_buy = ((current_tick.flags & TICK_FLAG_BUY) == TICK_FLAG_BUY);
   bool is_sell = ((current_tick.flags & TICK_FLAG_SELL) == TICK_FLAG_SELL);
   
   if (!is_buy && !is_sell) {
       if (current_tick.last > last_tick.last) is_buy = true;
       else if (current_tick.last < last_tick.last) is_sell = true;
   }
   
   double vol = (current_tick.volume_real > 0) ? current_tick.volume_real : (double)current_tick.volume;
   if(vol <= 0) vol = 1.0;
   
   if(is_buy) acc_buy_vol += vol;
   if(is_sell) acc_sell_vol += vol;
   
   last_tick = current_tick;

   // 2. Detect New Bar
   datetime current_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(last_bar_time != current_time)
     {
      if(last_bar_time != 0) 
        {
         Print("🕯️ New Bar Detected: ", TimeToString(current_time), " - Triggering AI Inference");
         // Pass the accumulated values for the JUST CLOSED bar
         SendInferenceRequest(acc_tick_count, acc_buy_vol, acc_sell_vol);
        }
      
      // Reset accumulators for the new bar
      acc_tick_count = 0;
      acc_buy_vol = 0;
      acc_sell_vol = 0;
      
      last_bar_time = current_time;
     }
  }

//+------------------------------------------------------------------+
//| Inference Request Logic                                          |
//+------------------------------------------------------------------+
string GetDOMJson(string symbol) 
  {
   MqlBookInfo book[];
   MarketBookAdd(symbol);
   if(!MarketBookGet(symbol, book)) return "{\"bids\":[],\"asks\":[]}";
   
   string bids="", asks="";
   int size = ArraySize(book);
   
   for(int i=0; i<size; i++) 
     {
      // Short keys to save bandwidth: p=price, v=volume
      string item = StringFormat("{\"p\":%.5f,\"v\":%I64d}", book[i].price, book[i].volume);
      
      if(book[i].type == BOOK_TYPE_BUY) 
        {
         if(StringLen(bids)>0) bids+=",";
         bids += item;
        } 
      else 
        {
         if(StringLen(asks)>0) asks+=",";
         asks += item;
        }
     }
   return StringFormat("{\"bids\":[%s],\"asks\":[%s]}", bids, asks);
  }

void SendInferenceRequest(long tick_count, double buy_vol, double sell_vol) 
  {
   // 1. Get Candles
   MqlRates rates[];
   ArraySetAsSeries(rates, true); // Index 0 is newest (not closed), 1 is last closed
   
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 1, HistoryBars, rates);
   
   if(copied < 60) {
       Print("⚠️ Not enough history for inference: ", copied);
       return;
   }
   
   string candles_json = "[";
   
   // Send in chronological order [t-N ... t-1]
   // We attach the passed microstructure data to the LAST closed bar (which corresponds to rates[0] in Series=true if we shift 1?)
   // Wait, CopyRates(..., 1, ...) means we start from index 1 (closed bar).
   // So rates[0] in the array is the most recent CLOSED bar.
   // We attach the accumulators to THIS bar.
   
   for(int i=copied-1; i>=0; i--) 
     {
      // Default microstruct to 0 for older bars
      double rv = (double)rates[i].real_volume; 
      long tc = rates[i].tick_volume; // Fallback
      double ab = 0;
      double as = 0;
      
      // If this is the most recent closed bar (i=0), inject our accumulated real data
      if(i == 0) {
          tc = tick_count;
          ab = buy_vol;
          as = sell_vol;
          // rv is already rates[i].real_volume or tick_volume
      }
      
      string item = StringFormat("{\"t\":%I64d,\"o\":%.5f,\"h\":%.5f,\"l\":%.5f,\"c\":%.5f,\"v\":%I64d,\"rv\":%.2f,\"tc\":%I64d,\"ab\":%.2f,\"as\":%.2f}",
         rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume,
         rv, tc, ab, as);
      
      if(candles_json != "[") candles_json += ",";
      candles_json += item;
     }
   candles_json += "]";
   
   // 2. DOM
   string dom_json = GetDOMJson(_Symbol);
   
   // 3. Current Market State
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // 4. Construct Payload
   string json = StringFormat(
       "{\"type\":\"INFERENCE\",\"symbol\":\"%s\",\"timeframe\":\"%s\",\"action\":\"SCAN\",\"candles\":%s,\"dom\":%s,\"ask\":%.5f,\"bid\":%.5f}",
       _Symbol, EnumToString(Period()), candles_json, dom_json, ask, bid
   );
   
   // 5. Send
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char res_data[];
   string headers = "Content-Type: application/json\r\n";
   string res_headers;
   
   // Use 1000ms timeout - needs to be fast
   int res = WebRequest("POST", ApiUrl + "/inference", headers, 1000, data, res_data, res_headers);
   
   if(res == 200) {
       Print("✅ Inference Request Sent for ", _Symbol);
   } else {
       Print("❌ Inference Request Failed: ", res, " Error: ", GetLastError());
   }
  }

//+------------------------------------------------------------------+
//| Status Reporting (Scanner Version - Minimal)                     |
//+------------------------------------------------------------------+
void SendStatusUpdate()
  {
   // Scanner only reports the symbol it is watching, to update Active Symbols in Bridge
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   string json = StringFormat(
      "{\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"period\":\"%s\"}",
      _Symbol, bid, ask, EnumToString(Period())
   );
   
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   string result_headers;
   
   // Use shorter timeout for status
   WebRequest("POST", ApiUrl + "/status/update", headers, 200, data, result, result_headers);
  }
