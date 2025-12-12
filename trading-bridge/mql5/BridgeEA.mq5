//+------------------------------------------------------------------+
//|                                                     BridgeEA.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                       HTTP Polling Version (MVP) |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.20"

input string ApiUrl = "http://100.91.208.22:8000"; 
input int PollInterval = 1000;
input int MagicNumber = 888888;
input int SlippagePoints = 20;
input int HistoryBars = 120; // For AI Inference

// Global Variables
datetime last_bar_time = 0;

int OnInit()
  {
   Print("BridgeEA (HTTP): Initializing...");
   Print("Target API URL: ", ApiUrl);
   
   if(MarketBookAdd(Symbol()))
      Print("Subscribed to MarketBook for ", Symbol());
   else
      Print("Failed to subscribe to MarketBook for ", Symbol());

   EventSetMillisecondTimer(PollInterval);
   
   // Initialize last_bar_time
   last_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   MarketBookRelease(Symbol());
   EventKillTimer();
  }

void OnTick()
  {
   // Detect New Bar (1m/5m etc based on chart)
   datetime current_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(last_bar_time != current_time)
     {
      if(last_bar_time != 0) 
        {
         Print("🕯️ New Bar Detected: ", TimeToString(current_time), " - Triggering AI Inference");
         SendInferenceRequest();
        }
      last_bar_time = current_time;
     }
  }

void OnTimer()
  {
   CheckForCommands();
   SendStatusUpdate();
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

void SendInferenceRequest() 
  {
   // 1. Get Candles
   MqlRates rates[];
   ArraySetAsSeries(rates, true); // Index 0 is newest (not closed), 1 is last closed
   
   // We want completed bars for history, plus maybe the open of current?
   // Typically models train on closed bars.
   // Let's send [1..HistoryBars]
   
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 1, HistoryBars, rates);
   
   if(copied < 60) {
       Print("⚠️ Not enough history for inference: ", copied);
       return;
   }
   
   string candles_json = "[";
   // Iterate backwards to send in chronological order? 
   // CopyRates with AsSeries=true means 0 is newest (time t-1). 
   // Python usually expects chronological [t-N, ..., t-1].
   // So if rates[0] is t-1, rates[copied-1] is t-N.
   // We should iterate from copied-1 down to 0.
   
   for(int i=copied-1; i>=0; i--) 
     {
      // Short keys: t=time, o=open, h=high, l=low, c=close, v=volume
      string item = StringFormat("{\"t\":%I64d,\"o\":%.5f,\"h\":%.5f,\"l\":%.5f,\"c\":%.5f,\"v\":%I64d}",
         rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume);
      
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
       // Response might contain immediate action (if we want synchronous execution support in future)
       // Currently AI Engine sends command back via queue, but direct response is faster.
       // TODO: If response contains action, execute immediately?
       // The Python bridge /inference endpoint will return the AI response.
       string res_str = CharArrayToString(res_data);
       // If res_str contains "action":"BUY", we could execute here.
       // But sticking to Command Queue architecture for safety/consistency for now.
   } else {
       Print("❌ Inference Request Failed: ", res, " Error: ", GetLastError());
   }
  }

void CheckForCommands()
  {
   char data[];
   char result[];
   string result_headers;
   
   // Standard GET request
   // FIXED: Removed extra argument '0'
   int res = WebRequest("GET", ApiUrl + "/commands/pop", "", 500, data, result, result_headers);
   
   if(res == 200)
     {
      string json = CharArrayToString(result);
      if(json != "null" && StringLen(json) > 5) 
        {
         Print("Received command: ", json);
         
         string type = ExtractJsonString(json, "type");
         
         if (type == "PENDING") {
             string side = ExtractJsonString(json, "action"); // "BUY" or "SELL"
             string symbol = ExtractJsonString(json, "symbol");
             double volume = ExtractJsonDouble(json, "volume");
             double price = ExtractJsonDouble(json, "price");
             double sl = ExtractJsonDouble(json, "sl");
             double tp = ExtractJsonDouble(json, "tp");
             
             if(volume <= 0) volume = 0.01;
             if(symbol == "") symbol = Symbol();
             
             ENUM_ORDER_TYPE order_type;
             double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
             double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
             
             if (side == "BUY") {
                 if (price > current_ask) order_type = ORDER_TYPE_BUY_STOP;
                 else order_type = ORDER_TYPE_BUY_LIMIT;
             } else { // SELL
                 if (price < current_bid) order_type = ORDER_TYPE_SELL_STOP;
                 else order_type = ORDER_TYPE_SELL_LIMIT;
             }
             
             ExecutePendingOrder(symbol, order_type, volume, price, sl, tp);
         }
         else if (type == "TRADE" || type == "") { // Default to TRADE if type missing (legacy)
             string action = ExtractJsonString(json, "action");
             string symbol = ExtractJsonString(json, "symbol");
             double volume = ExtractJsonDouble(json, "volume");
             double sl = ExtractJsonDouble(json, "sl");
             double tp = ExtractJsonDouble(json, "tp");
             long ticket = (long)ExtractJsonDouble(json, "ticket");
             
             if(action == "CLOSE") {
                 if(ticket > 0) {
                     Print("Executing CLOSE command for ticket: ", ticket);
                     ClosePosition(ticket);
                 }
                 else Print("CLOSE command missing ticket or invalid ticket: ", ticket);
             }
             else if(action == "PENDING") {
                 // Handled by top-level PENDING check usually, but legacy might come here
                 Print("Received PENDING inside TRADE block - unexpected");
             }
             else {
                 if(volume <= 0) volume = 0.01;
                 if(symbol == "") symbol = Symbol(); 
                 
                 if(action == "BUY") ExecuteTrade(symbol, ORDER_TYPE_BUY, volume, sl, tp);
                 else if(action == "SELL") ExecuteTrade(symbol, ORDER_TYPE_SELL, volume, sl, tp);
             }
         }
         else if (type == "GET_HISTORY") {
             string req_id = ExtractJsonString(json, "request_id");
             string symbol = ExtractJsonString(json, "symbol");
             string tf_str = ExtractJsonString(json, "timeframe");
             int count = (int)ExtractJsonDouble(json, "count");
             long from_t = (long)ExtractJsonDouble(json, "from");
             long to_t = (long)ExtractJsonDouble(json, "to");
             
             if(count > 1000) count = 1000; // Limit to 1000 per request
             if(count <= 0) count = 100;
             
             ENUM_TIMEFRAMES tf = StringToTimeframe(tf_str);
             
             SendHistoryData(req_id, symbol, tf, count, from_t, to_t);
         }
         else if (type == "GET_DOM") {
             string req_id = ExtractJsonString(json, "request_id");
             string symbol = ExtractJsonString(json, "symbol");
             if(symbol == "") symbol = Symbol();
             
             SendDOMData(req_id, symbol);
         }
        }
     }
  }

//+------------------------------------------------------------------+
//| Helper: Extract JSON String                                      |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
  {
   string pattern = "\"" + key + "\":\"";
   int start = StringFind(json, pattern);
   if(start < 0) return "";
   start += StringLen(pattern);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
  }

//+------------------------------------------------------------------+
//| Helper: Extract JSON Number                                      |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
  {
   string pattern = "\"" + key + "\":";
   int start = StringFind(json, pattern);
   if(start < 0) return 0.0;
   start += StringLen(pattern);
   int end = StringFind(json, ",", start);
   if(end < 0) end = StringFind(json, "}", start);
   if(end < 0) return 0.0;
   
   string valStr = StringSubstr(json, start, end - start);
   return StringToDouble(valStr);
  }

//+------------------------------------------------------------------+
//| String to Timeframe                                              |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf) {
   if(tf == "M1" || tf == "PERIOD_M1") return PERIOD_M1;
   if(tf == "M5" || tf == "PERIOD_M5") return PERIOD_M5;
   if(tf == "M15" || tf == "PERIOD_M15") return PERIOD_M15;
   if(tf == "M30" || tf == "PERIOD_M30") return PERIOD_M30;
   if(tf == "H1" || tf == "PERIOD_H1") return PERIOD_H1;
   if(tf == "H4" || tf == "PERIOD_H4") return PERIOD_H4;
   if(tf == "D1" || tf == "PERIOD_D1") return PERIOD_D1;
   if(tf == "W1" || tf == "PERIOD_W1") return PERIOD_W1;
   if(tf == "MN1" || tf == "PERIOD_MN1") return PERIOD_MN1;
   return PERIOD_CURRENT;
}

//+------------------------------------------------------------------+
//| Get History Data and Send                                        |
//+------------------------------------------------------------------+
void SendHistoryData(string req_id, string symbol, ENUM_TIMEFRAMES tf, int count, long from_t, long to_t) {
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    int copied = 0;
    
    if(from_t > 0 && to_t > 0) {
        copied = CopyRates(symbol, tf, (datetime)from_t, (datetime)to_t, rates);
    } else {
        copied = CopyRates(symbol, tf, 0, count, rates);
    }
    
    if(copied > 0) {
        // Build JSON
        string json = StringFormat("{\"request_id\":\"%s\",\"symbol\":\"%s\",\"timeframe\":\"%s\",\"count\":%d,\"data\":[", 
                                   req_id, symbol, EnumToString(tf), copied);
        
        for(int i=0; i<copied; i++) {
            if(i > 0) json += ",";
            // Use simple format to save space
            json += StringFormat("{\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_volume\":%I64d}",
                                 rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume);
        }
        json += "]}";
        
        // Send
        char data[];
        StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
        char res_data[];
        string headers = "Content-Type: application/json\r\n";
        string result_headers;
        
        // Increase timeout to 2000ms for larger data
        WebRequest("POST", ApiUrl + "/data/history", headers, 2000, data, res_data, result_headers);
        Print("Sent history data: ", copied, " candles for ", symbol);
    } else {
        Print("Failed to copy rates for ", symbol);
    }
}

//+------------------------------------------------------------------+
//| Get DOM Data and Send                                            |
//+------------------------------------------------------------------+
void SendDOMData(string req_id, string symbol) {
    // Implementation handled by GetDOMJson in Inference path roughly, but this is legacy command
    // Can reuse if needed
    string dom_str = GetDOMJson(symbol);
    string json = StringFormat("{\"request_id\":\"%s\",\"symbol\":\"%s\",\"count\":0,%s}", req_id, symbol, StringSubstr(dom_str, 1, StringLen(dom_str)-2)); 
    // A bit hacky string manipulation but acceptable for legacy
    
        char data[];
        StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
        char res_data[];
        string headers = "Content-Type: application/json\r\n";
        string result_headers;
        
        WebRequest("POST", ApiUrl + "/data/dom", headers, 500, data, res_data, result_headers);
}

//+------------------------------------------------------------------+
//| Get Filling Mode (Fix 4756 Error)                                |
//+------------------------------------------------------------------+
uint GetFillingMode(string symbol)
  {
   long mode = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((mode & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   if((mode & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN;
  }

//+------------------------------------------------------------------+
//| Get DOM Summary (Imbalance, Volumes)                             |
//+------------------------------------------------------------------+
string GetDOMSummary(string symbol)
  {
   MqlBookInfo book[];
   
   // Ensure subscription
   MarketBookAdd(symbol);
   
   double imbalance = 0.0;
   double best_bid_vol = 0.0;
   double best_ask_vol = 0.0;
   
   if(MarketBookGet(symbol, book))
     {
      int size = ArraySize(book);
      double total_bid_vol = 0;
      double total_ask_vol = 0;
      
      for(int i=0; i<size; i++)
        {
         if(book[i].type == BOOK_TYPE_BUY)
           {
            total_bid_vol += (double)book[i].volume;
            if (book[i].price > 0) { /* Check for best bid logic here if needed */ }
           }
         else if(book[i].type == BOOK_TYPE_SELL)
           {
            total_ask_vol += (double)book[i].volume;
           }
        }
        
       double total_vol = total_bid_vol + total_ask_vol;
       if(total_vol > 0)
         imbalance = (total_bid_vol - total_ask_vol) / total_vol;
         
       if(size > 0) {
           // Just grab first items for simplicity (MQL Book is sorted)
           // Bids: High to Low. Asks: Low to High. 
           // But MarketBookGet output order depends.
           // Usually needs loop.
       }
     }
     
   return StringFormat("\"dom\":{\"imbalance\":%.4f,\"best_bid_vol\":%.2f,\"best_ask_vol\":%.2f}", 
                       imbalance, best_bid_vol, best_ask_vol);
  }

void SendStatusUpdate()
  {
   // 1. Account Info
   string account_json = StringFormat(
      "\"account\":{\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f}",
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE)
   );

   // 2. Positions
   string positions_json = "\"positions\":[";
   int total = PositionsTotal();
   int added_count = 0;
   // Print("Total Positions: ", total); // Debug
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
        {
         if(added_count > 0) positions_json += ",";
         
         string symbol = PositionGetString(POSITION_SYMBOL);
         string comment = PositionGetString(POSITION_COMMENT);
         
         // Sanitize comment to avoid JSON breakage
         StringReplace(comment, "\"", "'");
         StringReplace(comment, "\\", "/");
         
         positions_json += StringFormat(
            "{\"ticket\":%I64u,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"open_price\":%.5f,\"current_price\":%.5f,\"pnl\":%.2f,\"swap\":%.2f,\"sl\":%.5f,\"tp\":%.5f,\"comment\":\"%s\"}",
            ticket,
            symbol,
            (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "BUY" : "SELL"),
            PositionGetDouble(POSITION_VOLUME),
            PositionGetDouble(POSITION_PRICE_OPEN),
            PositionGetDouble(POSITION_PRICE_CURRENT),
            PositionGetDouble(POSITION_PROFIT),
            PositionGetDouble(POSITION_SWAP),
            PositionGetDouble(POSITION_SL),
            PositionGetDouble(POSITION_TP),
            comment
         );
         added_count++;
        }
     }
   positions_json += "]";

   // 3. Market Watch Quotes
   string quotes_json = "\"quotes\":[";
   int total_symbols = SymbolsTotal(true); // true = Market Watch only
   int added_quotes = 0;
   
   for(int i=0; i<total_symbols; i++) {
       string symbol = SymbolName(i, true);
       double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
       double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
       
       if(bid > 0 && ask > 0) {
           if(added_quotes > 0) quotes_json += ",";
           quotes_json += StringFormat("{\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f}", symbol, bid, ask);
           added_quotes++;
       }
   }
   quotes_json += "]";

   // 4. Current Chart Quote
   double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   
   string legacy_json = StringFormat("\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"period\":\"%s\"", 
                              Symbol(), bid, ask, EnumToString(Period()));

   // 5. DOM Summary
   string dom_json = GetDOMSummary(Symbol());

   // Combine Final JSON
   string json = "{" + account_json + "," + positions_json + "," + quotes_json + "," + legacy_json + "," + dom_json + "}";
   
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   string result_headers;
   
   WebRequest("POST", ApiUrl + "/status/update", headers, 500, data, result, result_headers);
  }

//+------------------------------------------------------------------+
//| Trade Transaction Listener                                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& _request,
                        const MqlTradeResult& _result)
  {
   // Only care about "Deal Add" (Execution)
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      ulong ticket = trans.deal;
      // Select deal to get details
      if(HistoryDealSelect(ticket))
        {
         string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
         long type = HistoryDealGetInteger(ticket, DEAL_TYPE); // 0=BUY, 1=SELL
         double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
         double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
         long time = HistoryDealGetInteger(ticket, DEAL_TIME);
         long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY); // 0=IN, 1=OUT, 2=INOUT
         long position_id = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
         
         if(type > DEAL_TYPE_SELL) return;

         string side = (type == DEAL_TYPE_BUY) ? "BUY" : "SELL";
         string entry_str = (entry == DEAL_ENTRY_IN) ? "IN" : (entry == DEAL_ENTRY_OUT) ? "OUT" : "INOUT";
         
         string json = StringFormat("{\"ticket\":%I64u,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"time\":\"%I64d\",\"entry\":\"%s\",\"position_id\":%I64d,\"profit\":%.2f,\"commission\":%.2f,\"swap\":%.2f}",
                                    ticket, symbol, side, volume, price, time, entry_str, position_id, profit, commission, swap);
         
         Print("Reporting Trade: ", json);
         
         char data[];
         StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
         char res_data[];
         string headers = "Content-Type: application/json\r\n";
         string result_headers;
         
         WebRequest("POST", ApiUrl + "/trade/report", headers, 1000, data, res_data, result_headers);
        }
     }
  }

void ExecuteTrade(string symbol, ENUM_ORDER_TYPE type, double volume, double sl, double tp)
  {
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = volume;
   request.type   = type;
   request.price  = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.deviation = SlippagePoints; 
   request.magic = MagicNumber;
   request.sl = sl;
   request.tp = tp;
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol); // Auto-adapt filling mode
   request.comment = "AI-AlphaOS";
   
   if(OrderSend(request, result)) 
     {
      Print("✅ Trade Executed! Ticket: ", result.deal);
     }
   else 
     {
      Print("❌ Trade Failed: Code=", result.retcode, " (", GetRetcodeDescription(result.retcode), ") Comment: ", result.comment);
     }
  }

string GetRetcodeDescription(int retcode)
{
   switch(retcode)
   {
      case 10004: return "REQUOTE";
      case 10006: return "REJECT";
      case 10013: return "INVALID_REQUEST";
      case 10014: return "INVALID_VOLUME";
      case 10015: return "INVALID_PRICE";
      case 10016: return "INVALID_STOPS";
      case 10017: return "TRADE_DISABLED";
      case 10018: return "MARKET_CLOSED";
      case 10019: return "NO_MONEY";
      case 10027: return "AUTO_TRADING_DISABLED";
      default: return "UNKNOWN_ERROR";
   }
}

void ClosePosition(long ticket)
  {
   if(!PositionSelectByTicket(ticket)) {
       Print("ClosePosition: Ticket not found ", ticket);
       return;
   }
   
   string symbol = PositionGetString(POSITION_SYMBOL);
   long type = PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = volume;
   request.deviation = SlippagePoints;
   request.magic = MagicNumber;
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol);
   
   // Close by opening opposite order
   if(type == POSITION_TYPE_BUY) {
       request.type = ORDER_TYPE_SELL;
       request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
   } else {
       request.type = ORDER_TYPE_BUY;
       request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }
   
   if(OrderSend(request, result)) {
       Print("✅ Position Closed! Ticket: ", ticket);
   } else {
       Print("❌ Close Failed: ", result.retcode, " (", GetRetcodeDescription(result.retcode), ")");
   }
  }

void ExecutePendingOrder(string symbol, ENUM_ORDER_TYPE type, double volume, double price, double sl, double tp)
  {
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_PENDING;
   request.symbol = symbol;
   request.volume = volume;
   request.type   = type;
   request.price  = price;
   request.sl = sl;
   request.tp = tp;
   request.magic = MagicNumber;
   request.type_time = ORDER_TIME_GTC; // Good Till Cancelled
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol);
   request.comment = "AI-Pending";
   
   if(OrderSend(request, result)) 
     {
      Print("✅ Pending Order Placed! Ticket: ", result.order);
     }
   else 
     {
      Print("❌ Pending Order Failed: ", result.retcode, " (", GetRetcodeDescription(result.retcode), ") Comment: ", result.comment);
   }
  }
