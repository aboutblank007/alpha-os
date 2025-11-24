//+------------------------------------------------------------------+
//|                                                     BridgeEA.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                       HTTP Polling Version (MVP) |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.10"

input string ApiUrl = "http://api.lootool.cn:8000"; 
input int PollInterval = 1000;

int OnInit()
  {
   Print("BridgeEA (HTTP): Initializing...");
   Print("Target API URL: ", ApiUrl);
   
   EventSetMillisecondTimer(PollInterval);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTick()
  {
   // Standard EA requirement
  }

void OnTimer()
  {
   CheckForCommands();
   SendStatusUpdate();
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
                 double price = ExtractJsonDouble(json, "price");
                 string type_str = ExtractJsonString(json, "type_str"); // e.g. "BUY_LIMIT"
                 
                 // Determine pending type based on price and current market price if not explicitly provided
                 ENUM_ORDER_TYPE pending_type = ORDER_TYPE_BUY_LIMIT; // Default
                 double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
                 double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
                 
                 // Logic to auto-determine pending type if generic BUY/SELL
                 // If we receive "BUY" with a price:
                 // - Price < Ask => Buy Limit
                 // - Price > Ask => Buy Stop
                 // If we receive "SELL" with a price:
                 // - Price > Bid => Sell Limit
                 // - Price < Bid => Sell Stop
                 
                 // Use the 'side' from JSON (which is passed as 'action' in some contexts, but here we need to be careful)
                 // The current protocol sends action="BUY"/"SELL" even for pending if we reuse the logic,
                 // but the new frontend sends action="BUY"/"SELL" AND type="PENDING"
                 
                 // Let's check the `action` variable which holds "BUY" or "SELL"
                 string side = action; // This variable is from line 57, but we need to re-read it if we are in PENDING block?
                 // Actually, line 57 reads 'action'. In the new frontend logic:
                 // payload.action = side ("BUY" or "SELL")
                 // payload.type = "PENDING" (this overrides the top level type check? No)
                 
                 // Wait, the frontend sends:
                 // { action: "BUY", type: "PENDING", ... }
                 // But the EA parses 'type' at line 54.
                 // If type is "PENDING", we enter a new block?
                 // No, the EA currently handles "TRADE" or "".
                 
                 // We need to modify the EA to handle the new "PENDING" type or modify the logic inside TRADE.
                 // Let's modify the logic inside the existing block since `type` is usually "TRADE" or empty.
                 // But the frontend sends type="PENDING".
                 // So we need to add a check for type == "PENDING" at the top level OR handle it here.
                 
                 // Correct approach: Add a new top-level type check for "PENDING" or "ORDER"
                 // OR, just handle generic "TRADE" and check if "price" is present?
                 
                 // Let's stick to the Plan: The frontend sends type="PENDING".
                 // So we need to add `else if (type == "PENDING")` block.
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
             
             if(count > 1000) count = 1000; // Limit to 1000 per request
             if(count <= 0) count = 100;
             
             ENUM_TIMEFRAMES tf = StringToTimeframe(tf_str);
             
             SendHistoryData(req_id, symbol, tf, count);
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
   // Trim spaces if any (simple trim)
   // StringTrimLeft(valStr); StringTrimRight(valStr); 
   
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
void SendHistoryData(string req_id, string symbol, ENUM_TIMEFRAMES tf, int count) {
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    int copied = CopyRates(symbol, tf, 0, count, rates);
    
    if(copied > 0) {
        // Build JSON
        string json = StringFormat("{\"request_id\":\"%s\",\"symbol\":\"%s\",\"timeframe\":\"%s\",\"count\":%d,\"data\":[", 
                                   req_id, symbol, EnumToString(tf), copied);
        
        for(int i=0; i<copied; i++) {
            if(i > 0) json += ",";
            // Use simple format to save space
            json += StringFormat("{\"time\":%d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_volume\":%d}",
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
//| Get Filling Mode (Fix 4756 Error)                                |
//+------------------------------------------------------------------+
uint GetFillingMode(string symbol)
  {
   long mode = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((mode & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   if((mode & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN;
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
            "{\"ticket\":%d,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"open_price\":%.5f,\"current_price\":%.5f,\"pnl\":%.2f,\"swap\":%.2f,\"sl\":%.5f,\"tp\":%.5f,\"comment\":\"%s\"}",
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
         Print("Found Position: Ticket=", ticket, " Symbol=", symbol);
        }
     }
   positions_json += "]";

   // 3. Current Chart Quote (Legacy + Active Symbol)
   double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   
   string legacy_json = StringFormat("\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f", 
                              Symbol(), bid, ask);

   // Combine Final JSON
   string json = "{" + account_json + "," + positions_json + "," + legacy_json + "}";
   
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   string result_headers;
   
   // Debug: Print JSON if positions > 0
   if (added_count > 0) Print("Sending Status JSON: ", json);
   
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
         
         // Filter out non-trade types (e.g. Balance)
         if(type > DEAL_TYPE_SELL) return;

         string side = (type == DEAL_TYPE_BUY) ? "BUY" : "SELL";
         string entry_str = (entry == DEAL_ENTRY_IN) ? "IN" : (entry == DEAL_ENTRY_OUT) ? "OUT" : "INOUT";
         
         // Build JSON - Use %I64u for ulong ticket
         string json = StringFormat("{\"ticket\":%I64u,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"time\":\"%I64d\",\"entry\":\"%s\",\"position_id\":%I64d,\"profit\":%.2f,\"commission\":%.2f,\"swap\":%.2f}",
                                    ticket, symbol, side, volume, price, time, entry_str, position_id, profit, commission, swap);
         
         Print("Reporting Trade: ", json);
         
         // Send Data
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
   request.deviation = 10; // Allow 10 points deviation
   request.sl = sl;
   request.tp = tp;
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol); // Auto-adapt filling mode
   
   if(OrderSend(request, result)) 
     {
      Print("Trade Executed! Ticket: ", result.deal);
     }
   else 
     {
      Print("Trade Failed: ", result.retcode, " Comment: ", result.comment);
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
   request.deviation = 10;
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
       Print("Position Closed! Ticket: ", ticket);
   } else {
       Print("Close Failed: ", result.retcode);
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
   request.type_time = ORDER_TIME_GTC; // Good Till Cancelled
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol);
   
   if(OrderSend(request, result)) 
     {
      Print("Pending Order Placed! Ticket: ", result.order);
     }
   else 
     {
      Print("Pending Order Failed: ", result.retcode, " Comment: ", result.comment);
   }
  }
