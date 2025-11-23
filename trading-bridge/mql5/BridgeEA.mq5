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

void OnTimer()
  {
   CheckForCommands();
   SendStatusUpdate();
  }

void CheckForCommands()
  {
   char data[];
   char result[];
   string headers;
   string cookie = NULL;
   string referer = NULL;
   
   // GET 请求的标准调用方式
   int res = WebRequest("GET", ApiUrl + "/commands/pop", cookie, referer, 500, data, 0, result, headers);
   
   if(res == 200)
     {
      string json = CharArrayToString(result);
      if(json != "null" && StringLen(json) > 5) 
        {
         Print("Received command: ", json);
         
         string type = ExtractJsonString(json, "type");
         
         if (type == "TRADE" || type == "") { // Default to TRADE if type missing (legacy)
             string action = ExtractJsonString(json, "action");
             string symbol = ExtractJsonString(json, "symbol");
             double volume = ExtractJsonDouble(json, "volume");
             double sl = ExtractJsonDouble(json, "sl");
             double tp = ExtractJsonDouble(json, "tp");
             long ticket = (long)ExtractJsonDouble(json, "ticket");
             
             if(action == "CLOSE") {
                 if(ticket > 0) ClosePosition(ticket);
                 else Print("CLOSE command missing ticket");
             }
             else {
                 if(volume <= 0) volume = 0.01;
                 if(symbol == "" || symbol == NULL) symbol = Symbol(); 
                 
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
//| 辅助函数: 提取 JSON 字符串值                                         |
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
//| 辅助函数: 提取 JSON 数值                                           |
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
   return StringToDouble(StringSubstr(json, start, end - start));
  }

//+------------------------------------------------------------------+
//| 字符串转时间周期                                                   |
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
//| 获取历史数据并发送                                                 |
//+------------------------------------------------------------------+
void SendHistoryData(string req_id, string symbol, ENUM_TIMEFRAMES tf, int count) {
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    int copied = CopyRates(symbol, tf, 0, count, rates);
    
    if(copied > 0) {
        // 构建 JSON
        string json = StringFormat("{\"request_id\":\"%s\",\"symbol\":\"%s\",\"timeframe\":\"%s\",\"count\":%d,\"data\":[", 
                                   req_id, symbol, EnumToString(tf), copied);
        
        for(int i=0; i<copied; i++) {
            if(i > 0) json += ",";
            // 使用简单格式以节省空间
            json += StringFormat("{\"time\":%d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_volume\":%d}",
                                 rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume);
        }
        json += "]}";
        
        // 发送
        char data[];
        StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
        char res_data[];
        string headers = "Content-Type: application/json\r\n";
        
        // 增加超时时间到 2000ms，因为数据量较大
        WebRequest("POST", ApiUrl + "/data/history", headers, 2000, data, res_data, headers);
        Print("Sent history data: ", copied, " candles for ", symbol);
    } else {
        Print("Failed to copy rates for ", symbol);
    }
}

//+------------------------------------------------------------------+
//| 获取正确的填充模式 (解决 4756 错误)                                   |
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
   // 1. 账户信息
   string account_json = StringFormat(
      "\"account\":{\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f}",
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE)
   );

   // 2. 持仓列表
   string positions_json = "\"positions\":[";
   int total = PositionsTotal();
   // Print("Total Positions: ", total); // Debug
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
        {
         if(i > 0) positions_json += ",";
         
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
        }
     }
   positions_json += "]";

   // 3. 当前图表报价 (Legacy + Active Symbol)
   double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   
   string legacy_json = StringFormat("\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f", 
                              Symbol(), bid, ask);

   // 组合最终 JSON
   string json = "{" + account_json + "," + positions_json + "," + legacy_json + "}";
   
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   
   // Debug: Print JSON if positions > 0
   if (total > 0) Print("Sending Status JSON: ", json);
   
   WebRequest("POST", ApiUrl + "/status/update", headers, 500, data, result, headers);
  }

//+------------------------------------------------------------------+
//| 监听交易事务 (成交回报)                                            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
  {
   // 只关心“交易历史添加”事件 (成交)
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      long ticket = trans.deal;
      // 选中该笔成交以获取详细信息
      if(HistoryDealSelect(ticket))
        {
         string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
         long type = HistoryDealGetInteger(ticket, DEAL_TYPE); // 0=BUY, 1=SELL
         double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
         double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
         long time = HistoryDealGetInteger(ticket, DEAL_TIME);
         
         // 过滤掉非交易类型 (如 Balance)
         if(type > DEAL_TYPE_SELL) return;

         string side = (type == DEAL_TYPE_BUY) ? "BUY" : "SELL";
         
         // 构造 JSON
         string json = StringFormat("{\"ticket\":%d,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"time\":\"%d\"}",
                                    ticket, symbol, side, volume, price, time);
         
         Print("Reporting Trade: ", json);
         
         // 发送数据
         char data[];
         StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
         char res_data[];
         string headers = "Content-Type: application/json\r\n";
         
         WebRequest("POST", ApiUrl + "/trade/report", headers, 1000, data, res_data, headers);
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
   request.deviation = 10; // 允许 10 点滑点
   request.sl = sl;
   request.tp = tp;
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol); // 自动适配填充模式
   
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
