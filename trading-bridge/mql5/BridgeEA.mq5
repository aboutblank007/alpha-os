//+------------------------------------------------------------------+
//|                                                     BridgeEA.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                       HTTP Polling Version (MVP) |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.00"

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
   // int WebRequest(const string method,const string url,const string cookie,const string referer,int timeout,const char &data[],int data_size,char &result[],string &result_headers)
   int res = WebRequest("GET", ApiUrl + "/commands/pop", cookie, referer, 500, data, 0, result, headers);
   
   if(res == 200)
     {
      string json = CharArrayToString(result);
      if(json != "null" && StringLen(json) > 5) 
        {
         Print("Received command: ", json);
         
         // 简单的 JSON 解析逻辑
         string action = ExtractJsonString(json, "action");
         string symbol = ExtractJsonString(json, "symbol");
         double volume = ExtractJsonDouble(json, "volume");
         
         if(volume <= 0) volume = 0.01;
         if(symbol == "" || symbol == NULL) symbol = Symbol(); // 默认当前品种
         
         if(action == "BUY") ExecuteTrade(symbol, ORDER_TYPE_BUY, volume);
         else if(action == "SELL") ExecuteTrade(symbol, ORDER_TYPE_SELL, volume);
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
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
        {
         if(i > 0) positions_json += ",";
         positions_json += StringFormat(
            "{\"ticket\":%d,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"open_price\":%.5f,\"current_price\":%.5f,\"pnl\":%.2f,\"swap\":%.2f,\"comment\":\"%s\"}",
            ticket,
            PositionGetString(POSITION_SYMBOL),
            (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "BUY" : "SELL"),
            PositionGetDouble(POSITION_VOLUME),
            PositionGetDouble(POSITION_PRICE_OPEN),
            PositionGetDouble(POSITION_PRICE_CURRENT),
            PositionGetDouble(POSITION_PROFIT),
            PositionGetDouble(POSITION_SWAP),
            PositionGetString(POSITION_COMMENT)
         );
        }
     }
   positions_json += "]";

   // 3. 兼容旧字段 (Legacy)
   double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   if(bid <= 0) bid = 1.12345;
   if(ask <= 0) ask = 1.12355;
   
   string legacy_json = StringFormat("\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f", 
                              Symbol(), bid, ask);

   // 组合最终 JSON
   string json = "{" + account_json + "," + positions_json + "," + legacy_json + "}";
   
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   
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

void ExecuteTrade(string symbol, ENUM_ORDER_TYPE type, double volume)
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
