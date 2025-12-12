//+------------------------------------------------------------------+
//|                                            AlphaOS_Executor.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                     Order Execution & Reporting  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.00"

input string ApiUrl = "http://192.168.3.8:8000"; 
input int PollInterval = 1000;
input int MagicNumber = 888888;
input int SlippagePoints = 20;

int OnInit()
  {
   Print("AlphaOS Executor: Initializing...");
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
   // Executor does not need OnTick logic, it's timer driven
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
   
   // Poll Command Queue
   int res = WebRequest("GET", ApiUrl + "/commands/pop", "", 500, data, result, result_headers);
   
   if(res == 200)
     {
      string json = CharArrayToString(result);
      if(json != "null" && StringLen(json) > 5) 
        {
         Print("Received command: ", json);
         
         string type = ExtractJsonString(json, "type");
         string action = ExtractJsonString(json, "action");
         string symbol = ExtractJsonString(json, "symbol");
         double volume = ExtractJsonDouble(json, "volume");
         double price = ExtractJsonDouble(json, "price");
         double sl = ExtractJsonDouble(json, "sl");
         double tp = ExtractJsonDouble(json, "tp");
         long ticket = (long)ExtractJsonDouble(json, "ticket");
         
         // Clean inputs
         if(symbol == "") symbol = Symbol();
         if(volume <= 0) volume = 0.01;

         if (type == "TRADE" || type == "") {
             if(action == "CLOSE") {
                 if(ticket > 0) {
                     Print("Executing CLOSE command for ticket: ", ticket);
                     ClosePosition(ticket);
                 }
                 else Print("CLOSE command missing ticket or invalid ticket: ", ticket);
             }
             else if(action == "BUY") {
                 ExecuteTrade(symbol, ORDER_TYPE_BUY, volume, sl, tp);
             }
             else if(action == "SELL") {
                 ExecuteTrade(symbol, ORDER_TYPE_SELL, volume, sl, tp);
             }
         }
         else if (type == "PENDING") {
             // PENDING logic
             ENUM_ORDER_TYPE order_type;
             double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
             double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
             
             string side = action; // BUY or SELL
             
             if (side == "BUY") {
                 if (price > current_ask) order_type = ORDER_TYPE_BUY_STOP;
                 else order_type = ORDER_TYPE_BUY_LIMIT;
             } else { // SELL
                 if (price < current_bid) order_type = ORDER_TYPE_SELL_STOP;
                 else order_type = ORDER_TYPE_SELL_LIMIT;
             }
             
             ExecutePendingOrder(symbol, order_type, volume, price, sl, tp);
         }
        }
     }
  }

//+------------------------------------------------------------------+
//| Helpers                                                          |
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

uint GetFillingMode(string symbol)
  {
   long mode = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((mode & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   if((mode & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN;
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

//+------------------------------------------------------------------+
//| Status Reporting                                                 |
//+------------------------------------------------------------------+
void SendStatusUpdate()
  {
   string account_json = StringFormat(
      "\"account\":{\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f}",
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE)
   );

   string positions_json = "\"positions\":[";
   int total = PositionsTotal();
   int added_count = 0;
   
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
        {
         if(added_count > 0) positions_json += ",";
         
         string symbol = PositionGetString(POSITION_SYMBOL);
         string comment = PositionGetString(POSITION_COMMENT);
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

   // Only send Account, Positions, and Quotes (Scanner handles DOM)
   string json = "{" + account_json + "," + positions_json + "," + quotes_json + "}";
   
   char data[];
   StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   string result_headers;
   
   WebRequest("POST", ApiUrl + "/status/update", headers, 500, data, result, result_headers);
  }

//+------------------------------------------------------------------+
//| Trade Execution                                                  |
//+------------------------------------------------------------------+
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
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol);
   request.comment = "AI-AlphaOS";
   
   if(OrderSend(request, result)) 
      Print("✅ Trade Executed! Ticket: ", result.deal);
   else 
      Print("❌ Trade Failed: Code=", result.retcode, " (", GetRetcodeDescription(result.retcode), ")");
  }

void ClosePosition(long ticket)
  {
   if(!PositionSelectByTicket(ticket)) return;
   
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
   
   if(type == POSITION_TYPE_BUY) {
       request.type = ORDER_TYPE_SELL;
       request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
   } else {
       request.type = ORDER_TYPE_BUY;
       request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }
   
   if(OrderSend(request, result))
       Print("✅ Position Closed! Ticket: ", ticket);
   else
       Print("❌ Close Failed: ", result.retcode);
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
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)GetFillingMode(symbol);
   request.comment = "AI-Pending";
   
   if(OrderSend(request, result)) 
      Print("✅ Pending Order Placed! Ticket: ", result.order);
   else 
      Print("❌ Pending Order Failed: ", result.retcode);
  }

//+------------------------------------------------------------------+
//| Trade Reporting                                                  |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& _request,
                        const MqlTradeResult& _result)
  {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      ulong ticket = trans.deal;
      if(HistoryDealSelect(ticket))
        {
         string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
         long type = HistoryDealGetInteger(ticket, DEAL_TYPE);
         
         if(type > DEAL_TYPE_SELL) return; // Skip balance ops

         double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
         double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
         long time = HistoryDealGetInteger(ticket, DEAL_TIME);
         long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
         long position_id = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
         
         string side = (type == DEAL_TYPE_BUY) ? "BUY" : "SELL";
         string entry_str = (entry == DEAL_ENTRY_IN) ? "IN" : (entry == DEAL_ENTRY_OUT) ? "OUT" : "INOUT";
         
         string json = StringFormat("{\"ticket\":%I64u,\"symbol\":\"%s\",\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"time\":\"%I64d\",\"entry\":\"%s\",\"position_id\":%I64d,\"profit\":%.2f,\"commission\":%.2f,\"swap\":%.2f}",
                                    ticket, symbol, side, volume, price, time, entry_str, position_id, profit, commission, swap);
         
         char data[];
         StringToCharArray(json, data, 0, StringLen(json), CP_UTF8);
         char res_data[];
         string headers = "Content-Type: application/json\r\n";
         string result_headers;
         
         WebRequest("POST", ApiUrl + "/trade/report", headers, 1000, data, res_data, result_headers);
        }
     }
  }

