//+------------------------------------------------------------------+
//|                                             AlphaOS_Executor.mq5 |
//|                                                   AlphaOS Team   |
//|                                             https://github.com/alphaos |
//+------------------------------------------------------------------+
// [Ref: AlphaOS 工作流设计与规范.md] 第5节 ZeroMQ 桥接架构设计
// [Ref: 自动化交易系统落地指南.md] 第4节 系统架构
// [Ref: GEMINI.md] 通信协议定义
#property copyright "AlphaOS Quantum HFT"
#property link      "https://github.com/alphaos"
#property version   "4.00"
#property strict

// Dependencies
// ZeroMQ 库 v3.0: https://github.com/ding9736/MQL5-ZeroMQ
#include <ZeroMQ/ZeroMQ.mqh>
#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| 输入参数                                                          |
//+------------------------------------------------------------------+
input group "ZeroMQ 配置"
input string InpBindAddressPub = "tcp://*:5555";     // 行情发布地址 (PUB)
input string InpBindAddressRep = "tcp://*:5556";     // 订单执行地址 (ROUTER)
input string InpHeartbeatAddr  = "tcp://*:5558";     // 心跳地址 (PAIR)
input string InpHistoryAddr    = "tcp://*:5559";     // 历史数据地址 (REP)
input int    InpHighWaterMark  = 1000;               // 消息缓冲区大小

input group "交易配置"
input int    InpMagicNumber    = 202412;             // EA 魔术数字
input int    InpSlippagePoints = 10;                 // 滑点容忍（点数）
input double InpMaxVolume      = 5.0;                // 单笔最大手数 (Kelly sizing may need up to 3+ lots)
input int    InpMaxPositions   = 2;                  // 最大持仓数（按 symbol + magic 计数）

input group "风控配置"
input double InpMaxDailyLossPct  = 2.0;              // 当日最大亏损百分比
input double InpMaxDrawdownPct   = 5.0;              // 最大回撤百分比
input bool   InpEnableCircuitBreaker = true;         // 启用熔断机制

input group "调试配置"
input bool   InpUseBinaryTick = true;                // 使用二进制 Tick 格式
input bool   InpVerboseLog    = false;               // 详细日志

//+------------------------------------------------------------------+
//| 全局对象 (MQL5-ZeroMQ v3.0 API)                                   |
//+------------------------------------------------------------------+
ZmqContext *g_context = NULL;           // ZeroMQ 上下文 (指针管理)
ZmqSocket  *g_socket_pub = NULL;        // PUB 套接字 (行情发布)
ZmqSocket  *g_socket_rep = NULL;        // ROUTER 套接字 (订单执行)
ZmqSocket  *g_socket_heartbeat = NULL;  // PAIR 套接字 (心跳监控)
ZmqSocket  *g_socket_history = NULL;    // REP 套接字 (历史数据请求)

CTrade trade;

// 状态变量
double g_startingEquity = 0;
double g_peakEquity = 0;
double g_dailyPnL = 0;
bool   g_circuitBreakerTripped = false;
datetime g_lastHeartbeat = 0;
int    g_heartbeatSequence = 0;
long   g_tickCount = 0;
long   g_orderCount = 0;

// ============================================================================
// v4.0: Tick History Replay 状态机（用于 BOOTSTRAP_REPLAY）
// 协议: START_REPLAY_TICKS|SYMBOL|WINDOW_SEC|END_EPS_MS|MAX_TICKS|PACE_TPS
// ============================================================================
bool      g_replayActive = false;       // 回放是否激活
MqlTick   g_replayTicks[];              // 历史 tick 缓存
int       g_replayIdx = 0;              // 当前发送位置
int       g_replayTotal = 0;            // 总 tick 数
int       g_replayPaceTps = 50000;      // 每秒发送速度（ticks/sec）
int       g_replaySent = 0;             // 已发送计数
datetime  g_replayStartTime = 0;        // 回放开始时间戳
datetime  g_replayTicksStart = 0;       // tick 历史起始时间
datetime  g_replayTicksEnd = 0;         // tick 历史结束时间

// 二进制 Tick 结构 (用于 StructToCharArray)
// 与 Python 端 protocol.py 对应: struct.pack("<ddqqi", bid, ask, time_msc, volume, flags)
struct BinaryTick
{
    double bid;       // 8 bytes
    double ask;       // 8 bytes
    long   time_msc;  // 8 bytes
    long   volume;    // 8 bytes
    int    flags;     // 4 bytes
};
// 总大小: 36 bytes

//+------------------------------------------------------------------+
//| JSON 解析辅助函数                                                 |
//+------------------------------------------------------------------+
string JsonGetString(const string &json, const string key)
{
    // 尝试两种格式: "key":"value" 或 "key": "value" (冒号后有空格)
    string searchKey1 = "\"" + key + "\":\"";      // 无空格
    string searchKey2 = "\"" + key + "\": \"";     // 有空格
    
    int startPos = StringFind(json, searchKey1);
    int keyLen = StringLen(searchKey1);
    
    if(startPos < 0)
    {
        startPos = StringFind(json, searchKey2);
        keyLen = StringLen(searchKey2);
    }
    
    if(startPos < 0) return "";
    
    startPos += keyLen;
    int endPos = StringFind(json, "\"", startPos);
    if(endPos < 0) return "";
    
    return StringSubstr(json, startPos, endPos - startPos);
}

double JsonGetDouble(const string &json, const string key)
{
    // 尝试两种格式: "key":value 或 "key": value
    string searchKey1 = "\"" + key + "\":";
    string searchKey2 = "\"" + key + "\": ";
    
    int startPos = StringFind(json, searchKey1);
    int keyLen = StringLen(searchKey1);
    
    if(startPos < 0)
    {
        startPos = StringFind(json, searchKey2);
        keyLen = StringLen(searchKey2);
    }
    
    if(startPos < 0)
    {
        if(key == "volume") Print("⚠️ JSON 解析: 未找到 key='", key, "'");
        return 0.0;
    }
    
    startPos += keyLen;
    
    // 跳过可能的额外空格
    while(startPos < StringLen(json) && StringGetCharacter(json, startPos) == ' ')
        startPos++;
    
    // 查找数值结束位置
    int endPos = startPos;
    while(endPos < StringLen(json))
    {
        ushort ch = StringGetCharacter(json, endPos);
        if(ch != '.' && ch != '-' && ch != 'e' && ch != 'E' && ch != '+' && (ch < '0' || ch > '9'))
            break;
        endPos++;
    }
    
    string numStr = StringSubstr(json, startPos, endPos - startPos);
    double result = StringToDouble(numStr);
    
    // v3.4: 手数解析调试
    if(key == "volume")
    {
        Print("📊 Volume 解析: numStr='", numStr, "' -> ", DoubleToString(result, 4));
    }
    
    return result;
}

int JsonGetInt(const string &json, const string key)
{
    return (int)JsonGetDouble(json, key);
}

//+------------------------------------------------------------------+
//| 风控检查                                                          |
//+------------------------------------------------------------------+
bool CheckRiskLimits()
{
    if(!InpEnableCircuitBreaker) return true;
    if(g_circuitBreakerTripped) return false;
    
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // 更新峰值
    if(currentEquity > g_peakEquity)
        g_peakEquity = currentEquity;
    
    // 计算当日 PnL
    g_dailyPnL = currentEquity - g_startingEquity;
    
    // 当日亏损检查
    double dailyLossPct = 0;
    if(g_startingEquity > 0)
        dailyLossPct = -g_dailyPnL / g_startingEquity * 100;
    
    if(dailyLossPct >= InpMaxDailyLossPct)
    {
        Print("⚠️ 熔断触发: 当日亏损 ", DoubleToString(dailyLossPct, 2), "% >= ", InpMaxDailyLossPct, "%");
        g_circuitBreakerTripped = true;
        return false;
    }
    
    // 回撤检查
    double drawdownPct = 0;
    if(g_peakEquity > 0)
        drawdownPct = (g_peakEquity - currentEquity) / g_peakEquity * 100;
    
    if(drawdownPct >= InpMaxDrawdownPct)
    {
        Print("⚠️ 熔断触发: 回撤 ", DoubleToString(drawdownPct, 2), "% >= ", InpMaxDrawdownPct, "%");
        g_circuitBreakerTripped = true;
        return false;
    }
    
    return true;
}

int CountOpenPositions(const string &symbol)
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong posTicket = PositionGetTicket(i);
        if(!PositionSelectByTicket(posTicket))
            continue;
        
        if(PositionGetString(POSITION_SYMBOL) == symbol &&
           PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
        {
            count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| 执行订单                                                          |
//+------------------------------------------------------------------+
string ExecuteOrder(const string &orderJson)
{
    // 调试：显示原始 JSON
    if(InpVerboseLog || StringLen(orderJson) < 300)
    {
        Print("📦 原始 JSON (", StringLen(orderJson), " bytes): ", orderJson);
    }
    
    // 解析 JSON
    string action = JsonGetString(orderJson, "action");
    string symbol = JsonGetString(orderJson, "symbol");
    double volume = JsonGetDouble(orderJson, "volume");
    double price  = JsonGetDouble(orderJson, "price");
    double sl     = JsonGetDouble(orderJson, "sl");
    double tp     = JsonGetDouble(orderJson, "tp");
    int    deviation = JsonGetInt(orderJson, "deviation");
    string requestId = JsonGetString(orderJson, "request_id");
    ulong  closeTicket = (ulong)JsonGetDouble(orderJson, "ticket");  // 平仓 ticket
    
    // 调试：解析结果 (v3.4: 始终打印手数解析结果)
    Print("📋 解析结果: action=", action, " symbol=", symbol, " vol=", DoubleToString(volume, 4), " ticket=", closeTicket);
    
    // 参数验证
    if(symbol == "") symbol = Symbol();
    if(volume <= 0)
    {
        Print("⚠️ 手数无效 (", DoubleToString(volume, 4), "), 使用默认 0.01");
        volume = 0.01;
    }
    if(volume > InpMaxVolume)
    {
        Print("⚠️ 手数超限 (", DoubleToString(volume, 2), " > ", DoubleToString(InpMaxVolume, 2), "), 限制为 ", DoubleToString(InpMaxVolume, 2));
        volume = InpMaxVolume;
    }
    if(deviation <= 0) deviation = InpSlippagePoints;
    
    // 如果指定了 ticket，这是平仓请求
    if(closeTicket > 0)
    {
        Print("📥 平仓请求: Ticket=", closeTicket, " [", requestId, "]");
    }
    else
    {
        Print("📥 开仓请求: ", action, " ", symbol, " ", DoubleToString(volume, 2), " lots [", requestId, "]");
    }
    
    // 风控检查
    if(!CheckRiskLimits())
    {
        return StringFormat(
            "{\"request_id\":\"%s\",\"status\":\"REJECTED\",\"error_code\":1001,\"error_message\":\"Circuit breaker tripped\",\"timestamp\":%I64d}",
            requestId, TimeCurrent() * 1000
        );
    }
    
    // 最大持仓数限制（仅开仓请求）
    if(closeTicket == 0)
    {
        bool isOpenAction = (
            action == "BUY" || action == "SELL" ||
            action == "BUY_LIMIT" || action == "SELL_LIMIT" ||
            action == "BUY_STOP" || action == "SELL_STOP"
        );
        if(isOpenAction)
        {
            int openCount = CountOpenPositions(symbol);
            if(openCount >= InpMaxPositions)
            {
                Print("⚠️ 持仓上限: 当前=", openCount, " >= ", InpMaxPositions, "，拒绝开仓");
                return StringFormat(
                    "{\"request_id\":\"%s\",\"status\":\"REJECTED\",\"error_code\":1003,\"error_message\":\"Max positions reached\",\"timestamp\":%I64d}",
                    requestId, TimeCurrent() * 1000
                );
            }
        }
    }
    
    // 设置交易参数
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(deviation);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    bool result = false;
    ulong ticket = 0;
    double filledPrice = 0;
    double filledVolume = 0;
    string status = "REJECTED";
    int errorCode = 0;
    string errorMessage = "";
    
    // 执行订单
    if(closeTicket > 0)
    {
        // ========== 平仓请求：使用 PositionClose ==========
        // 先选择该持仓
        if(PositionSelectByTicket(closeTicket))
        {
            result = trade.PositionClose(closeTicket, deviation);
            if(result)
            {
                Print("✅ 平仓成功: Ticket=", closeTicket);
            }
        }
        else
        {
            // 持仓可能已被平掉或不存在
            Print("⚠️ 持仓不存在: Ticket=", closeTicket);
            errorCode = 4756;  // TRADE_RETCODE_POSITION_NOT_FOUND
            errorMessage = "Position not found";
        }
    }
    else if(action == "BUY")
    {
        // ========== 开仓请求 ==========
        double askPrice = SymbolInfoDouble(symbol, SYMBOL_ASK);
        result = trade.Buy(volume, symbol, askPrice, sl, tp, "AlphaOS");
    }
    else if(action == "SELL")
    {
        double bidPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
        result = trade.Sell(volume, symbol, bidPrice, sl, tp, "AlphaOS");
    }
    else if(action == "BUY_LIMIT")
    {
        result = trade.BuyLimit(volume, price, symbol, sl, tp, ORDER_TIME_GTC, 0, "AlphaOS");
    }
    else if(action == "SELL_LIMIT")
    {
        result = trade.SellLimit(volume, price, symbol, sl, tp, ORDER_TIME_GTC, 0, "AlphaOS");
    }
    else if(action == "BUY_STOP")
    {
        result = trade.BuyStop(volume, price, symbol, sl, tp, ORDER_TIME_GTC, 0, "AlphaOS");
    }
    else if(action == "SELL_STOP")
    {
        result = trade.SellStop(volume, price, symbol, sl, tp, ORDER_TIME_GTC, 0, "AlphaOS");
    }
    else if(action == "CLOSE")
    {
        // 平掉所有指定品种的持仓
        int closedCount = 0;
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong posTicket = PositionGetTicket(i);
            if(PositionSelectByTicket(posTicket))
            {
                if(PositionGetString(POSITION_SYMBOL) == symbol &&
                   PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
                {
                    if(trade.PositionClose(posTicket))
                    {
                        closedCount++;
                        result = true;
                    }
                }
            }
        }
        if(closedCount > 0)
        {
            Print("平仓完成: 关闭 ", closedCount, " 个持仓");
        }
    }
    else if(action == "CLOSE_ALL")
    {
        // 平掉所有 AlphaOS 持仓
        int closedCount = 0;
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong posTicket = PositionGetTicket(i);
            if(PositionSelectByTicket(posTicket))
            {
                if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
                {
                    if(trade.PositionClose(posTicket))
                    {
                        closedCount++;
                        result = true;
                    }
                }
            }
        }
        Print("全部平仓: 关闭 ", closedCount, " 个持仓");
    }
    else
    {
        errorCode = 1002;
        errorMessage = "Unknown action: " + action;
    }
    
    // 获取结果 - 检查 RetCode 而不仅仅依赖返回值
    uint retcode = trade.ResultRetcode();
    
    // TRADE_RETCODE_DONE (10009) 或 TRADE_RETCODE_DONE_PARTIAL (10010) 表示成功
    if(result || retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_DONE_PARTIAL)
    {
        ticket = trade.ResultOrder();
        filledPrice = trade.ResultPrice();
        filledVolume = trade.ResultVolume();
        status = "FILLED";
        g_orderCount++;
        Print("✅ 订单成交: Ticket=", ticket, " Price=", DoubleToString(filledPrice, 5), " Vol=", DoubleToString(filledVolume, 2));
    }
    else
    {
        errorCode = (int)retcode;
        errorMessage = trade.ResultRetcodeDescription();
        Print("❌ 订单失败: ", errorCode, " - ", errorMessage);
    }
    
    // 构建响应 JSON
    return StringFormat(
        "{\"request_id\":\"%s\",\"status\":\"%s\",\"ticket\":%I64d,\"volume_filled\":%.2f,\"price_filled\":%.5f,\"error_code\":%d,\"error_message\":\"%s\",\"timestamp\":%I64d}",
        requestId, status, ticket, filledVolume, filledPrice, errorCode, errorMessage, TimeCurrent() * 1000
    );
}

//+------------------------------------------------------------------+
//| 发送二进制 Tick 数据                                              |
//| 格式: <ddqqi (bid:8, ask:8, time_msc:8, volume:8, flags:4)       |
//| 总大小: 36 bytes，小端序                                          |
//+------------------------------------------------------------------+
void SendBinaryTick(const MqlTick &tick)
{
    if(CheckPointer(g_socket_pub) == POINTER_INVALID) return;
    
    // 使用结构体序列化
    BinaryTick binTick;
    binTick.bid = tick.bid;
    binTick.ask = tick.ask;
    binTick.time_msc = tick.time_msc;
    // 优先使用 real_volume（真实成交量），如果为0则回退到 tick_volume
    binTick.volume = (tick.volume_real > 0) ? (long)tick.volume_real : (long)tick.volume;
    binTick.flags = (int)tick.flags;
    
    uchar data[];
    StructToCharArray(binTick, data);
    
    // 发送 (v3.0 API: send with uchar[])
    g_socket_pub.send(data);
    
    g_tickCount++;
    
    if(InpVerboseLog && g_tickCount % 1000 == 0)
    {
        Print("📊 Tick #", g_tickCount, " Bid=", DoubleToString(tick.bid, 5), " Ask=", DoubleToString(tick.ask, 5),
              " Vol=", tick.volume, " RealVol=", tick.volume_real);
    }
    
    // 首次 1000 个 Tick 时打印一次成交量信息，帮助调试
    if(g_tickCount == 1000)
    {
        Print("📊 成交量检测: volume=", tick.volume, " volume_real=", tick.volume_real,
              " (使用: ", (tick.volume_real > 0 ? "real_volume" : "tick_volume"), ")");
    }
}

//+------------------------------------------------------------------+
//| 发送 JSON Tick 数据（调试用）                                     |
//+------------------------------------------------------------------+
void SendJsonTick(const MqlTick &tick)
{
    if(CheckPointer(g_socket_pub) == POINTER_INVALID) return;
    
    string json = StringFormat(
        "{\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"volume\":%I64d,\"time_msc\":%I64d,\"flags\":%d}",
        Symbol(), tick.bid, tick.ask, tick.volume, tick.time_msc, (int)tick.flags
    );
    
    // v3.0 API: send string directly
    g_socket_pub.send(json);
    
    g_tickCount++;
}

//+------------------------------------------------------------------+
//| 处理心跳                                                          |
//+------------------------------------------------------------------+
void HandleHeartbeat()
{
    if(CheckPointer(g_socket_heartbeat) == POINTER_INVALID) return;
    
    string received_msg;
    
    // v3.0 API: 非阻塞 recv，避免阻塞 OnTimer
    if(g_socket_heartbeat.recv(received_msg, ZMQ_FLAG_DONTWAIT))
    {
        // 解析 timestamp|sequence
        int sepPos = StringFind(received_msg, "|");
        if(sepPos > 0)
        {
            long timestamp = StringToInteger(StringSubstr(received_msg, 0, sepPos));
            int sequence = (int)StringToInteger(StringSubstr(received_msg, sepPos + 1));
            
            // 回复相同的消息（用于延迟计算）
            g_socket_heartbeat.send(received_msg);
            
            g_lastHeartbeat = TimeCurrent();
            g_heartbeatSequence = sequence;
            
            if(InpVerboseLog)
            {
                Print("💓 Heartbeat #", sequence);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 初始化日开始状态                                                  |
//+------------------------------------------------------------------+
void InitDayStart()
{
    g_startingEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    g_peakEquity = g_startingEquity;
    g_dailyPnL = 0;
    g_circuitBreakerTripped = false;
    
    Print("📅 日初权益: ", DoubleToString(g_startingEquity, 2));
}

//+------------------------------------------------------------------+
//| 获取状态 JSON                                                     |
//+------------------------------------------------------------------+
string GetStatusJson()
{
    return StringFormat(
        "{\"equity\":%.2f,\"daily_pnl\":%.2f,\"drawdown_pct\":%.2f,\"circuit_breaker\":%s,\"tick_count\":%I64d,\"order_count\":%I64d,\"version\":\"3.00\"}",
        AccountInfoDouble(ACCOUNT_EQUITY),
        g_dailyPnL,
        g_peakEquity > 0 ? (g_peakEquity - AccountInfoDouble(ACCOUNT_EQUITY)) / g_peakEquity * 100 : 0,
        g_circuitBreakerTripped ? "true" : "false",
        g_tickCount,
        g_orderCount
    );
}

//+------------------------------------------------------------------+
//| 获取持仓列表 JSON                                                  |
//| [Ref: AlphaOS 工作流设计与规范.md] MT5 持仓同步                    |
//+------------------------------------------------------------------+
string GetPositionsJson(const string filterSymbol = "")
{
    string result = "{\"positions\":[";
    int posCount = 0;
    int totalPos = PositionsTotal();
    
    Print("📊 GetPositionsJson: 总持仓数=", totalPos, " 过滤品种=", filterSymbol, " MagicNumber=", InpMagicNumber);
    
    for(int i = 0; i < totalPos; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        
        // 调试：显示每个持仓的信息
        string sym = PositionGetString(POSITION_SYMBOL);
        long magic = PositionGetInteger(POSITION_MAGIC);
        Print("   持仓[", i, "]: ticket=", ticket, " symbol=", sym, " magic=", magic);
        
        // 只返回 AlphaOS 的持仓（通过 Magic Number 过滤）
        // 如果 Magic 为 0，也返回（可能是手动开的仓）
        if(magic != InpMagicNumber && magic != 0) continue;
        
        // 如果指定了品种过滤
        string symbol = PositionGetString(POSITION_SYMBOL);
        if(filterSymbol != "" && symbol != filterSymbol) continue;
        
        // 获取持仓信息
        ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        string direction = (posType == POSITION_TYPE_BUY) ? "LONG" : "SHORT";
        double volume = PositionGetDouble(POSITION_VOLUME);
        double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
        double profit = PositionGetDouble(POSITION_PROFIT);
        double swap = PositionGetDouble(POSITION_SWAP);
        double sl = PositionGetDouble(POSITION_SL);
        double tp = PositionGetDouble(POSITION_TP);
        datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
        
        // 添加逗号分隔（第二个持仓开始）
        if(posCount > 0) result += ",";
        
        // 构建持仓 JSON
        result += StringFormat(
            "{\"ticket\":%I64d,\"symbol\":\"%s\",\"direction\":\"%s\",\"volume\":%.2f,"
            "\"entry_price\":%.5f,\"current_price\":%.5f,\"profit\":%.2f,\"swap\":%.2f,"
            "\"sl\":%.5f,\"tp\":%.5f,\"open_time\":\"%s\",\"magic\":%I64d}",
            ticket, symbol, direction, volume,
            entryPrice, currentPrice, profit, swap,
            sl, tp, TimeToString(openTime, TIME_DATE|TIME_SECONDS), magic
        );
        
        posCount++;
    }
    
    result += StringFormat("],\"count\":%d,\"timestamp\":%I64d}", posCount, TimeCurrent() * 1000);
    
    Print("📊 查询持仓: 找到 ", posCount, " 个 AlphaOS 持仓");
    
    return result;
}

//+------------------------------------------------------------------+
//| 解析时间周期字符串为 ENUM_TIMEFRAMES                              |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES ParseTimeframe(const string tf)
{
    if(tf == "M1")  return PERIOD_M1;
    if(tf == "M5")  return PERIOD_M5;
    if(tf == "M15") return PERIOD_M15;
    if(tf == "M30") return PERIOD_M30;
    if(tf == "H1")  return PERIOD_H1;
    if(tf == "H4")  return PERIOD_H4;
    if(tf == "D1")  return PERIOD_D1;
    if(tf == "W1")  return PERIOD_W1;
    if(tf == "MN1") return PERIOD_MN1;
    return PERIOD_M5;  // 默认 M5
}

//+------------------------------------------------------------------+
//| v4.0: 处理回放 tick 发送（非阻塞 chunk 推送）                       |
//| 在 OnTimer() 中调用，每 100ms 发送一批 ticks                       |
//+------------------------------------------------------------------+
void ProcessReplayTicks()
{
    if(!g_replayActive) return;
    if(g_replayIdx >= g_replayTotal)
    {
        // 回放完成
        Print("✅ Tick replay completed: sent=", g_replaySent, " total=", g_replayTotal);
        g_replayActive = false;
        ArrayFree(g_replayTicks);
        return;
    }
    
    // 计算本次发送数量：pace_tps * timer_interval / 1000
    // OnTimer 每 100ms 调用一次，所以每次发送 pace_tps / 10 个 ticks
    int chunkSize = g_replayPaceTps / 10;
    if(chunkSize < 100) chunkSize = 100;  // 最小 100
    if(chunkSize > 10000) chunkSize = 10000;  // 最大 10000
    
    int endIdx = MathMin(g_replayIdx + chunkSize, g_replayTotal);
    
    // 发送 chunk
    for(int i = g_replayIdx; i < endIdx; i++)
    {
        SendBinaryTick(g_replayTicks[i]);
        g_replaySent++;
    }
    
    g_replayIdx = endIdx;
    
    // 进度日志（每 50000 个 tick 打印一次）
    if(g_replaySent % 50000 == 0 || g_replayIdx >= g_replayTotal)
    {
        Print("📡 Replay progress: ", g_replaySent, "/", g_replayTotal, 
              " (", DoubleToString(100.0 * g_replaySent / g_replayTotal, 1), "%)");
    }
}

//+------------------------------------------------------------------+
//| v4.0: 启动 tick 历史回放                                           |
//| 请求格式: START_REPLAY_TICKS|SYMBOL|WINDOW_SEC|END_EPS_MS|MAX_TICKS|PACE_TPS |
//| 返回: OK|REPLAY_STARTED|count=...|start=...|end=...                |
//+------------------------------------------------------------------+
string StartTickReplay(const string symbol, int windowSec, int endEpsMs, int maxTicks, int paceTps)
{
    if(g_replayActive)
    {
        return "ERROR|Replay already active. Stop first with STOP_REPLAY_TICKS";
    }
    
    // 计算时间范围
    long endTimeMs = (long)TimeCurrent() * 1000 - endEpsMs;
    long startTimeMs = endTimeMs - (long)windowSec * 1000;
    
    datetime endTime = (datetime)(endTimeMs / 1000);
    datetime startTime = (datetime)(startTimeMs / 1000);
    
    Print("📚 Starting tick replay: ", symbol, 
          " from ", TimeToString(startTime, TIME_DATE|TIME_SECONDS),
          " to ", TimeToString(endTime, TIME_DATE|TIME_SECONDS));
    
    // 获取历史 ticks
    ArrayFree(g_replayTicks);
    int copied = CopyTicksRange(symbol, g_replayTicks, COPY_TICKS_ALL, startTimeMs, endTimeMs);
    
    if(copied <= 0)
    {
        int lastError = GetLastError();
        return StringFormat("ERROR|CopyTicksRange failed. Error=%d, symbol=%s", lastError, symbol);
    }
    
    // 限制最大数量
    if(copied > maxTicks)
    {
        Print("⚠️ Tick count ", copied, " exceeds maxTicks ", maxTicks, ", truncating...");
        ArrayResize(g_replayTicks, maxTicks);
        copied = maxTicks;
    }
    
    // 初始化回放状态
    g_replayActive = true;
    g_replayIdx = 0;
    g_replayTotal = copied;
    g_replayPaceTps = paceTps > 0 ? paceTps : 50000;
    g_replaySent = 0;
    g_replayStartTime = TimeCurrent();
    g_replayTicksStart = startTime;
    g_replayTicksEnd = endTime;
    
    Print("✅ Tick replay started: count=", copied, 
          " pace=", g_replayPaceTps, " tps",
          " window=", windowSec, "s");
    
    return StringFormat("OK|REPLAY_STARTED|count=%d|start=%s|end=%s|pace=%d",
        copied,
        TimeToString(startTime, TIME_DATE|TIME_SECONDS),
        TimeToString(endTime, TIME_DATE|TIME_SECONDS),
        g_replayPaceTps);
}

//+------------------------------------------------------------------+
//| v4.0: 停止 tick 历史回放                                           |
//| 返回: OK|REPLAY_STOPPED|sent=...|remaining=...                     |
//+------------------------------------------------------------------+
string StopTickReplay()
{
    if(!g_replayActive)
    {
        return "OK|REPLAY_NOT_ACTIVE|sent=0|remaining=0";
    }
    
    int remaining = g_replayTotal - g_replayIdx;
    
    Print("🛑 Stopping tick replay: sent=", g_replaySent, " remaining=", remaining);
    
    g_replayActive = false;
    ArrayFree(g_replayTicks);
    
    return StringFormat("OK|REPLAY_STOPPED|sent=%d|remaining=%d", g_replaySent, remaining);
}

//+------------------------------------------------------------------+
//| v4.0: 查询回放状态                                                 |
//| 返回: OK|REPLAY_STATUS|active=...|sent=...|total=...|progress=...  |
//+------------------------------------------------------------------+
string GetReplayStatus()
{
    if(!g_replayActive)
    {
        return "OK|REPLAY_STATUS|active=false|sent=0|total=0|progress=0";
    }
    
    double progress = g_replayTotal > 0 ? 100.0 * g_replaySent / g_replayTotal : 0.0;
    
    return StringFormat("OK|REPLAY_STATUS|active=true|sent=%d|total=%d|progress=%.1f",
        g_replaySent, g_replayTotal, progress);
}

//+------------------------------------------------------------------+
//| 发送历史数据响应 (REP 模式辅助函数)                               |
//| REP 模式自动处理路由，只需发送数据                                |
//+------------------------------------------------------------------+
void SendHistoryResponse(const string &response)
{
    bool sendOk = g_socket_history.send(response);
    
    if(!sendOk)
    {
        Print("❌ send 失败! 数据长度=", StringLen(response));
    }
    else if(InpVerboseLog)
    {
        Print("✅ send 成功: 数据长度=", StringLen(response));
    }
}

//+------------------------------------------------------------------+
//| 处理历史数据请求 (REP 模式)                                       |
//| 请求格式: GET_HISTORY|SYMBOL|TIMEFRAME|START_DATE|END_DATE        |
//| 响应格式: 简化单次响应 (适合中小数据量)                           |
//| REP 模式自动处理 identity 路由，无需手动管理                      |
//+------------------------------------------------------------------+
void HandleHistoryRequest()
{
    if(CheckPointer(g_socket_history) == POINTER_INVALID) 
    {
        static datetime lastWarn = 0;
        if(TimeCurrent() - lastWarn > 60)
        {
            Print("⚠️ g_socket_history 指针无效");
            lastWarn = TimeCurrent();
        }
        return;
    }
    
    // 调试：每 10 秒打印一次状态
    static datetime lastDebug = 0;
    static long recvAttempts = 0;
    static long recvSuccess = 0;
    recvAttempts++;
    
    // REP 模式: 使用非阻塞 recv 接收请求
    // 必须使用 ZMQ_FLAG_DONTWAIT，否则会阻塞整个 EA
    string request = "";
    bool recvOk = g_socket_history.recv(request, ZMQ_FLAG_DONTWAIT);
    
    if(InpVerboseLog && TimeCurrent() - lastDebug > 10)
    {
        Print("📡 历史数据服务 (REP): 尝试=", recvAttempts, " 成功=", recvSuccess);
        lastDebug = TimeCurrent();
    }
    
    if(!recvOk) return;
    
    recvSuccess++;
    if(InpVerboseLog) Print("📥 recv 成功! 请求长度=", StringLen(request));
    
    // 记录请求
    if(InpVerboseLog) Print("📚 历史数据请求: ", StringSubstr(request, 0, 80));
    
    // 解析请求: GET_HISTORY|SYMBOL|TIMEFRAME|START_DATE|END_DATE
    string reqParts[];
    int partCount = StringSplit(request, '|', reqParts);
    
    if(partCount < 5 || reqParts[0] != "GET_HISTORY")
    {
        // 检查是否是 PING 请求
        if(request == "PING")
        {
            SendHistoryResponse("PONG");
            if(InpVerboseLog) Print("📡 PING -> PONG");
            return;
        }
        
        // 检查是否是 GET_SYMBOL_INFO 请求
        if(partCount >= 2 && reqParts[0] == "GET_SYMBOL_INFO")
        {
            string sym = reqParts[1];
            string infoJson = GetSymbolInfoJson(sym);
            SendHistoryResponse(infoJson);
            return;
        }
        
        // v4.0: 检查是否是 START_REPLAY_TICKS 请求
        // 格式: START_REPLAY_TICKS|SYMBOL|WINDOW_SEC|END_EPS_MS|MAX_TICKS|PACE_TPS
        if(partCount >= 6 && reqParts[0] == "START_REPLAY_TICKS")
        {
            string sym = reqParts[1];
            int windowSec = (int)StringToInteger(reqParts[2]);
            int endEpsMs = (int)StringToInteger(reqParts[3]);
            int maxTicks = (int)StringToInteger(reqParts[4]);
            int paceTps = (int)StringToInteger(reqParts[5]);
            
            string response = StartTickReplay(sym, windowSec, endEpsMs, maxTicks, paceTps);
            SendHistoryResponse(response);
            return;
        }
        
        // v4.0: 检查是否是 STOP_REPLAY_TICKS 请求
        if(reqParts[0] == "STOP_REPLAY_TICKS")
        {
            string response = StopTickReplay();
            SendHistoryResponse(response);
            return;
        }
        
        // v4.0: 检查是否是 GET_REPLAY_STATUS 请求
        if(reqParts[0] == "GET_REPLAY_STATUS")
        {
            string response = GetReplayStatus();
            SendHistoryResponse(response);
            return;
        }
        
        string errorResp = "ERROR|Invalid request format. Use: GET_HISTORY|SYMBOL|TIMEFRAME|START_DATE|END_DATE or START_REPLAY_TICKS|...";
        SendHistoryResponse(errorResp);
        return;
    }
    
    string symbol = reqParts[1];
    string tfStr = reqParts[2];
    string startStr = reqParts[3];
    string endStr = reqParts[4];
    
    // 解析时间周期
    ENUM_TIMEFRAMES timeframe = ParseTimeframe(tfStr);
    
    // 解析日期 (格式: YYYY-MM-DD 或 YYYY.MM.DD HH:MM:SS)
    datetime startDate = StringToTime(startStr);
    datetime endDate = StringToTime(endStr);
    
    if(startDate == 0 || endDate == 0)
    {
        string errorResp = "ERROR|Invalid date format. Use: YYYY-MM-DD or YYYY.MM.DD HH:MM:SS";
        SendHistoryResponse(errorResp);
        Print("❌ 日期解析失败: ", startStr, " ~ ", endStr);
        return;
    }
    
    Print("📅 请求范围: ", TimeToString(startDate), " ~ ", TimeToString(endDate));
    
    // 获取历史数据
    MqlRates rates[];
    ArraySetAsSeries(rates, false);  // 正序（从旧到新）
    
    int copied = CopyRates(symbol, timeframe, startDate, endDate, rates);
    
    if(copied <= 0)
    {
        int lastError = GetLastError();
        string errorResp = StringFormat("ERROR|Failed to get rates for %s. Error: %d", symbol, lastError);
        SendHistoryResponse(errorResp);
        Print("❌ CopyRates 失败: Error=", lastError);
        return;
    }
    
    Print("📊 获取 ", copied, " 条 ", symbol, " ", tfStr, " 数据");
    
    // 简化协议：直接返回 CSV 数据（单次响应，适合 < 10000 条）
    // 格式: CSV|COUNT|time,open,high,low,close,volume\ndata1\ndata2...
    
    // 之前: string csvData = "";
    int maxRows = MathMin(copied, 10000);  // Restored: 限制最大行数
    
    // 优化: 预分配内存 (每行约 70 字符)
    string csvData = "";
    if(!StringReserve(csvData, maxRows * 70))
    {
        Print("❌ 内存分配失败");
        return;
    }
    
    for(int i = 0; i < maxRows; i++)
    {
        string line = StringFormat("%s,%.5f,%.5f,%.5f,%.5f,%I64d,%d,%I64d",
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            rates[i].open,
            rates[i].high,
            rates[i].low,
            rates[i].close,
            rates[i].tick_volume,
            rates[i].spread,
            rates[i].real_volume
        );
        
        if(i > 0) StringAdd(csvData, "\n");
        StringAdd(csvData, line);
    }
    
    // 构建响应: CSV|COUNT|COLUMNS|DATA
    string header = StringFormat("CSV|%d|time,open,high,low,close,tick_volume,spread,real_volume|", maxRows);
    
    // 组合最终字符串 (header + csvData)
    // 注意: 这里创建一个巨大的字符串
    string fullResponse = header + csvData;
    
    Print("📦 准备发送响应: ", StringLen(fullResponse), " bytes");
    
    // 性能优化: 转换为 uchar[] 发送，避免库内部的隐式转换开销
    uchar data[];
    StringToCharArray(fullResponse, data);
    
    // StringToCharArray 会包含结尾的 \0，ZeroMQ 不需要发送它
    // 调整数组大小去掉最后一个字节
    if(ArraySize(data) > 0) ArrayResize(data, ArraySize(data) - 1);
    
    bool sendOk = g_socket_history.send(data);
    
    if(sendOk)
    {
        Print("✅ 历史数据发送成功: ", ArraySize(data), " bytes");
    }
    else
    {
        Print("❌ 历史数据发送失败!");
    }
}

//+------------------------------------------------------------------+
//| 获取品种信息 JSON                                                 |
//+------------------------------------------------------------------+
string GetSymbolInfoJson(const string symbol)
{
    if(!SymbolSelect(symbol, true))
    {
        return "{\"error\":\"Symbol not found\"}";
    }
    
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double contractSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    double volumeMin = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double volumeMax = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double volumeStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
    
    return StringFormat(
        "{\"symbol\":\"%s\",\"point\":%.10f,\"digits\":%d,\"tick_size\":%.10f,\"tick_value\":%.5f,"
        "\"contract_size\":%.2f,\"volume_min\":%.2f,\"volume_max\":%.2f,\"volume_step\":%.2f,\"spread\":%d}",
        symbol, point, digits, tickSize, tickValue, contractSize, volumeMin, volumeMax, volumeStep, spread
    );
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=================================================");
    Print("AlphaOS Executor v3.00 初始化中...");
    Print("MQL5-ZeroMQ v3.0 API");
    Print("=================================================");
   
    // 初始化日开始状态
    InitDayStart();
    
    //----------------------------------------------------------------
    // 1. 创建 ZmqContext (v3.0 API: 使用指针)
    //----------------------------------------------------------------
    g_context = new ZmqContext();
    if(CheckPointer(g_context) == POINTER_INVALID)
    {
        Print("❌ 错误: 无法创建 ZmqContext");
        return INIT_FAILED;
    }
    
    //----------------------------------------------------------------
    // 2. 创建 PUB 套接字 (行情发布)
    //    v3.0 API: ZmqSocket(context.ref(), ZMQ_SOCKET_PUB)
    //----------------------------------------------------------------
    g_socket_pub = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_PUB);
    if(CheckPointer(g_socket_pub) == POINTER_INVALID)
    {
        Print("❌ 错误: 无法创建 PUB 套接字");
        CleanupZmq();
        return INIT_FAILED;
    }
    
    // v3.0 API: 使用默认选项，HWM 和 LINGER 由库处理
    
    if(!g_socket_pub.bind(InpBindAddressPub))
    {
        Print("❌ 错误: 无法绑定 PUB 套接字到 ", InpBindAddressPub);
        CleanupZmq();
        return INIT_FAILED;
   }
   
    //----------------------------------------------------------------
    // 4. 创建 ROUTER 套接字 (订单执行)
    //    v3.0 API: ZMQ_SOCKET_ROUTER
    //----------------------------------------------------------------
    g_socket_rep = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_ROUTER);
    if(CheckPointer(g_socket_rep) == POINTER_INVALID)
    {
        Print("❌ 错误: 无法创建 ROUTER 套接字");
        CleanupZmq();
        return INIT_FAILED;
    }
    
    if(!g_socket_rep.bind(InpBindAddressRep))
    {
        Print("❌ 错误: 无法绑定 ROUTER 套接字到 ", InpBindAddressRep);
        CleanupZmq();
        return INIT_FAILED;
    }
    
    //----------------------------------------------------------------
    // 5. 创建 PAIR 套接字 (心跳监控)
    //    v3.0 API: ZMQ_SOCKET_PAIR
    //----------------------------------------------------------------
    g_socket_heartbeat = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_PAIR);
    if(CheckPointer(g_socket_heartbeat) == POINTER_INVALID)
    {
        Print("❌ 错误: 无法创建 PAIR 套接字");
        CleanupZmq();
        return INIT_FAILED;
    }
    
    // v3.0 API: 使用默认选项
    
    if(!g_socket_heartbeat.bind(InpHeartbeatAddr))
    {
        Print("❌ 错误: 无法绑定心跳套接字到 ", InpHeartbeatAddr);
        CleanupZmq();
        return INIT_FAILED;
    }
    
    //----------------------------------------------------------------
    // 6. 创建 REP 套接字 (历史数据请求)
    //    REQ-REP 模式更简单，自动处理 identity 路由
    //----------------------------------------------------------------
    g_socket_history = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_REP);
    if(CheckPointer(g_socket_history) == POINTER_INVALID)
    {
        Print("❌ 错误: 无法创建 REP 套接字 (历史数据)");
        CleanupZmq();
        return INIT_FAILED;
    }
    
    if(!g_socket_history.bind(InpHistoryAddr))
    {
        Print("❌ 错误: 无法绑定历史数据套接字到 ", InpHistoryAddr);
        CleanupZmq();
        return INIT_FAILED;
   }
   
    Print("✅ ZeroMQ v3.0 桥接就绪:");
    Print("   - 行情发布 (PUB):    ", InpBindAddressPub);
    Print("   - 订单执行 (ROUTER): ", InpBindAddressRep);
    Print("   - 心跳监控 (PAIR):   ", InpHeartbeatAddr);
    Print("   - 历史数据 (REP):    ", InpHistoryAddr);
    Print("   - Tick 格式: ", InpUseBinaryTick ? "Binary (36 bytes)" : "JSON");
    
    // 设置定时器（100ms 轮询，10ms 太快可能导致阻塞）
    EventSetMillisecondTimer(100);
    
    Print("=================================================");
    Print("AlphaOS Executor v3.00 初始化完成");
    Print("=================================================");
   
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| 清理 ZeroMQ 资源                                                  |
//+------------------------------------------------------------------+
void CleanupZmq()
{
    Print("🧹 开始清理 ZeroMQ 资源...");
    
    // 关闭并删除套接字 (设置 linger=0 确保立即关闭)
    if(CheckPointer(g_socket_pub) != POINTER_INVALID)
    {
        g_socket_pub.setLinger(0);
        delete g_socket_pub;
        g_socket_pub = NULL;
        Print("   - PUB 套接字已关闭");
    }
    
    if(CheckPointer(g_socket_rep) != POINTER_INVALID)
    {
        g_socket_rep.setLinger(0);
        delete g_socket_rep;
        g_socket_rep = NULL;
        Print("   - ROUTER (订单) 套接字已关闭");
    }
    
    if(CheckPointer(g_socket_heartbeat) != POINTER_INVALID)
    {
        g_socket_heartbeat.setLinger(0);
        delete g_socket_heartbeat;
        g_socket_heartbeat = NULL;
        Print("   - PAIR (心跳) 套接字已关闭");
    }
    
    if(CheckPointer(g_socket_history) != POINTER_INVALID)
    {
        g_socket_history.setLinger(0);
        delete g_socket_history;
        g_socket_history = NULL;
        Print("   - REP (历史) 套接字已关闭");
    }
    
    // 短暂延迟让 ZeroMQ 完成内部清理
    Sleep(100);
    
    // 删除上下文 (最后删除)
    if(CheckPointer(g_context) != POINTER_INVALID)
    {
        delete g_context;
        g_context = NULL;
        Print("   - 上下文已销毁");
    }
    
    Print("✅ ZeroMQ 资源清理完成");
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
    
    Print("=================================================");
    Print("AlphaOS Executor v3.00 关闭中...");
    Print("   - 当日 PnL: ", DoubleToString(g_dailyPnL, 2));
    Print("   - Tick 总数: ", g_tickCount);
    Print("   - 订单总数: ", g_orderCount);
    Print("   - 心跳序列: ", g_heartbeatSequence);
    
    // 清理 ZeroMQ 资源
    CleanupZmq();
    
    Print("=================================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // v4.0: 回放期间暂停发送 live ticks（避免 replay 与 live 交叠）
    if(g_replayActive)
    {
        return;
    }
    
    MqlTick lastTick;
    if(!SymbolInfoTick(Symbol(), lastTick)) return;
   
    // 发送 Tick 数据
    if(InpUseBinaryTick)
    {
        SendBinaryTick(lastTick);
    }
    else
    {
        SendJsonTick(lastTick);
    }
}

//+------------------------------------------------------------------+
//| Timer event function                                             |
//+------------------------------------------------------------------+
void OnTimer()
{
    // 调试：确认 OnTimer 是否活着（仅在 VerboseLog 模式下输出）
    static long timerCounter = 0;
    timerCounter++;
    if(InpVerboseLog && timerCounter % 100 == 0) 
        Print("💓 EA OnTimer ALIVE: ", timerCounter);
   
    //----------------------------------------------------------------
    // 1. 处理 ZAP 后台任务 (v3.0 CRITICAL: 必须调用!)
    //    已禁用 CurveZMQ，暂时注释以防阻塞
    //----------------------------------------------------------------
    /*
    if(CheckPointer(g_context) != POINTER_INVALID)
    {
        g_context.ProcessAuthTasks();
    }
    */
    
    //----------------------------------------------------------------
    // 2. 处理心跳
    //----------------------------------------------------------------
    HandleHeartbeat();
    
    //----------------------------------------------------------------
    // 2.5 v4.0: 处理 tick 历史回放（非阻塞 chunk 推送）
    //----------------------------------------------------------------
    ProcessReplayTicks();
    
    //----------------------------------------------------------------
    // 3. 处理历史数据请求 (REP 模式)
    //----------------------------------------------------------------
    HandleHistoryRequest();
    
    //----------------------------------------------------------------
    // 4. 处理订单请求 (ROUTER 模式)
    //----------------------------------------------------------------
    if(CheckPointer(g_socket_rep) == POINTER_INVALID) return;
    
    // v3.0 API: 使用 recvMultipart 接收多帧消息
    string parts[];
    
    // 使用非阻塞模式，防止阻塞 OnTimer
    if(g_socket_rep.recvMultipart(parts, ZMQ_FLAG_DONTWAIT))
    {
        int numParts = ArraySize(parts);
        
        // 调试日志：显示接收到的帧数和内容
        if(InpVerboseLog)
        {
            Print("📨 收到多帧消息: ", numParts, " 帧");
            for(int p = 0; p < numParts && p < 5; p++)
            {
                string preview = StringLen(parts[p]) > 80 ? StringSubstr(parts[p], 0, 80) + "..." : parts[p];
                Print("   帧[", p, "]: ", StringLen(parts[p]), " bytes = ", preview);
            }
        }
        
        // 提取订单 JSON 和身份帧
        // ROUTER 收到的消息格式: [identity, "", json] 或 [identity, json]
        string identity = "";
        string orderJson = "";
        
        // ROUTER 模式：第一帧始终是 identity（即使看起来为空也要保留）
        if(numParts >= 2)
        {
            identity = parts[0];  // 始终保存第一帧作为 identity
        }
        
        // 遍历所有帧，找到包含 JSON 的那一帧
        for(int i = 0; i < numParts; i++)
        {
            // 检查是否是 JSON（以 { 开头）
            if(StringLen(parts[i]) > 2 && StringGetCharacter(parts[i], 0) == '{')
            {
                orderJson = parts[i];
                break;
            }
        }
        
        if(InpVerboseLog)
        {
            Print("📨 Identity 长度: ", StringLen(identity), " JSON 长度: ", StringLen(orderJson));
        }
        
        if(StringLen(orderJson) == 0)
        {
            Print("⚠️ 未找到有效的 JSON 数据，帧数: ", numParts);
            for(int p = 0; p < numParts; p++)
            {
                Print("   帧[", p, "]: len=", StringLen(parts[p]), " first_char=", StringGetCharacter(parts[p], 0));
            }
        }
        
        if(StringLen(orderJson) > 0)
        {
            string responseJson;
            
            // 解析 action 字段
            string action = JsonGetString(orderJson, "action");
            
            // 检查是否是状态查询
            if(action == "STATUS")
            {
                responseJson = GetStatusJson();
            }
            // 检查是否是持仓查询
            else if(action == "GET_POSITIONS")
            {
                string filterSymbol = JsonGetString(orderJson, "symbol");
                responseJson = GetPositionsJson(filterSymbol);
                Print("📊 持仓查询结果: ", StringSubstr(responseJson, 0, 200));
            }
            else
            {
                // 执行订单
                responseJson = ExecuteOrder(orderJson);
            }
            
            // v3.0 API: 使用 sendMultipart 发送多帧响应
            // ROUTER 模式：始终发送三帧 [identity, "", response]
            // 即使 identity 看起来为空，也必须发送以正确路由回 DEALER
            string reply[];
            ArrayResize(reply, 3);
            reply[0] = identity;      // 身份帧（必须保留原始值）
            reply[1] = "";            // 空分隔符
            reply[2] = responseJson;
            
            bool sendOk = g_socket_rep.sendMultipart(reply);
            if(InpVerboseLog || !sendOk)
            {
                Print("📤 响应发送: ", sendOk ? "成功" : "失败", 
                      " Identity=", StringLen(identity), "bytes",
                      " Response=", StringLen(responseJson), "bytes");
            }
        }
    }
    
    //----------------------------------------------------------------
    // 5. 每日重置检查
    //----------------------------------------------------------------
    static datetime lastDate = 0;
    datetime currentDate = TimeCurrent() - TimeCurrent() % 86400;
    
    if(currentDate != lastDate)
    {
        lastDate = currentDate;
        InitDayStart();
        Print("📅 新交易日开始，状态已重置");
    }
               }
               
//+------------------------------------------------------------------+
//| Trade transaction function                                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(
    const MqlTradeTransaction &trans,
    const MqlTradeRequest &request,
    const MqlTradeResult &result
)
{
    // 监控交易事件
    if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
    {
        if(InpVerboseLog)
        {
            Print("📝 交易成交: Deal=", trans.deal, " Order=", trans.order, " Type=", EnumToString(trans.deal_type));
           }
       }
   }
