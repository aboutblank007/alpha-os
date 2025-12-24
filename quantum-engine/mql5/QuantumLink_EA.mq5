//+------------------------------------------------------------------+
//| QuantumLink_EA.mq5                                               |
//| Q-Link 量子交易系统执行端                                        |
//| 基于 ding9736/MQL5-ZeroMQ 库                                     |
//| Copyright 2025, AlphaOS                                          |
//+------------------------------------------------------------------+
#property copyright "AlphaOS"
#property version   "2.00"
#property strict
#property description "Q-Link 量子-经典双优侧车架构执行端"
#property description "使用 MQL5-ZeroMQ 高性能消息库"

//--- 引入 ZeroMQ 库
#include <ZeroMQ/ZeroMQ.mqh>

//--- 输入参数
input group "ZeroMQ Settings"
input string InpBindAddress   = "tcp://0.0.0.0";     // 本地绑定地址 (0.0.0.0=所有网卡)
input string InpPythonHost    = "tcp://192.168.3.10"; // Python 主机地址 (Mac IP)
input int    InpMarketPort    = 5557;                // Market Stream 端口
input int    InpCommandPort   = 5558;                // Command Bus 端口
input int    InpStatePort     = 5559;                // State Sync 端口

input group "Trading Settings"
input int    InpMagicNumber   = 999001;              // 魔术数
input int    InpTimerMs       = 20;                  // 轮询间隔(ms)
input int    InpHeartbeatSec  = 1;                   // 心跳间隔(秒)
input int    InpDeadManSec    = 5;                   // 死人开关超时(秒)
input int    InpVolMaPeriod   = 20;                  // 成交量均线周期

//--- ZMQ 全局对象
ZmqContext *g_context = NULL;
ZmqSocket  *g_pushSocket = NULL;   // Market Stream (PUSH) - 本地 bind
ZmqSocket  *g_pullSocket = NULL;   // Command Bus (PULL) - 连接 Python
ZmqSocket  *g_repSocket = NULL;    // State Sync (REP) - 本地 bind

//--- 状态变量
int g_hRsi = INVALID_HANDLE;
int g_hEmaFast = INVALID_HANDLE;
int g_hEmaSlow = INVALID_HANDLE;

datetime g_lastBarTime = 0;
datetime g_lastHeartbeat = 0;
datetime g_lastPythonHB = 0;
int g_tickCount = 0;
bool g_safeMode = false;
bool g_initialized = false;

//--- 多品种支持
string g_symbols[];
int g_symbolCount = 0;
datetime g_lastBarTimes[];  // 每个品种的最后 K 线时间

//--- 函数原型
ENUM_ORDER_TYPE_FILLING GetFillingMode(string symbol);
string                  GetRetcodeDescription(uint retcode);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("====== QuantumLink EA v2.2 初始化 (单 EA 模式) ======");
    
    //--- 初始化多品种列表 (根据需求调整，这里恢复为 XAUUSD)
    ArrayResize(g_symbols, 1);
    g_symbols[0] = _Symbol; // 默认使用当前图表品种
    g_symbolCount = ArraySize(g_symbols);
    ArrayResize(g_lastBarTimes, g_symbolCount);
    for(int i = 0; i < g_symbolCount; i++) {
        g_lastBarTimes[i] = 0;
    }
    Print("📊 监控品种: ", g_symbols[0]);
    
    //--- 创建 ZMQ Context
    g_context = new ZmqContext();
    if(!g_context.isValid()) {
        Print("错误: 无法创建 ZMQ Context");
        return INIT_FAILED;
    }
    Print("✅ ZMQ Context 创建成功");
    
    //--- 创建 Market Stream Socket (PUSH) - 本地 bind
    string marketAddr = StringFormat("%s:%d", InpBindAddress, InpMarketPort);
    g_pushSocket = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_PUSH);
    g_pushSocket.setSendHighWaterMark(1000); // 设置 SNDHWM 防止背压
    if(!g_pushSocket.bind(marketAddr)) {
        Print("错误: 无法绑定 Market Stream: ", marketAddr);
        return INIT_FAILED;
    }
    Print("✅ Market Stream 已绑定: ", marketAddr);
    
    //--- 初始化指标句柄 (以当前品种为例)
    g_hRsi = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    g_hEmaFast = iMA(_Symbol, PERIOD_M1, 14, 0, MODE_EMA, PRICE_CLOSE);
    g_hEmaSlow = iMA(_Symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);
    
    if(g_hRsi == INVALID_HANDLE || g_hEmaFast == INVALID_HANDLE || g_hEmaSlow == INVALID_HANDLE) {
        Print("错误: 无法创建指标句柄");
        return INIT_FAILED;
    }
    
    //--- 创建 Command Bus Socket (PULL) - 连接 Python (connect)
    string commandAddr = StringFormat("%s:%d", InpPythonHost, InpCommandPort);
    g_pullSocket = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_PULL);
    g_pullSocket.setReceiveTimeout(1);  // 1ms 超时，非阻塞
    if(!g_pullSocket.connect(commandAddr)) {
        Print("错误: 无法连接 Command Bus: ", commandAddr);
        return INIT_FAILED;
    }
    Print("✅ Command Bus 已连接: ", commandAddr);
    
    //--- 创建 State Sync Socket (REP) - 本地 bind
    string stateAddr = StringFormat("%s:%d", InpBindAddress, InpStatePort);
    g_repSocket = new ZmqSocket(g_context.ref(), ZMQ_SOCKET_REP);
    g_repSocket.setReceiveTimeout(1);  // 1ms 超时，非阻塞
    if(!g_repSocket.bind(stateAddr)) {
        Print("错误: 无法绑定 State Sync: ", stateAddr);
        return INIT_FAILED;
    }
    Print("✅ State Sync 已绑定: ", stateAddr);
    
    //--- 启动定时器
    EventSetMillisecondTimer(InpTimerMs);
    
    g_initialized = true;
    Print("====== QuantumLink EA v2.2 初始化完成 ======");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("QuantumLink EA 停止中...");
    
    EventKillTimer();
    
    //--- 清理 ZMQ 资源
    if(g_pushSocket != NULL) {
        delete g_pushSocket;
        g_pushSocket = NULL;
    }
    if(g_pullSocket != NULL) {
        delete g_pullSocket;
        g_pullSocket = NULL;
    }
    if(g_repSocket != NULL) {
        delete g_repSocket;
        g_repSocket = NULL;
    }
    if(g_context != NULL) {
        delete g_context;
        g_context = NULL;
    }
    
    g_initialized = false;
    Print("✅ QuantumLink EA 已停止");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(!g_initialized) return;
    
    g_tickCount++;
    
    //--- 检测新K线
    datetime currentBarTime = iTime(_Symbol, _Period, 0);
    if(currentBarTime == g_lastBarTime) return;
    g_lastBarTime = currentBarTime;
    
    //--- 安全模式检查
    if(g_safeMode) {
        return;
    }
    
    //--- 计算链上特征
    double wick_ratio = CalculateWickRatio(1);
    double vol_density = CalculateVolumeDensity(1);
    double vol_shock = CalculateVolumeShock(1);
    int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    
    //--- 获取指标值
    double rsi[], emaF[], emaS[];
    double rsi_val = 0, emaF_val = 0, emaS_val = 0;
    
    if(CopyBuffer(g_hRsi, 0, 1, 1, rsi) > 0) rsi_val = rsi[0];
    if(CopyBuffer(g_hEmaFast, 0, 1, 1, emaF) > 0) emaF_val = emaF[0];
    if(CopyBuffer(g_hEmaSlow, 0, 1, 1, emaS) > 0) emaS_val = emaS[0];
    
    //--- 序列化为紧凑 CSV (Float64 适配: %.7f 精度)
    string packet = StringFormat("TICK,%I64d,%s,%.7f,%.7f,%d,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%d,%d,%.7f",
        (long)(TimeCurrent() * 1000),
        _Symbol,
        SymbolInfoDouble(_Symbol, SYMBOL_BID),
        SymbolInfoDouble(_Symbol, SYMBOL_ASK),
        (int)iVolume(_Symbol, _Period, 1),
        wick_ratio,
        vol_density,
        vol_shock,
        emaF_val,
        emaS_val,
        rsi_val,
        CalculateBidAskImbalance(1), // Dom Pressure Proxy (Vol Weighted)
        spread,
        g_tickCount,
        CalculateBidAskImbalance(1)
    );
    
    //--- 非阻塞发送
    if(g_pushSocket != NULL) {
        if(!g_pushSocket.send(packet)) {
            // 发送失败，静默忽略（可能队列满）
        }
    }
    
    g_tickCount = 0;
    
    //--- 优化：Tick驱动的即时订单轮询 (双重轮询)
    PollCommandBus();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(!g_initialized) return;
    
    //--- 1. 遍历所有品种发送 Tick 数据
    for(int i = 0; i < g_symbolCount; i++) {
        SendTickDataForSymbol(g_symbols[i], i);
    }
    
    //--- 2. 轮询订单指令
    PollCommandBus();
    
    //--- 3. 处理状态同步请求
    HandleStateSync();
    
    //--- 4. 发送心跳
    SendHeartbeat();
    
    //--- 5. 死人开关检查
    CheckDeadManSwitch();
}

//+------------------------------------------------------------------+
//| 轮询订单总线                                                     |
//+------------------------------------------------------------------+
void PollCommandBus()
{
    if(g_pullSocket == NULL) return;
    
    string msg;
    while(g_pullSocket.recv(msg)) {
        if(StringLen(msg) > 0) {
            ExecuteOrderFromJson(msg);
        }
    }
}

//+------------------------------------------------------------------+
//| 执行 JSON 订单                                                   |
//+------------------------------------------------------------------+
void ExecuteOrderFromJson(string json)
{
    string action = ExtractJsonString(json, "action");
    string symbol = ExtractJsonString(json, "symbol");
    string side = ExtractJsonString(json, "side");
    double lots = ExtractJsonDouble(json, "lots");
    double sl = ExtractJsonDouble(json, "sl");
    double tp = ExtractJsonDouble(json, "tp");
    int magic = (int)ExtractJsonDouble(json, "magic");
    string comment = ExtractJsonString(json, "comment");
    
    Print("📥 收到订单: ", action, " ", symbol, " ", side, " ", lots, " lots");
    
    if(action == "OPEN") {
        ENUM_ORDER_TYPE orderType = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
        double price = (side == "BUY") ? ask : bid;
        
        MqlTradeRequest request = {};
        MqlTradeResult result = {};
        
        request.action = TRADE_ACTION_DEAL;
        request.symbol = symbol;
        request.volume = lots;
        request.type = orderType;
        request.price = price;
        request.sl = sl;
        request.tp = tp;
        request.magic = (magic > 0) ? magic : InpMagicNumber;
        request.comment = comment;
        request.deviation = 20;
        request.type_filling = GetFillingMode(symbol);
        
        request.type_filling = GetFillingMode(symbol);
        
        // 使用 OrderSendAsync 减少执行延迟 (生产环境方案 3.4)
        if(OrderSendAsync(request, result)) {
            Print("✅ 异步订单发送成功: result=", result.retcode);
        } else {
            PrintFormat("❌ 异步订单发送失败: retcode=%d (%s)", 
                result.retcode, GetRetcodeDescription(result.retcode));
        }
    }
    else if(action == "MODIFY") {
        ulong ticket = (ulong)ExtractJsonDouble(json, "ticket");
        if(ticket > 0) {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            
            request.action = TRADE_ACTION_SLTP;
            request.position = ticket;
            request.symbol = symbol;
            request.sl = sl;
            request.tp = tp;
            request.magic = (magic > 0) ? magic : InpMagicNumber;
            
            if(OrderSend(request, result)) {
                Print("🔧 订单修改成功: ticket=", ticket, " sl=", sl, " tp=", tp);
            } else {
                Print("❌ 订单修改失败: ticket=", ticket, " retcode=", result.retcode);
            }
        }
    }
    else if(action == "CLOSE") {
        ClosePositionsByMagic(magic > 0 ? magic : InpMagicNumber);
    }
}

//+------------------------------------------------------------------+
//| 平仓指定魔术数的持仓                                             |
//+------------------------------------------------------------------+
void ClosePositionsByMagic(int magic)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetInteger(POSITION_MAGIC) == magic) {
                string symbol = PositionGetString(POSITION_SYMBOL);
                double volume = PositionGetDouble(POSITION_VOLUME);
                ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
                
                MqlTradeRequest request = {};
                MqlTradeResult result = {};
                
                request.action = TRADE_ACTION_DEAL;
                request.symbol = symbol;
                request.volume = volume;
                request.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
                request.price = (type == POSITION_TYPE_BUY) ? 
                    SymbolInfoDouble(symbol, SYMBOL_BID) : 
                    SymbolInfoDouble(symbol, SYMBOL_ASK);
                request.deviation = 10;
                request.position = ticket;
                
                OrderSendAsync(request, result);
                Print("📤 平仓请求: ticket=", ticket);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 发送指定品种的 Tick 数据                                         |
//+------------------------------------------------------------------+
void SendTickDataForSymbol(string symbol, int index)
{
    if(g_pushSocket == NULL) return;
    
    //--- 检测新 K 线
    datetime currentBarTime = iTime(symbol, PERIOD_M1, 0);
    if(currentBarTime == g_lastBarTimes[index]) return;
    g_lastBarTimes[index] = currentBarTime;
    
    //--- 获取 K 线数据
    MqlRates rates[];
    if(CopyRates(symbol, PERIOD_M1, 1, 1, rates) < 1) return;
    
    //--- 计算链上特征
    double wick_ratio = CalculateWickRatioForSymbol(symbol, 1);
    double vol_density = CalculateVolumeDensityForSymbol(symbol, 1);
    double vol_shock = CalculateVolumeShockForSymbol(symbol, 1);
    int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
    
    //--- 获取指标值
    int hRsi = iRSI(symbol, PERIOD_M1, 14, PRICE_CLOSE);
    int hEmaF = iMA(symbol, PERIOD_M1, 14, 0, MODE_EMA, PRICE_CLOSE);
    int hEmaS = iMA(symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);
    
    double rsi_val = 0, emaF_val = 0, emaS_val = 0;
    double buf[];
    if(CopyBuffer(hRsi, 0, 1, 1, buf) > 0) rsi_val = buf[0];
    if(CopyBuffer(hEmaF, 0, 1, 1, buf) > 0) emaF_val = buf[0];
    if(CopyBuffer(hEmaS, 0, 1, 1, buf) > 0) emaS_val = buf[0];
    
    IndicatorRelease(hRsi);
    IndicatorRelease(hEmaF);
    IndicatorRelease(hEmaS);
    
    //--- 序列化为紧凑 CSV (对齐 protocol.py)
    string packet = StringFormat("TICK,%I64d,%s,%.5f,%.5f,%d,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f,%d,%d,%.5f",
        (long)(TimeCurrent() * 1000),
        symbol,
        SymbolInfoDouble(symbol, SYMBOL_BID),
        SymbolInfoDouble(symbol, SYMBOL_ASK),
        (int)rates[0].tick_volume,
        wick_ratio,
        vol_density,
        vol_shock,
        emaF_val,
        emaS_val,
        rsi_val,
        CalculateBidAskImbalanceForSymbol(symbol, 1), // 用作为 dom_pressure_proxy
        spread,
        0,  // tick_count
        CalculateBidAskImbalanceForSymbol(symbol, 1)
    );
    
    //--- 非阻塞发送
    if(!g_pushSocket.send(packet)) {
        // 发送失败，静默忽略
    }
}

//+------------------------------------------------------------------+
//| 处理状态同步请求                                                 |
//+------------------------------------------------------------------+
void HandleStateSync()
{
    if(g_repSocket == NULL) return;
    
    string request;
    if(g_repSocket.recv(request)) {
        if(StringFind(request, "STATE_REQ") >= 0) {
            string response = BuildAccountStateJson();
            g_repSocket.send(response);
        }
    }
}

//+------------------------------------------------------------------+
//| 构建账户状态 JSON                                                |
//+------------------------------------------------------------------+
string BuildAccountStateJson()
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double margin_used = AccountInfoDouble(ACCOUNT_MARGIN);
    double margin_free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    
    string positions = "[";
    bool first = true;
    
    for(int i = 0; i < PositionsTotal(); i++) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        
        if(PositionSelectByTicket(ticket)) {
            if(!first) positions += ",";
            first = false;
            
            positions += StringFormat(
                "{\"ticket\":%I64u,\"symbol\":\"%s\",\"side\":\"%s\",\"lots\":%.2f,\"open_price\":%.5f,\"current_price\":%.5f,\"profit\":%.2f,\"magic\":%d}",
                PositionGetInteger(POSITION_TICKET),
                PositionGetString(POSITION_SYMBOL),
                (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL",
                PositionGetDouble(POSITION_VOLUME),
                PositionGetDouble(POSITION_PRICE_OPEN),
                PositionGetDouble(POSITION_PRICE_CURRENT),
                PositionGetDouble(POSITION_PROFIT),
                PositionGetInteger(POSITION_MAGIC)
            );
        }
    }
    positions += "]";
    
    return StringFormat(
        "{\"balance\":%.2f,\"equity\":%.2f,\"margin_used\":%.2f,\"margin_free\":%.2f,\"positions\":%s,\"timestamp\":%I64d}",
        balance, equity, margin_used, margin_free, positions, (long)(TimeCurrent() * 1000)
    );
}

//+------------------------------------------------------------------+
//| 发送心跳                                                         |
//+------------------------------------------------------------------+
void SendHeartbeat()
{
    if(TimeCurrent() - g_lastHeartbeat < InpHeartbeatSec) return;
    g_lastHeartbeat = TimeCurrent();
    
    if(g_pushSocket != NULL) {
        string hb = StringFormat("HB,MT5,%I64d,OK", (long)(TimeCurrent() * 1000));
        g_pushSocket.send(hb);
    }
}

//+------------------------------------------------------------------+
//| 死人开关检查                                                     |
//+------------------------------------------------------------------+
void CheckDeadManSwitch()
{
    if(g_safeMode) return;
    
    // TODO: 实现 Python 心跳检测
    // if(TimeCurrent() - g_lastPythonHB > InpDeadManSec) {
    //     Print("⚠️ Python 心跳超时，进入安全模式");
    //     g_safeMode = true;
    //     ClosePositionsByMagic(InpMagicNumber);
    // }
}

//+------------------------------------------------------------------+
//| 计算影线比率                                                     |
//+------------------------------------------------------------------+
double CalculateWickRatio(int shift)
{
    double high = iHigh(_Symbol, _Period, shift);
    double low = iLow(_Symbol, _Period, shift);
    double open = iOpen(_Symbol, _Period, shift);
    double close = iClose(_Symbol, _Period, shift);
    
    double range = high - low;
    if(range < _Point) return 0;
    
    double wick_upper = high - MathMax(open, close);
    double wick_lower = MathMin(open, close) - low;
    
    return (wick_upper + wick_lower) / range;
}

//+------------------------------------------------------------------+
//| 计算成交量密度                                                   |
//+------------------------------------------------------------------+
double CalculateVolumeDensity(int shift)
{
    double high = iHigh(_Symbol, _Period, shift);
    double low = iLow(_Symbol, _Period, shift);
    long volume = iVolume(_Symbol, _Period, shift);
    
    double range = high - low;
    if(range < _Point) return 0;
    
    return (double)volume / (range / _Point);
}

//+------------------------------------------------------------------+
//| 计算成交量冲击                                                   |
//+------------------------------------------------------------------+
double CalculateVolumeShock(int shift)
{
    double vol_sum = 0;
    for(int i = shift + 1; i <= shift + InpVolMaPeriod; i++) {
        vol_sum += (double)iVolume(_Symbol, _Period, i);
    }
    double vol_ma = vol_sum / InpVolMaPeriod;
    
    if(vol_ma <= 0) return 1.0;
    
    return (double)iVolume(_Symbol, _Period, shift) / vol_ma;
}

//+------------------------------------------------------------------+
//| 计算 Bid/Ask 不平衡度 (结合 Alpha101 能量修正)                   |
//+------------------------------------------------------------------+
double CalculateBidAskImbalance(int shift)
{
    double high = iHigh(_Symbol, _Period, shift);
    double low = iLow(_Symbol, _Period, shift);
    double close = iClose(_Symbol, _Period, shift);
    
    double range = high - low;
    if(range < _Point) return 0;
    
    // 基础位置因子 [-0.5, 0.5]
    double position_factor = ((close - low) / range) - 0.5;
    
    // 引入成交量冲击进行能量加权 [0.5, 3.0+]
    double vol_shock = CalculateVolumeShock(shift);
    
    // 最终压力值 = 位置 * 能量
    return position_factor * vol_shock;
}

//+------------------------------------------------------------------+
//| 多品种版本特征计算函数                                           |
//+------------------------------------------------------------------+
double CalculateWickRatioForSymbol(string symbol, int shift)
{
    double high = iHigh(symbol, PERIOD_M1, shift);
    double low = iLow(symbol, PERIOD_M1, shift);
    double open = iOpen(symbol, PERIOD_M1, shift);
    double close = iClose(symbol, PERIOD_M1, shift);
    
    double range = high - low;
    if(range < SymbolInfoDouble(symbol, SYMBOL_POINT)) return 0;
    
    double wick_upper = high - MathMax(open, close);
    double wick_lower = MathMin(open, close) - low;
    
    return (wick_upper + wick_lower) / range;
}

double CalculateVolumeDensityForSymbol(string symbol, int shift)
{
    double high = iHigh(symbol, PERIOD_M1, shift);
    double low = iLow(symbol, PERIOD_M1, shift);
    long volume = iVolume(symbol, PERIOD_M1, shift);
    
    double range = high - low;
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    if(range < point) return 0;
    
    return (double)volume / (range / point);
}

double CalculateVolumeShockForSymbol(string symbol, int shift)
{
    double vol_sum = 0;
    for(int i = shift + 1; i <= shift + InpVolMaPeriod; i++) {
        vol_sum += (double)iVolume(symbol, PERIOD_M1, i);
    }
    double vol_ma = vol_sum / InpVolMaPeriod;
    
    if(vol_ma <= 0) return 1.0;
    
    return (double)iVolume(symbol, PERIOD_M1, shift) / vol_ma;
}

double CalculateBidAskImbalanceForSymbol(string symbol, int shift)
{
    double high = iHigh(symbol, PERIOD_M1, shift);
    double low = iLow(symbol, PERIOD_M1, shift);
    double close = iClose(symbol, PERIOD_M1, shift);
    
    double range = high - low;
    if(range < SymbolInfoDouble(symbol, SYMBOL_POINT)) return 0;
    
    return ((close - low) / range) - 0.5;
}

//+------------------------------------------------------------------+
//| 从 JSON 提取字符串值                                             |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
    // 支持两种格式: "key":"value" 和 "key": "value"
    string search1 = "\"" + key + "\":\"";
    string search2 = "\"" + key + "\": \"";
    
    int start = StringFind(json, search1);
    int searchLen = StringLen(search1);
    
    if(start < 0) {
        start = StringFind(json, search2);
        searchLen = StringLen(search2);
    }
    
    if(start < 0) return "";
    
    start += searchLen;
    int end = StringFind(json, "\"", start);
    if(end < 0) return "";
    
    return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| 从 JSON 提取数值                                                 |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if(start < 0) return 0;
    
    start += StringLen(search);
    
    while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '"')) {
        start++;
    }
    
    int end = start;
    while(end < StringLen(json)) {
        ushort ch = StringGetCharacter(json, end);
        if(ch == ',' || ch == '}' || ch == '"' || ch == ' ') break;
        end++;
    }
    
    return StringToDouble(StringSubstr(json, start, end - start));
}
//+------------------------------------------------------------------+
//| 获取品种支持的填写模式                                           |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode(string symbol)
{
    uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
    if((filling & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
    if((filling & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
    return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| 获取返回码描述                                                   |
//+------------------------------------------------------------------+
string GetRetcodeDescription(uint retcode)
{
    switch(retcode)
    {
        case 10004: return "Requote";
        case 10006: return "Request rejected";
        case 10007: return "Request canceled by trader";
        case 10008: return "Order placed";
        case 10009: return "Request completed";
        case 10013: return "Invalid request";
        case 10014: return "Invalid volume";
        case 10015: return "Invalid price";
        case 10016: return "Invalid stops";
        case 10017: return "Trade is disabled";
        case 10018: return "Market is closed";
        case 10019: return "No enough money";
        case 10020: return "Prices changed";
        case 10021: return "No quotes to process the request";
        case 10024: return "Too frequent requests";
        case 10025: return "No changes in request";
        case 10026: return "Autotrading disabled by server";
        case 10027: return "Autotrading disabled by client terminal";
        case 10030: return "Unsupported filling mode";
        case 10031: return "No connection with the trade server";
        case 10033: return "Invalid order expiration date";
        case 10034: return "Order has been locked";
        case 10035: return "Modify denied";
        default: return "Unknown error";
    }
}
//+------------------------------------------------------------------+
