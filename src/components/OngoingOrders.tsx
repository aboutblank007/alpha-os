import { Trade } from '@/lib/supabase';
import { ArrowUpRight, ArrowDownRight, Clock, WifiOff } from 'lucide-react';
import { useEffect, useState, useCallback } from 'react';

interface OngoingOrdersProps {
    orders: Trade[];
}

// 定义合约规格用于盈亏计算
const CONTRACT_SPECS: { [key: string]: { pipValueMultiplier: number } } = {
    'JPY': { pipValueMultiplier: 1000 },
    'XAU': { pipValueMultiplier: 100 },
    'DEFAULT_FX': { pipValueMultiplier: 100000 },
};

export function OngoingOrders({ orders }: OngoingOrdersProps) {
    // 存储实时价格数据 { symbol: price }
    const [marketData, setMarketData] = useState<Record<string, number>>({});
    const [isConnected, setIsConnected] = useState(true);
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

    // 获取实时价格的函数
    const fetchPrices = useCallback(async () => {
        if (orders.length === 0) return;

        // 获取所有唯一的交易品种
        const symbols = Array.from(new Set(orders.map(o => o.symbol)));

        try {
            const response = await fetch(`/api/prices?symbols=${symbols.join(',')}`);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: '未知错误' }));
                console.error(`
╔═══════════════════════════════════════════════════════════════╗
║         ❌ 获取价格失败 (HTTP ${response.status})              ║
╚═══════════════════════════════════════════════════════════════╝

请求的品种: ${symbols.join(', ')}

错误详情:
${JSON.stringify(errorData, null, 2)}

💡 可能的原因:
  1. OANDA API 配置错误或未配置
  2. OANDA API 密钥已过期
  3. 请求的交易品种格式不正确
  4. OANDA API 服务暂时不可用

📝 解决方法:
  1. 检查 .env.local 中的 OANDA_API_KEY 和 OANDA_ACCOUNT_ID
  2. 访问 http://localhost:3000/api/test-env 验证配置
  3. 手动测试: http://localhost:3000/api/prices?symbols=${symbols[0]}
                `);
                throw new Error(`HTTP ${response.status}: ${errorData.error || '获取价格失败'}`);
            }

            const data = await response.json();
            
            if (data.prices) {
                setMarketData(data.prices);
                setIsConnected(data.source === 'oanda'); // 只有真实数据才算连接成功
                setLastUpdate(new Date());
                
                // 记录数据源
                if (data.source === 'mock') {
                    console.warn('⚠️ 使用模拟价格数据（OANDA API 未配置）');
        }
        }
        } catch (error: any) {
            console.error('获取实时价格失败:', error.message || error);
            setIsConnected(false);
        }
    }, [orders]);

    // 初始加载价格
    useEffect(() => {
        fetchPrices();
    }, [fetchPrices]);

    // 定期更新价格（每2秒）
    useEffect(() => {
        const interval = setInterval(() => {
            fetchPrices();
        }, 2000);

        return () => clearInterval(interval);
    }, [fetchPrices]);

    return (
        <div className="glass-panel rounded-xl p-6 overflow-hidden flex flex-col h-full">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Clock size={18} className="text-accent-primary" />
                    持仓订单
                </h3>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                        {isConnected ? (
                            <>
                                <div className="w-2 h-2 rounded-full bg-accent-success animate-pulse"></div>
                                <span className="text-xs text-slate-400">
                                    {lastUpdate.toLocaleTimeString('zh-CN', { 
                                        hour: '2-digit', 
                                        minute: '2-digit',
                                        second: '2-digit',
                                        hour12: false 
                                    })}
                                </span>
                            </>
                        ) : (
                            <>
                                <WifiOff size={14} className="text-accent-danger" />
                                <span className="text-xs text-accent-danger">连接断开</span>
                            </>
                        )}
                    </div>
                <span className="text-xs font-medium px-2 py-1 rounded bg-white/5 text-slate-400">
                    {orders.length} 活跃
                </span>
                </div>
            </div>

            <div className="overflow-x-auto flex-1 custom-scrollbar">
                <table className="w-full text-left text-sm">
                    <thead>
                        <tr className="text-xs uppercase text-slate-500 font-medium border-b border-white/5">
                            <th className="px-4 py-3 font-medium">时间</th>
                            <th className="px-4 py-3 font-medium">品种</th>
                            <th className="px-4 py-3 font-medium">方向</th>
                            <th className="px-4 py-3 font-medium">开仓价</th>
                            <th className="px-4 py-3 font-medium text-right">浮动盈亏</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {orders.map((order) => {
                            // 获取当前市场价格
                            const currentPrice = marketData[order.symbol] ?? order.entry_price;
                            const diff = currentPrice - order.entry_price;
                            
                            // 根据品种计算盈亏
                            let pnl = 0;
                            const symbolUpper = order.symbol.toUpperCase();

                            if (symbolUpper.includes('JPY')) {
                                // JPY 货币对
                                pnl = diff * order.quantity * CONTRACT_SPECS.JPY.pipValueMultiplier;
                            } else if (symbolUpper.includes('XAU')) {
                                // 黄金
                                pnl = diff * order.quantity * CONTRACT_SPECS.XAU.pipValueMultiplier;
                            } else {
                                // 其他外汇对
                                pnl = diff * order.quantity * CONTRACT_SPECS.DEFAULT_FX.pipValueMultiplier;
                            }
                            
                            // 卖单需要反转盈亏
                            if (order.side === 'sell') pnl = -pnl;

                            return (
                                <tr
                                    key={order.id}
                                    className="group hover:bg-white/[0.02] transition-colors"
                                >
                                    <td className="px-4 py-3 text-slate-400 font-mono text-xs">
                                        {new Date(order.created_at).toLocaleTimeString('zh-CN', {
                                            hour: '2-digit',
                                            minute: '2-digit',
                                            hour12: false
                                        })}
                                    </td>
                                    <td className="px-4 py-3 font-medium text-slate-200 group-hover:text-white transition-colors">
                                        {order.symbol}
                                    </td>
                                    <td className="px-4 py-3">
                                        <span
                                            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${order.side === 'buy'
                                                ? 'bg-accent-success/10 text-accent-success'
                                                : 'bg-accent-danger/10 text-accent-danger'
                                                }`}
                                        >
                                            {order.side === 'buy' ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                                            {order.side === 'buy' ? '买入' : '卖出'}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-slate-400 font-mono text-xs">{order.entry_price}</td>
                                    <td className="px-4 py-3 text-right">
                                        <div className="flex flex-col items-end gap-0.5">
                                            <span className={`text-xs font-mono font-bold ${pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                                            </span>
                                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-medium bg-accent-primary/10 text-accent-primary uppercase tracking-wide animate-pulse">
                                                实时
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                {orders.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-40 text-slate-500">
                        <p className="text-sm font-medium">暂无活跃订单</p>
                    </div>
                )}
            </div>
        </div>
    );
}
