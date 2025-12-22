import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const limit = parseInt(searchParams.get("limit") || "100");
        const symbol = searchParams.get("symbol") || null;
        const action = searchParams.get("action") || null;
        
        const supabase = createClient(supabaseUrl, supabaseKey);
        
        let query = supabase
            .from("training_signals")
            .select("id, symbol, action, signal_price, ai_features, timestamp, result_profit")
            .order("timestamp", { ascending: false })
            .limit(limit);
        
        if (symbol) {
            query = query.eq("symbol", symbol);
        }
        
        if (action && action !== "ALL") {
            query = query.eq("action", action);
        }
        
        const { data, error } = await query;
        
        if (error) {
            console.error("Supabase error:", error);
            return NextResponse.json({ error: error.message }, { status: 500 });
        }
        
        // 格式化日志数据
        const logs = (data || []).map((item) => ({
            id: item.id,
            symbol: item.symbol,
            action: item.action,
            price: item.signal_price,
            timestamp: item.timestamp,
            resultProfit: item.result_profit,
            aiScore: item.ai_features?.ai_score || 0,
            regime: item.ai_features?.regime || "UNKNOWN",
            metaProb: item.ai_features?.meta_prob || 0,
            dqnAction: item.ai_features?.dqn_action || 0,
            quantumPolicy: item.ai_features?.quantum_policy || [],
        }));
        
        // 统计信息
        const stats = {
            total: logs.length,
            buyCount: logs.filter(l => l.action === "BUY").length,
            sellCount: logs.filter(l => l.action === "SELL").length,
            waitCount: logs.filter(l => l.action === "WAIT").length,
        };
        
        return NextResponse.json({ logs, stats });
    } catch (e) {
        console.error("AI Logs API error:", e);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}

