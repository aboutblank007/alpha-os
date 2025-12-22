import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const limit = parseInt(searchParams.get("limit") || "500");
        const status = searchParams.get("status") || "closed";
        
        const supabase = createClient(supabaseUrl, supabaseKey);
        
        // 获取交易数据
        const { data: trades, error } = await supabase
            .from("trades")
            .select("*")
            .eq("status", status)
            .order("entry_time", { ascending: false })
            .limit(limit);
        
        if (error) {
            console.error("Supabase error:", error);
            return NextResponse.json({ error: error.message }, { status: 500 });
        }
        
        // 转换数据格式
        const formattedTrades = (trades || []).map((t) => ({
            id: t.id,
            symbol: t.symbol,
            type: t.side?.toUpperCase() || "BUY",
            profit: parseFloat(t.pnl_net) || 0,
            volume: parseFloat(t.quantity) || 0.01,
            entryPrice: parseFloat(t.entry_price) || 0,
            exitPrice: parseFloat(t.exit_price) || 0,
            openTime: t.entry_time,
            closeTime: t.exit_time || t.entry_time,
            swap: parseFloat(t.swap) || 0,
            commission: parseFloat(t.commission) || 0,
            stopLoss: parseFloat(t.stop_loss) || 0,
            takeProfit: parseFloat(t.take_profit) || 0,
            status: t.status,
        }));
        
        return NextResponse.json({
            trades: formattedTrades,
            total: formattedTrades.length,
        });
    } catch (e) {
        console.error("Analytics API error:", e);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}

