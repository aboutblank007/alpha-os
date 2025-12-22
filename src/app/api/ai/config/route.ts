import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

export async function GET() {
    try {
        const supabase = createClient(supabaseUrl, supabaseKey);
        
        // 获取全局 AI 设置
        const { data: settings } = await supabase
            .from("ai_settings")
            .select("*")
            .single();
        
        // 获取自动化规则
        const { data: rules } = await supabase
            .from("automation_rules")
            .select("*")
            .order("symbol");
        
        return NextResponse.json({
            settings: settings || {},
            rules: rules || [],
        });
    } catch (e) {
        console.error("AI Config API error:", e);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}

export async function PUT(request: Request) {
    try {
        const supabase = createClient(supabaseUrl, supabaseKey);
        const body = await request.json();
        const { type, id, data } = body;
        
        if (type === "settings") {
            const { error } = await supabase
                .from("ai_settings")
                .update(data)
                .eq("id", id || 1);
            
            if (error) throw error;
        } else if (type === "rule") {
            const { error } = await supabase
                .from("automation_rules")
                .update(data)
                .eq("id", id);
            
            if (error) throw error;
        }
        
        return NextResponse.json({ success: true });
    } catch (e) {
        console.error("AI Config update error:", e);
        return NextResponse.json({ error: "Update failed" }, { status: 500 });
    }
}

