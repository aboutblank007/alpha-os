import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function GET() {
    try {
        const { data, error } = await supabase
            .from('ai_settings')
            .select('*')
            .single();

        if (error) throw error;

        return NextResponse.json(data);
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: message }, { status: 500 });
    }
}

export async function POST(req: Request) {
    try {
        const body = await req.json();
        // Ensure we only update row 1
        const { data, error } = await supabase
            .from('ai_settings')
            .update({ ...body, updated_at: new Date() })
            .eq('id', 1)
            .select()
            .single();

        if (error) throw error;

        return NextResponse.json(data);
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
