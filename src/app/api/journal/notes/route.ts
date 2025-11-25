import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// GET - 获取笔记 (可选日期范围)
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get('date');
    const startDate = searchParams.get('startDate');
    const endDate = searchParams.get('endDate');
    const page = parseInt(searchParams.get('page') || '0');
    const pageSize = parseInt(searchParams.get('pageSize') || '20');

    // Calculate range for pagination
    const from = page * pageSize;
    const to = from + pageSize - 1;

    let query = supabase.from('journal_notes').select('*', { count: 'exact' });

    if (date) {
      // 获取特定日期的笔记
      query = query.eq('date', date);
      const { data, error } = await query.single();

      if (error) {
        if (error.code === 'PGRST116') {
          // 未找到记录
          return NextResponse.json({ note: null });
        }
        throw error;
      }

      return NextResponse.json({ note: data });
    } else if (startDate && endDate) {
      // 获取日期范围内的笔记
      query = query
        .gte('date', startDate)
        .lte('date', endDate)
        .order('date', { ascending: false });
    } else {
      // 获取最近的笔记 (Paginated)
      query = query
        .order('date', { ascending: false })
        .range(from, to);
    }

    const { data, error, count } = await query;

    if (error) throw error;

    return NextResponse.json({ 
        notes: data,
        count,
        page,
        pageSize 
    });
  } catch (error: unknown) {
    console.error('获取笔记错误:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: '获取笔记失败: ' + errorMessage },
      { status: 500 }
    );
  }
}

// POST - 创建新笔记
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { date, content, mood, tags } = body;

    if (!date || !content) {
      return NextResponse.json(
        { error: '日期和内容不能为空' },
        { status: 400 }
      );
    }

    // 检查该日期是否已有笔记
    const { data: existing } = await supabase
      .from('journal_notes')
      .select('id')
      .eq('date', date)
      .single();

    if (existing) {
      return NextResponse.json(
        { error: '该日期已有笔记,请使用更新接口' },
        { status: 400 }
      );
    }

    const { data, error } = await supabase
      .from('journal_notes')
      .insert([
        {
          date,
          content,
          mood: mood || null,
          tags: tags || [],
        },
      ])
      .select()
      .single();

    if (error) throw error;

    return NextResponse.json({ note: data });
  } catch (error: unknown) {
    console.error('创建笔记错误:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: '创建笔记失败: ' + errorMessage },
      { status: 500 }
    );
  }
}

// PUT - 更新笔记
export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const { date, content, mood, tags } = body;

    if (!date || !content) {
      return NextResponse.json(
        { error: '日期和内容不能为空' },
        { status: 400 }
      );
    }

    const { data, error } = await supabase
      .from('journal_notes')
      .update({
        content,
        mood: mood || null,
        tags: tags || [],
      })
      .eq('date', date)
      .select()
      .single();

    if (error) throw error;

    return NextResponse.json({ note: data });
  } catch (error: unknown) {
    console.error('更新笔记错误:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: '更新笔记失败: ' + errorMessage },
      { status: 500 }
    );
  }
}

// DELETE - 删除笔记
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get('date');

    if (!date) {
      return NextResponse.json({ error: '日期不能为空' }, { status: 400 });
    }

    const { error } = await supabase
      .from('journal_notes')
      .delete()
      .eq('date', date);

    if (error) throw error;

    return NextResponse.json({ success: true });
  } catch (error: unknown) {
    console.error('删除笔记错误:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: '删除笔记失败: ' + errorMessage },
      { status: 500 }
    );
  }
}
