import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

/**
 * 调试 Supabase 连接
 * GET /api/debug/supabase
 */
export async function GET() {
  try {
    // 检查环境变量
    const hasUrl = !!process.env.NEXT_PUBLIC_SUPABASE_URL;
    const hasKey = !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!hasUrl || !hasKey) {
      return NextResponse.json({
        success: false,
        error: '环境变量配置缺失',
        config: {
          NEXT_PUBLIC_SUPABASE_URL: hasUrl ? '✅ 已配置' : '❌ 缺失',
          NEXT_PUBLIC_SUPABASE_ANON_KEY: hasKey ? '✅ 已配置' : '❌ 缺失',
        },
        recommendation: '请检查 .env.local 文件，确保包含 NEXT_PUBLIC_SUPABASE_URL 和 NEXT_PUBLIC_SUPABASE_ANON_KEY，然后重启开发服务器。'
      });
    }

    // 测试连接：尝试查询 trades 表
    const { error, count } = await supabase
      .from('trades')
      .select('id', { count: 'exact', head: true });

    if (error) {
      return NextResponse.json({
        success: false,
        error: 'Supabase 连接失败',
        details: {
          message: error.message,
          code: error.code,
          details: error.details,
          hint: error.hint,
        },
        config: {
          url: process.env.NEXT_PUBLIC_SUPABASE_URL?.substring(0, 30) + '...',
          keyLength: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.length,
        },
        possibleCauses: [
          '1. Supabase 项目已暂停（免费计划会自动暂停）',
          '2. API 密钥已过期或无效',
          '3. trades 表不存在或 RLS 策略配置错误',
          '4. 网络连接问题',
        ],
        recommendation: '请登录 Supabase 控制台检查项目状态，确保项目正在运行且 trades 表已创建。'
      });
    }

    // 测试成功
    return NextResponse.json({
      success: true,
      message: '✅ Supabase 连接正常',
      config: {
        url: process.env.NEXT_PUBLIC_SUPABASE_URL,
        keyLength: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.length,
      },
      database: {
        tradesTableExists: true,
        tradesCount: count || 0,
      },
      timestamp: new Date().toISOString(),
    });

  } catch (error: unknown) {
    console.error('Debug API error:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json({
      success: false,
      error: '调试 API 错误',
      message: errorMessage,
    }, { status: 500 });
  }
}

