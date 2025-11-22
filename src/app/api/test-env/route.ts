import { NextResponse } from 'next/server';

/**
 * 测试环境变量配置
 * GET /api/test-env
 */
export async function GET() {
  const envStatus = {
    oanda: {
      apiKey: !!process.env.OANDA_API_KEY,
      apiKeyLength: process.env.OANDA_API_KEY?.length || 0,
      accountId: !!process.env.OANDA_ACCOUNT_ID,
      environment: process.env.OANDA_ENVIRONMENT || 'not set',
    },
    supabase: {
      url: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      urlValue: process.env.NEXT_PUBLIC_SUPABASE_URL?.substring(0, 30) + '...',
      anonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      anonKeyLength: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.length || 0,
    },
    nodeEnv: process.env.NODE_ENV,
  };

  return NextResponse.json({
    success: true,
    message: '环境变量检查',
    env: envStatus,
    recommendations: {
      oanda: envStatus.oanda.apiKey && envStatus.oanda.accountId 
        ? '✅ OANDA 配置完整' 
        : '❌ OANDA 配置缺失',
      supabase: envStatus.supabase.url && envStatus.supabase.anonKey 
        ? '✅ Supabase 配置完整' 
        : '❌ Supabase 配置缺失',
    }
  });
}

