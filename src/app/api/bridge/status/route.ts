import { NextResponse } from 'next/server';
import { env } from '@/env';

const BRIDGE_API_URL = env.TRADING_BRIDGE_API_URL;

export async function GET() {
  try {
    // 设置 5 秒超时
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const res = await fetch(`${BRIDGE_API_URL}/status`, {
      next: { revalidate: 0 }, // 禁用缓存
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(`Bridge API error: ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Bridge status check failed:', error);
    return NextResponse.json(
      {
        bridge_status: 'disconnected',
        error: error instanceof Error ? error.message : 'Unknown error',
        last_mt5_update: {},
        pending_commands: 0
      },
      { status: 200 } // 返回 200 但状态为 disconnected，方便前端处理
    );
  }
}

