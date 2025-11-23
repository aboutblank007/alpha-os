import { NextResponse } from 'next/server';

const BRIDGE_API_URL = process.env.TRADING_BRIDGE_API_URL || 'http://api.lootool.cn:8000';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action, symbol, volume } = body;

    if (!action || !symbol || !volume) {
      return NextResponse.json(
        { error: 'Missing required fields: action, symbol, volume' },
        { status: 400 }
      );
    }

    // 转发请求到 Bridge API
    const res = await fetch(`${BRIDGE_API_URL}/trade/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ action, symbol, volume }),
    });

    if (!res.ok) {
      throw new Error(`Bridge API error: ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Trade execution failed:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 502 }
    );
  }
}

