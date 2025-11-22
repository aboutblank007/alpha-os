import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

// 验证环境变量
if (!supabaseUrl || !supabaseAnonKey) {
    const errorMsg = `
╔═══════════════════════════════════════════════════════════════╗
║          ❌ SUPABASE 配置错误                                 ║
╚═══════════════════════════════════════════════════════════════╝

缺少必需的环境变量:
  • NEXT_PUBLIC_SUPABASE_URL: ${supabaseUrl ? '✅ 已设置' : '❌ 未设置'}
  • NEXT_PUBLIC_SUPABASE_ANON_KEY: ${supabaseAnonKey ? '✅ 已设置' : '❌ 未设置'}

📝 解决步骤:
  1. 确认 .env.local 文件存在于项目根目录
  2. 文件中包含以下内容:
     NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
     NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJxxx...
  3. 完全重启开发服务器 (Ctrl+C 然后 npm run dev)
  4. 访问 http://localhost:3000/debug 查看详细诊断

🔗 快速诊断: http://localhost:3000/debug
    `;
    
    console.error(errorMsg);
    throw new Error('Supabase configuration is missing. Please check .env.local file.');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Database Types
export interface Trade {
    id: string
    created_at: string
    symbol: string
    side: 'buy' | 'sell'
    entry_price: number
    exit_price: number | null
    quantity: number
    pnl_net: number
    pnl_gross: number
    commission: number
    swap: number
    status: 'open' | 'closed'
    notes?: string
    emotion_score?: number
    strategies?: string[]
}

export interface Account {
    id: string
    created_at: string
    initial_balance: number
    current_balance: number
}
