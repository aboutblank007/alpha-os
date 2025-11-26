import { createClient } from '@supabase/supabase-js'
import { env } from '../env'

export const supabase = createClient(env.NEXT_PUBLIC_SUPABASE_URL, env.NEXT_PUBLIC_SUPABASE_ANON_KEY)

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
  mae?: number
  mfe?: number
}

export interface Account {
  id: string
  created_at: string
  initial_balance: number
  current_balance: number
}
