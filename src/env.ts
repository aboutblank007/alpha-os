import { z } from 'zod';

const envSchema = z.object({
    // Supabase
    NEXT_PUBLIC_SUPABASE_URL: z.string().url("Supabase URL must be a valid URL"),
    NEXT_PUBLIC_SUPABASE_ANON_KEY: z.string().min(1, "Supabase Anon Key is required"),

    // MT5 Trading Bridge
    // Allow empty string to fallback to default, or use default if undefined
    TRADING_BRIDGE_API_URL: z.string().optional().or(z.literal('')).transform(val => val || 'http://127.0.0.1:8000'),
});

const processEnv = {
    NEXT_PUBLIC_SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL,
    NEXT_PUBLIC_SUPABASE_ANON_KEY: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    // Support legacy env names for backward compatibility
    TRADING_BRIDGE_API_URL: process.env.TRADING_BRIDGE_API_URL || process.env.BRIDGE_API_URL || process.env.NEXT_PUBLIC_BRIDGE_API_URL,
};

// Validate immediately
const parsed = envSchema.safeParse(processEnv);

if (!parsed.success) {
    console.error('❌ Invalid environment variables:', parsed.error.flatten().fieldErrors);
    throw new Error('Invalid environment variables');
}

export const env = parsed.data;
