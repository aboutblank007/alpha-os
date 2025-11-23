import { z } from 'zod';

const envSchema = z.object({
    // Supabase
    NEXT_PUBLIC_SUPABASE_URL: z.string().url("Supabase URL must be a valid URL"),
    NEXT_PUBLIC_SUPABASE_ANON_KEY: z.string().min(1, "Supabase Anon Key is required"),

    // OANDA
    OANDA_API_KEY: z.string().optional(),
    OANDA_ACCOUNT_ID: z.string().optional(),
    OANDA_ENVIRONMENT: z.enum(['practice', 'live', 'mock']).default('practice'),

    // Bridge (Optional for now, but good to have)
    TRADING_BRIDGE_API_URL: z.string().url().optional().default('http://api.lootool.cn:8000'),
});

const processEnv = {
    NEXT_PUBLIC_SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL,
    NEXT_PUBLIC_SUPABASE_ANON_KEY: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    OANDA_API_KEY: process.env.OANDA_API_KEY,
    OANDA_ACCOUNT_ID: process.env.OANDA_ACCOUNT_ID,
    OANDA_ENVIRONMENT: process.env.OANDA_ENVIRONMENT,
    TRADING_BRIDGE_API_URL: process.env.TRADING_BRIDGE_API_URL,
};

// Validate immediately
const parsed = envSchema.safeParse(processEnv);

if (!parsed.success) {
    console.error('❌ Invalid environment variables:', parsed.error.flatten().fieldErrors);
    throw new Error('Invalid environment variables');
}

export const env = parsed.data;
