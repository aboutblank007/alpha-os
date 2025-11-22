import type { Config } from "tailwindcss";

export default {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "var(--background)",
                foreground: "var(--foreground)",
                accent: {
                    primary: "var(--accent-primary)",
                    secondary: "var(--accent-secondary)",
                    cyan: "var(--accent-cyan)",
                    success: "var(--accent-success)",
                    danger: "var(--accent-danger)",
                    warning: "var(--accent-warning)",
                },
                surface: {
                    glass: "var(--surface-glass)",
                    "glass-strong": "var(--surface-glass-strong)",
                    border: "var(--surface-border)",
                }
            },
            animation: {
                "fade-in": "fadeIn 0.5s ease-out forwards",
                "slide-up": "slideUp 0.5s ease-out forwards",
                "pulse-slow": "pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
            },
            keyframes: {
                fadeIn: {
                    "0%": { opacity: "0" },
                    "100%": { opacity: "1" },
                },
                slideUp: {
                    "0%": { opacity: "0", transform: "translateY(20px)" },
                    "100%": { opacity: "1", transform: "translateY(0)" },
                },
            },
            backdropBlur: {
                xs: '2px',
            }
        },
    },
    plugins: [],
} satisfies Config;
