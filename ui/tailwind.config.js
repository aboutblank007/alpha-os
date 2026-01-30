/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: "var(--bg-dark)",
        panel: "var(--bg-panel)",
        primary: "var(--primary)",
        secondary: "var(--secondary)",
        success: "var(--success)",
        danger: "var(--danger)",
        warning: "var(--warning)",
        main: "var(--text-main)",
        dim: "var(--text-dim)",
        border: "var(--border-color)",
        glass: {
          bg: "var(--glass-bg)",
          border: "var(--glass-border)",
        }
      },
      fontFamily: {
        mono: ["var(--font-mono)", "monospace"],
        sans: ["var(--font-sans)", "sans-serif"],
      },
      animation: {
        'pulse-glow': 'pulse-glow 3s ease-in-out infinite',
      }
    },
  },
  plugins: [],
}
