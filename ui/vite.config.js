import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // 添加这一行，允许局域网访问
    port: 5173,        // 可选：指定端口
    // 可选：配置代理（如果需要）
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
