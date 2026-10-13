import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/patient-search': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        timeout: 900000,
        proxyTimeout: 900000,
      },
    },
  },
})
