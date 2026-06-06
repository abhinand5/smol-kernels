import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The curriculum markdown lives in ../docs (repo root). Allow Vite to read it;
// content is glob-imported as raw strings and inlined at build time.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { fs: { allow: ['..'] } },
})
