import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The curriculum markdown lives in ../docs (repo root). Allow Vite to read it;
// content is glob-imported as raw strings and inlined at build time.
// Served from the root of a custom domain (kernels.abhinandb.com via public/CNAME),
// so the base path is '/'. (For a project page at user.github.io/<repo> this would
// need to be '/<repo>/' instead.)
export default defineConfig({
  base: '/',
  plugins: [react(), tailwindcss()],
  server: { fs: { allow: ['..'] } },
})
