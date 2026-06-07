import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The curriculum markdown lives in ../docs (repo root). Allow Vite to read it;
// content is glob-imported as raw strings and inlined at build time.
// GitHub Pages serves project sites from /<repo>/, so build with that base in CI.
const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = process.env.GITHUB_PAGES === 'true' && repoName ? `/${repoName}/` : '/'

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  server: { fs: { allow: ['..'] } },
})
