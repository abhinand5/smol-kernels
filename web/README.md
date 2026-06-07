# smol-kernels · web reader

A modern reader for the CUDA + Triton curriculum in [`../docs`](../docs). Renders
the unit markdown with a "profiler editorial" design — editorial serif body,
monospace technical chrome, a thermal accent, and a roofline motif that threads
memory-bound → compute-bound through the units.

## Run

```bash
cd web
npm install
npm run dev      # http://localhost:5173
```

```bash
npm run build    # type-check + production build to web/dist
npm run preview  # serve the production build
```

## Deploy on GitHub Pages

This repo includes `.github/workflows/deploy-web.yml`. On every push to `main`,
GitHub Actions runs `npm ci && npm run build` in `web/`, uploads `web/dist`, and
deploys it to GitHub Pages.

One-time setup in GitHub:

1. Go to **Settings → Pages**.
2. Set **Build and deployment → Source** to **GitHub Actions**.
3. Push to `main`, then watch **Actions → Deploy web to GitHub Pages**.

The workflow sets the Vite base path for a project Pages site like
`https://<user>.github.io/<repo>/` and copies `index.html` to `404.html` so deep
links such as `/u/unit00` route back into the SPA.

## How it works

- **Content** is the markdown in `../docs/*.md`, glob-imported as raw strings at
  build time (`src/lib/content.ts`) — the docs stay the single source of truth.
  Add a unit doc and it appears in the nav automatically. `my-gpu-spec.md` (the
  gitignored personal calibration) and `README.md` are excluded.
- **Reader** (`src/components/Reader.tsx`) renders via `react-markdown` +
  `remark-gfm` (tables) + `rehype-highlight` (code), with cross-doc `.md` links
  rewritten to in-app routes and a scroll-spy table of contents.
- **Routes:** `/` landing (roofline hero + unit index), `/u/:slug` reader,
  `/dashboard` and `/visualizer` are stubs for the next phases.

## Stack

Vite · React · TypeScript · Tailwind CSS v4 · React Router · Motion.

## Notes / next

- Uses `BrowserRouter`; for static hosting (e.g. GitHub Pages) add a 404 →
  `index.html` fallback, or switch to `HashRouter`.
- The JS bundle is large because `rehype-highlight` pulls in many languages;
  restrict to `{ python, cpp, bash, plaintext }` to shrink it.
- Next phases: the benchmark **dashboard** (ingest result JSON) and the
  execution **visualizer** (animated grids/warps/tiles).
