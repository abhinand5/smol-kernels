import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'
import { Layout } from './components/Layout'
import { Landing } from './components/Landing'
import { Reader } from './components/Reader'
import { ComingSoon } from './components/ComingSoon'

const basename = import.meta.env.BASE_URL === '/' ? '/' : import.meta.env.BASE_URL.replace(/\/$/, '')

const router = createBrowserRouter(
  [
    {
      element: <Layout />,
      children: [
        { path: '/', element: <Landing /> },
        { path: '/u/:slug', element: <Reader /> },
        {
          path: '/dashboard',
          element: (
            <ComingSoon
              tag="benchmark dashboard"
              title="The Scoreboard"
              blurb="Once your kernels emit results, this becomes a live roofline: CUDA vs Triton vs PyTorch, GB/s and % of your measured ceilings, tracked across every unit and every iteration."
              detail="Each benchmark harness writes a structured result; the dashboard ingests them and charts your climb up the roofline."
            />
          ),
        },
        {
          path: '/visualizer',
          element: (
            <ComingSoon
              tag="execution visualizer"
              title="The Machine, Animated"
              blurb="An interactive view of the execution model — grids, blocks, warps in lockstep, memory coalescing, and tiles streaming through SRAM."
              detail="The showpiece. Watch a warp diverge, a coalesced load collapse into one transaction, and a tiled GEMM climb past the ridge."
            />
          ),
        },
      ],
    },
  ],
  { basename },
)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
