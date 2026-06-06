import { useEffect } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'

export function Layout() {
  const { pathname } = useLocation()
  useEffect(() => {
    if (!pathname.includes('#')) window.scrollTo({ top: 0 })
  }, [pathname])

  return (
    <div className="lg:grid lg:grid-cols-[270px_minmax(0,1fr)]">
      {/* mobile top bar */}
      <header className="flex items-center justify-between border-b border-[var(--color-line)] px-5 py-3 lg:hidden">
        <Link to="/" className="mono text-[15px] font-bold tracking-tight text-[var(--color-ink)]">
          smol<span className="text-[var(--color-amber)]">·</span>kernels
        </Link>
        <Link to="/u/unit-00-gpu-internals-and-triton-model" className="eyebrow">
          start →
        </Link>
      </header>

      <Sidebar />
      <main className="min-w-0">
        <Outlet />
      </main>
    </div>
  )
}
