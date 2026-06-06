import { NavLink, Link } from 'react-router-dom'
import { units, references, type Regime } from '../lib/content'

export function RegimeDot({ regime, size = 7 }: { regime: Regime; size?: number }) {
  const bg =
    regime === 'compute'
      ? 'var(--color-amber)'
      : regime === 'mem'
        ? 'var(--color-cool)'
        : 'linear-gradient(90deg, var(--color-cool), var(--color-amber))'
  return (
    <span
      style={{ width: size, height: size, background: bg }}
      className="inline-block shrink-0 rounded-full"
      aria-hidden="true"
    />
  )
}

function navItemClass({ isActive }: { isActive: boolean }) {
  return [
    'group flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13.5px] transition-colors',
    isActive
      ? 'bg-[var(--color-surface)] text-[var(--color-ink)]'
      : 'text-[var(--color-ink-dim)] hover:bg-[var(--color-surface)]/60 hover:text-[var(--color-ink)]',
  ].join(' ')
}

export function Sidebar() {
  return (
    <aside className="hidden lg:flex lg:sticky lg:top-0 lg:h-screen lg:flex-col border-r border-[var(--color-line)] bg-[var(--color-bg-2)]/40">
      <div className="flex h-full flex-col overflow-y-auto px-4 py-6">
        {/* brand */}
        <Link to="/" className="px-2.5">
          <div className="mono text-[17px] font-bold leading-none tracking-tight text-[var(--color-ink)]">
            smol<span className="text-[var(--color-amber)]">·</span>kernels
          </div>
          <div className="eyebrow mt-2">CUDA + Triton · from scratch</div>
        </Link>

        {/* calibrate hint */}
        <div className="mono mt-5 rounded-md border border-[var(--color-line)] bg-[var(--color-surface)]/50 px-2.5 py-2 text-[10.5px] leading-relaxed text-[var(--color-faint)]">
          <span className="text-[var(--color-cool)]">$</span> uv run scripts/
          <wbr />
          calibrate_gpu.py
          <div className="mt-1 text-[var(--color-faint)]/70">calibrate first — targets adapt to your GPU</div>
        </div>

        {/* curriculum */}
        <nav className="mt-6 flex-1 space-y-6">
          <div>
            <div className="eyebrow mb-2 px-2.5">Curriculum</div>
            <ul className="space-y-0.5">
              {units.map((d) => (
                <li key={d.slug}>
                  <NavLink to={`/u/${d.slug}`} className={navItemClass}>
                    <span className="mono w-5 shrink-0 text-[11px] text-[var(--color-faint)] group-hover:text-[var(--color-amber)]">
                      {String(d.unit).padStart(2, '0')}
                    </span>
                    <span className="flex-1 truncate">{d.shortTitle}</span>
                    <RegimeDot regime={d.regime} />
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>

          {references.length > 0 && (
            <div>
              <div className="eyebrow mb-2 px-2.5">Reference</div>
              <ul className="space-y-0.5">
                {references.map((d) => (
                  <li key={d.slug}>
                    <NavLink to={`/u/${d.slug}`} className={navItemClass}>
                      <span className="mono w-5 shrink-0 text-[11px] text-[var(--color-faint)]">◇</span>
                      <span className="flex-1 truncate">{d.shortTitle}</span>
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <div className="eyebrow mb-2 px-2.5">Tools</div>
            <ul className="space-y-0.5">
              <ToolLink to="/dashboard" label="Benchmark dashboard" />
              <ToolLink to="/visualizer" label="Execution visualizer" />
            </ul>
          </div>
        </nav>

        <a
          href="https://github.com/abhinand5/smol-kernels"
          target="_blank"
          rel="noreferrer"
          className="eyebrow mt-6 px-2.5 hover:text-[var(--color-amber)]"
        >
          github ↗
        </a>
      </div>
    </aside>
  )
}

function ToolLink({ to, label }: { to: string; label: string }) {
  return (
    <li>
      <NavLink to={to} className={navItemClass}>
        <span className="mono w-5 shrink-0 text-[11px] text-[var(--color-faint)]">▸</span>
        <span className="flex-1 truncate">{label}</span>
        <span className="mono rounded-sm bg-[var(--color-surface-2)] px-1 py-0.5 text-[8.5px] tracking-wider text-[var(--color-faint)]">
          SOON
        </span>
      </NavLink>
    </li>
  )
}
