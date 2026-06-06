import { Link } from 'react-router-dom'
import { units, references } from '../lib/content'
import { RegimeDot } from './Sidebar'

export function Landing() {
  return (
    <div className="mx-auto max-w-[1180px] px-6 py-14 lg:px-12 lg:py-20">
      {/* ---------------- hero ---------------- */}
      <section className="grid items-center gap-12 lg:grid-cols-[1.05fr_1fr]">
        <div className="rise">
          <div className="eyebrow mb-5">CUDA + Triton · written from scratch</div>
          <h1 className="font-[var(--font-body)] text-5xl font-medium leading-[1.04] tracking-tight text-ink lg:text-6xl">
            Learn the GPU{' '}
            <span className="italic text-amber">twice.</span>
          </h1>
          <p className="mt-6 max-w-xl text-[1.06rem] leading-relaxed text-ink-dim">
            A hands-on kernel curriculum — every idea once in{' '}
            <span className="text-ink">CUDA&nbsp;C++</span>, once in{' '}
            <span className="text-ink">Triton</span> — from a memcpy bandwidth
            benchmark to FlashAttention. You write every kernel; the roofline grades
            it against <span className="text-ink">your own GPU</span>.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              to="/u/unit-00-gpu-internals-and-triton-model"
              className="mono rounded-md bg-amber px-4 py-2.5 text-[13px] font-semibold text-[#1a1206] transition-transform hover:-translate-y-0.5"
            >
              Start at Unit 00 →
            </Link>
            <code className="mono rounded-md border border-line bg-surface/60 px-3 py-2.5 text-[12px] text-faint">
              <span className="text-cool">$</span> uv run scripts/calibrate_gpu.py
            </code>
          </div>

          <div className="mono mt-9 flex flex-wrap gap-x-7 gap-y-2 text-[11px] uppercase tracking-wider text-faint">
            <span>
              <span className="text-ink-dim">{units.length}</span> units
            </span>
            <span>
              <span className="text-ink-dim">2</span> languages
            </span>
            <span>
              <span className="text-ink-dim">memcpy</span> → flashattention
            </span>
          </div>
        </div>

        <div className="rise" style={{ animationDelay: '120ms' }}>
          <RooflineHero />
        </div>
      </section>

      {/* ---------------- the curriculum ---------------- */}
      <section className="mt-24">
        <div className="mb-7 flex items-end justify-between">
          <h2 className="font-[var(--font-body)] text-2xl font-medium text-ink">The climb</h2>
          <div className="eyebrow hidden items-center gap-4 sm:flex">
            <span className="flex items-center gap-1.5">
              <RegimeDot regime="mem" /> memory-bound
            </span>
            <span className="flex items-center gap-1.5">
              <RegimeDot regime="compute" /> compute-bound
            </span>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {units.map((d, i) => (
            <Link
              key={d.slug}
              to={`/u/${d.slug}`}
              className="rise group flex flex-col rounded-xl border border-line bg-surface/30 p-5 transition-all hover:-translate-y-0.5 hover:border-line-bright hover:bg-surface"
              style={{ animationDelay: `${140 + i * 45}ms` }}
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="mono text-[12px] text-faint group-hover:text-amber">
                  {String(d.unit).padStart(2, '0')}
                </span>
                <RegimeDot regime={d.regime} />
              </div>
              <div className="font-[var(--font-body)] text-[1.15rem] font-medium leading-snug text-ink">
                {d.shortTitle}
              </div>
              <p className="mt-2 line-clamp-3 text-[13.5px] leading-snug text-faint">
                {d.tagline}
              </p>
            </Link>
          ))}
        </div>

        {references.length > 0 && (
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {references.map((d) => (
              <Link
                key={d.slug}
                to={`/u/${d.slug}`}
                className="group flex items-center gap-3 rounded-xl border border-dashed border-line p-4 transition-colors hover:border-line-bright hover:bg-surface/40"
              >
                <span className="mono text-[12px] text-faint">◇</span>
                <span className="font-[var(--font-body)] text-[1.02rem] text-ink-dim group-hover:text-ink">
                  {d.shortTitle}
                </span>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

/** The roofline as the curriculum's spine: a cool memory-bound slope rising
    to a hot compute-bound roof, with the units placed along it. */
function RooflineHero() {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-line bg-gradient-to-b from-[#120f0a] to-[#0c0a07] p-5 shadow-[0_30px_60px_-40px_rgba(0,0,0,0.9)]">
      <div className="eyebrow mb-3 flex justify-between">
        <span>roofline</span>
        <span className="text-faint">throughput vs intensity</span>
      </div>
      <svg viewBox="0 0 640 340" className="w-full" role="img" aria-label="Roofline chart of the curriculum">
        <defs>
          <linearGradient id="roof" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0" stopColor="var(--color-cool)" />
            <stop offset="0.55" stopColor="var(--color-cool)" />
            <stop offset="0.7" stopColor="var(--color-amber)" />
            <stop offset="1" stopColor="var(--color-amber)" />
          </linearGradient>
          <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--color-amber)" stopOpacity="0.10" />
            <stop offset="1" stopColor="var(--color-amber)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* axes */}
        <line x1="64" y1="300" x2="612" y2="300" stroke="var(--color-line-bright)" strokeWidth="1" />
        <line x1="64" y1="40" x2="64" y2="300" stroke="var(--color-line-bright)" strokeWidth="1" />

        {/* ridge marker */}
        <line x1="372" y1="60" x2="372" y2="300" stroke="var(--color-line)" strokeWidth="1" strokeDasharray="3 4" />
        <text x="372" y="70" textAnchor="middle" className="mono" fontSize="9.5" fill="var(--color-faint)">
          RIDGE
        </text>

        {/* region labels — placed in clear zones, off the curve and markers */}
        <text x="108" y="184" className="mono" fontSize="10" fill="var(--color-cool-dim)">
          MEMORY-BOUND
        </text>
        <text x="604" y="92" textAnchor="end" className="mono" fontSize="10" fill="var(--color-amber)" opacity="0.85">
          COMPUTE-BOUND
        </text>

        {/* fill under the roof */}
        <path d="M64 300 L372 108 L612 108 L612 300 Z" fill="url(#fill)" />

        {/* the roof */}
        <path
          className="roofdraw"
          d="M64 300 L372 108 L612 108"
          fill="none"
          stroke="url(#roof)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* unit markers along the slope, then the flat */}
        {[
          { x: 96, y: 282, l: '00 memcpy', lx: 108, ly: 295, anchor: 'start' as const },
          { x: 150, y: 248 },
          { x: 210, y: 210 },
          { x: 270, y: 172 },
          { x: 330, y: 134 },
          { x: 430, y: 108, l: '06 GEMM', lx: 430, ly: 127, anchor: 'middle' as const },
          { x: 560, y: 108, l: '07 FlashAttn', lx: 560, ly: 127, anchor: 'middle' as const },
        ].map((p, i) => (
          <g key={i}>
            <circle
              cx={p.x}
              cy={p.y}
              r="4.5"
              fill="var(--color-bg)"
              stroke={p.x >= 372 ? 'var(--color-amber)' : 'var(--color-cool)'}
              strokeWidth="2"
            />
            {p.l && (
              <text x={p.lx} y={p.ly} textAnchor={p.anchor} className="mono" fontSize="9.5" fill="var(--color-ink-dim)">
                {p.l}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}
