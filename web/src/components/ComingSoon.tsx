import { Link } from 'react-router-dom'

export function ComingSoon({
  tag,
  title,
  blurb,
  detail,
}: {
  tag: string
  title: string
  blurb: string
  detail: string
}) {
  return (
    <div className="mx-auto flex min-h-[70vh] max-w-2xl flex-col justify-center px-6 py-20">
      <div className="rise">
        <div className="eyebrow mb-4 flex items-center gap-2">
          {tag}
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber" />
          <span className="text-amber">soon</span>
        </div>
        <h1 className="font-[var(--font-body)] text-4xl font-medium tracking-tight text-ink lg:text-5xl">
          {title}
        </h1>
        <p className="mt-5 text-[1.05rem] leading-relaxed text-ink-dim">{blurb}</p>

        <div className="mono mt-8 rounded-xl border border-line bg-surface/40 p-5 text-[12.5px] leading-relaxed text-faint">
          <span className="text-cool">// </span>
          {detail}
        </div>

        <Link to="/" className="eyebrow mt-10 inline-block hover:text-amber">
          ← back to the curriculum
        </Link>
      </div>
    </div>
  )
}
