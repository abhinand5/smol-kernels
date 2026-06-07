import { useEffect, useState } from 'react'

const REPO = 'abhinand5/smol-kernels'
const REPO_URL = `https://github.com/${REPO}`

const StarIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>
)

function formatCount(n: number): string {
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, '') + 'k'
  return String(n)
}

/** A quiet "star on GitHub" pill with a live count — invitation, not a plea.
    Degrades to just "Star" if the API is unreachable or rate-limited. */
export function GitHubStar({ className = '' }: { className?: string }) {
  const [count, setCount] = useState<number | null>(() => {
    const cached = typeof localStorage !== 'undefined' ? localStorage.getItem('gh-stars') : null
    return cached ? Number(cached) : null
  })

  useEffect(() => {
    let alive = true
    fetch(`https://api.github.com/repos/${REPO}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('rate-limited'))))
      .then((d: { stargazers_count?: number }) => {
        if (!alive || typeof d.stargazers_count !== 'number') return
        setCount(d.stargazers_count)
        localStorage.setItem('gh-stars', String(d.stargazers_count))
      })
      .catch(() => {
        /* keep cached/null — the pill still works as a plain link */
      })
    return () => {
      alive = false
    }
  }, [])

  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noreferrer"
      aria-label="Star smol-kernels on GitHub"
      className={`group inline-flex items-center gap-2 rounded-md border border-[var(--color-line)] bg-[var(--color-surface)]/50 px-3 py-1.5 transition-colors hover:border-[var(--color-line-bright)] ${className}`}
    >
      <span className="text-[var(--color-faint)] transition-colors group-hover:text-[var(--color-amber)]">
        <StarIcon />
      </span>
      <span className="mono text-[12px] font-medium text-[var(--color-ink-dim)] transition-colors group-hover:text-[var(--color-ink)]">
        Star
      </span>
      {/* show the count only once it means something — a bare "0" reads desperate */}
      {count !== null && count > 0 && (
        <>
          <span className="h-3.5 w-px bg-[var(--color-line-bright)]" aria-hidden="true" />
          <span className="mono text-[12px] tabular-nums text-[var(--color-faint)] transition-colors group-hover:text-[var(--color-ink-dim)]">
            {formatCount(count)}
          </span>
        </>
      )}
    </a>
  )
}
