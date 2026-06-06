import { useEffect, useState } from 'react'
import type { Heading } from '../lib/content'

function useScrollSpy(ids: string[]) {
  const [active, setActive] = useState('')
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setActive(visible[0].target.id)
      },
      { rootMargin: '-72px 0px -68% 0px', threshold: 0 },
    )
    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })
    return () => observer.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids.join(',')])
  return active
}

export function Toc({ headings }: { headings: Heading[] }) {
  const sections = headings.filter((h) => h.depth === 2)
  const active = useScrollSpy(sections.map((h) => h.id))
  if (sections.length === 0) return null

  return (
    <nav>
      <div className="eyebrow mb-3">On this page</div>
      <ul className="space-y-1.5 border-l border-line">
        {sections.map((h) => {
          const on = active === h.id
          return (
            <li key={h.id}>
              <a
                href={`#${h.id}`}
                className={[
                  '-ml-px block border-l-2 pl-3 text-[12.5px] leading-snug transition-colors',
                  on
                    ? 'border-amber text-amber'
                    : 'border-transparent text-faint hover:text-ink-dim',
                ].join(' ')}
              >
                {h.text.replace(/^[\d.]+\s+/, '')}
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
