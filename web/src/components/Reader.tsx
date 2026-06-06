import { Navigate, useParams, Link } from 'react-router-dom'
import { getDoc, units, regimeLabel, type Doc } from '../lib/content'
import { Markdown } from './Markdown'
import { Toc } from './Toc'
import { RegimeDot } from './Sidebar'

export function Reader() {
  const { slug } = useParams()
  const doc = getDoc(slug ?? '')
  if (!doc) return <Navigate to="/" replace />

  const idx = units.findIndex((u) => u.slug === doc.slug)
  const prev = idx > 0 ? units[idx - 1] : null
  const next = idx >= 0 && idx < units.length - 1 ? units[idx + 1] : null

  return (
    <div className="mx-auto max-w-[1180px] px-6 py-12 lg:px-12 xl:grid xl:grid-cols-[minmax(0,1fr)_190px] xl:gap-14">
      <article className="rise min-w-0">
        <div className="eyebrow mb-7 flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="text-ink-dim">
            {doc.unit !== null ? `Unit ${String(doc.unit).padStart(2, '0')}` : 'Reference'}
          </span>
          {doc.regime && (
            <span className="flex items-center gap-1.5">
              <RegimeDot regime={doc.regime} />
              {regimeLabel[doc.regime]}
            </span>
          )}
        </div>

        <div className="prose">
          <Markdown>{doc.body}</Markdown>
        </div>

        <nav className="mt-20 grid grid-cols-2 gap-4 border-t border-line pt-7">
          {prev ? <PrevNext dir="prev" doc={prev} /> : <span />}
          {next ? <PrevNext dir="next" doc={next} /> : <span />}
        </nav>
      </article>

      <aside className="hidden xl:block">
        <div className="sticky top-12">
          <Toc headings={doc.headings} />
        </div>
      </aside>
    </div>
  )
}

function PrevNext({ dir, doc }: { dir: 'prev' | 'next'; doc: Doc }) {
  const isNext = dir === 'next'
  return (
    <Link
      to={`/u/${doc.slug}`}
      className={[
        'group rounded-lg border border-line bg-surface/40 p-4 transition-colors hover:border-line-bright hover:bg-surface',
        isNext ? 'text-right' : '',
      ].join(' ')}
    >
      <div className="eyebrow mb-1.5 text-faint group-hover:text-amber">
        {isNext ? 'next →' : '← prev'}
      </div>
      <div className="mono text-[13px] text-ink-dim group-hover:text-ink">
        {doc.unit !== null ? `${String(doc.unit).padStart(2, '0')} · ` : ''}
        {doc.shortTitle}
      </div>
    </Link>
  )
}
