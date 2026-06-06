import { slugify } from './slug'

// Glob-import every curriculum doc as a raw string, inlined at build time.
const raw = import.meta.glob('../../../docs/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

export type Regime = 'mem' | 'compute' | 'mixed' | null

export interface Heading {
  depth: 2 | 3
  text: string
  id: string
}

export interface Doc {
  slug: string // route param, = filename stem
  file: string
  kind: 'unit' | 'reference'
  unit: number | null
  title: string // full H1 text, e.g. "Unit 01: Vector Add"
  shortTitle: string // e.g. "Vector Add"
  tagline: string // first epigraph line (the "> ..." under H1)
  regime: Regime
  body: string
  headings: Heading[]
}

const REGIME: Record<number, Regime> = {
  0: 'mem',
  1: 'mem',
  2: 'mem',
  3: 'mem',
  4: 'mem',
  5: 'mem',
  6: 'compute',
  7: 'compute',
  8: 'mixed',
}

function firstH1(md: string): string {
  const m = md.match(/^#\s+(.+)$/m)
  return m ? m[1].trim() : 'Untitled'
}

function firstEpigraph(md: string): string {
  // first blockquote line after the H1
  const m = md.match(/^>\s+(.+)$/m)
  return m ? m[1].replace(/\s+/g, ' ').trim() : ''
}

function extractHeadings(md: string): Heading[] {
  const out: Heading[] = []
  // ignore fenced code blocks
  const lines = md.split('\n')
  let inFence = false
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const m = line.match(/^(##|###)\s+(.+)$/)
    if (m) {
      const text = m[2].replace(/\s+#*\s*$/, '').trim()
      out.push({ depth: m[1].length as 2 | 3, text, id: slugify(text) })
    }
  }
  return out
}

function shorten(title: string): string {
  // "Unit 01: Vector Add" -> "Vector Add"; "Reference: ..." -> "..."
  const m = title.match(/^(?:Unit\s+\d+|Reference|Init\s+\d+)\s*[:—-]\s*(.+)$/i)
  return m ? m[1].trim() : title
}

function build(): Doc[] {
  const docs: Doc[] = []
  for (const [path, body] of Object.entries(raw)) {
    const file = path.split('/').pop() ?? path
    const stem = file.replace(/\.md$/, '')

    // skip the index README and the gitignored personal spec
    if (stem === 'README' || stem === 'my-gpu-spec') continue

    const unitMatch = stem.match(/unit-(\d+)/)
    const unit = unitMatch ? parseInt(unitMatch[1], 10) : null
    const kind: Doc['kind'] = unit !== null ? 'unit' : 'reference'
    const title = firstH1(body)

    docs.push({
      slug: stem,
      file,
      kind,
      unit,
      title,
      shortTitle: shorten(title),
      tagline: firstEpigraph(body),
      regime: unit !== null ? REGIME[unit] ?? null : null,
      body,
      headings: extractHeadings(body),
    })
  }

  // units in order, then reference docs
  docs.sort((a, b) => {
    if (a.unit !== null && b.unit !== null) return a.unit - b.unit
    if (a.unit !== null) return -1
    if (b.unit !== null) return 1
    return a.title.localeCompare(b.title)
  })
  return docs
}

export const docs = build()
export const units = docs.filter((d) => d.kind === 'unit')
export const references = docs.filter((d) => d.kind === 'reference')

export function getDoc(slug: string): Doc | undefined {
  return docs.find((d) => d.slug === slug)
}

export const regimeLabel: Record<Exclude<Regime, null>, string> = {
  mem: 'memory-bound',
  compute: 'compute-bound',
  mixed: 'capstone',
}
