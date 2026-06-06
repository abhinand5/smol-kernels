import type { ReactNode } from 'react'

/** Slugify heading text into a stable anchor id. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

/** Flatten React children (strings / nested elements) to plain text. */
export function nodeText(node: ReactNode): string {
  if (node == null || node === false) return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(nodeText).join('')
  if (typeof node === 'object' && 'props' in node) {
    // @ts-expect-error -- traversing arbitrary element children
    return nodeText(node.props?.children)
  }
  return ''
}
