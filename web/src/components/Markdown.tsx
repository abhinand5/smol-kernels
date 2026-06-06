import type { ComponentPropsWithoutRef, ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Link } from 'react-router-dom'
import { slugify, nodeText } from '../lib/slug'

function HeadingAnchor({ level, children }: { level: 2 | 3; children: ReactNode }) {
  const id = slugify(nodeText(children))
  const Tag = `h${level}` as 'h2' | 'h3'
  return (
    <Tag id={id}>
      {children}
      <a className="anchor" href={`#${id}`} aria-hidden="true">
        §
      </a>
    </Tag>
  )
}

function MdLink({ href = '', children }: ComponentPropsWithoutRef<'a'>) {
  // external
  if (/^https?:\/\//.test(href)) {
    return (
      <a href={href} target="_blank" rel="noreferrer noopener">
        {children}
      </a>
    )
  }
  // cross-doc links to other markdown files -> in-app routes
  if (href.endsWith('.md')) {
    const stem = href.replace(/\.md$/, '').split('/').pop() ?? ''
    const to = stem === 'README' ? '/' : `/u/${stem}`
    return <Link to={to}>{children}</Link>
  }
  // in-page anchors and everything else
  return <a href={href}>{children}</a>
}

const components: Components = {
  h2: ({ children }) => <HeadingAnchor level={2}>{children}</HeadingAnchor>,
  h3: ({ children }) => <HeadingAnchor level={3}>{children}</HeadingAnchor>,
  a: MdLink,
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
}

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
      components={components}
    >
      {children}
    </ReactMarkdown>
  )
}
