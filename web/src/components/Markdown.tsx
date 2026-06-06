import { useState, type ComponentPropsWithoutRef, type ReactNode } from 'react'
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

const CopyIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
)
const CheckIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // fallback for non-secure contexts / older browsers
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try {
        document.execCommand('copy')
      } catch {
        /* give up silently */
      }
      ta.remove()
    }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }
  return (
    <button
      type="button"
      onClick={copy}
      className={`copy-btn${copied ? ' is-copied' : ''}`}
      aria-label={copied ? 'Copied to clipboard' : 'Copy code'}
      title={copied ? 'Copied' : 'Copy'}
    >
      {copied ? <CheckIcon /> : <CopyIcon />}
    </button>
  )
}

function Pre({ children }: ComponentPropsWithoutRef<'pre'>) {
  // children is the highlighted <code> element; flatten it back to source text
  const text = nodeText(children).replace(/\n$/, '')
  return (
    <div className="code-block">
      <CopyButton text={text} />
      <pre>{children}</pre>
    </div>
  )
}

const components: Components = {
  h2: ({ children }) => <HeadingAnchor level={2}>{children}</HeadingAnchor>,
  h3: ({ children }) => <HeadingAnchor level={3}>{children}</HeadingAnchor>,
  a: MdLink,
  pre: Pre,
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
