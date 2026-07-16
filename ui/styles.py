"""Shared visual constants and CSS used across multiple pages."""

import html as _html

TOOL_ICONS = {"docs": "📚", "specs": "🗄️", "web": "🌐"}
CITATION_ICONS = {"document": "📄", "specs_db": "🗄️", "web": "🌐"}
TOOL_COLORS = {"docs": "#3B82F6", "specs": "#8B5CF6", "web": "#10B981"}
CITATION_TYPE_TO_TOOL = {"document": "docs", "specs_db": "specs", "web": "web"}

CUSTOM_CSS = """
<style>
.citation-card {
    border-left: 4px solid var(--accent-color, #999);
    background: rgba(128, 128, 128, 0.06);
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
}
.citation-card .citation-source { font-weight: 600; font-size: 1.05rem; }
.citation-card .citation-meta { font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.25rem; }
.tool-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    margin-right: 0.3rem;
    text-transform: capitalize;
}
.trace-step { font-family: monospace; font-size: 0.8rem; padding: 0.2rem 0; }
</style>
"""


def escape_html(text) -> str:
    """
    HTML-escapes untrusted text before interpolating it into raw HTML
    strings passed to st.markdown(..., unsafe_allow_html=True).

    Citation data is NOT trusted: web citations carry a page title/URL/
    snippet straight from Tavily search results (fully attacker-
    controlled if a malicious page gets indexed and returned for a
    query), and document citations carry a section heading/snippet
    pulled from corpus files -- which, since the Documents tab lets
    users add new .md files, could contain a deliberately crafted
    `<script>` tag. Confirmed exploitable prior to this fix: interpolating
    either straight into citation-card HTML executes it in the viewer's
    browser the moment that source gets cited in an answer.
    """
    return _html.escape(str(text), quote=True)


def safe_url(url: str) -> str:
    """
    Returns `url` (HTML-escaped) only if it's an http(s) URL, else "".

    Citation URLs are equally untrusted (see escape_html) -- rendering
    an arbitrary scheme as `<a href="...">` would let a `javascript:` or
    `data:` URI execute when clicked, which plain HTML-escaping alone
    does not prevent (escaping stops attribute/tag breakout, not a
    dangerous scheme inside an otherwise well-formed href).
    """
    url = (url or "").strip()
    if url.lower().startswith(("http://", "https://")):
        return _html.escape(url, quote=True)
    return ""
