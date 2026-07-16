"""Documents page: browser + editor for the document corpus, and a
read-only summary of the specs database."""

import gc
import re
import shutil
import sqlite3
from pathlib import Path

import streamlit as st

from retrieval.ingest import build_indexes

from .resources import get_app, get_retriever

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
DELETED_DIR = CORPUS_DIR / "_deleted"
SPECS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "specs.db"

_SAFE_MD_FILENAME = re.compile(r"^[A-Za-z0-9_\-]+\.md$")


def _sanitize_md_filename(name: str) -> str | None:
    """
    Strips any directory components (Path(name).name) and rejects
    anything that isn't a plain `word-chars.md` filename -- both the
    manual filename field and an uploaded file's original name are
    user-controlled strings, so this is the one thing standing between
    a crafted name (e.g. "../../../etc/passwd" or a name with no .md
    extension) and a write outside CORPUS_DIR. Returns None if unsafe.
    """
    name = Path(name).name
    return name if _SAFE_MD_FILENAME.match(name) else None


def _reindex_and_refresh():
    """
    Re-runs the full ingest pipeline (chunk -> embed -> Chroma + BM25)
    against whatever is currently in data/corpus/, then clears every
    cache that could still be holding a reference to the PRE-reindex
    data -- get_app's cached graph closure captured a HybridRetriever
    built from the old on-disk index at construction time, so
    get_retriever and get_app both have to be cleared or the agent
    keeps searching stale data after a UI-visible "added/deleted"
    message, which is exactly the inconsistency a user would notice the
    moment a just-added document doesn't show up in citations.

    NOTE: retrieval.ingest.build_indexes() itself no longer deletes
    data/chroma_db/ at the filesystem level (see its docstring)
    specifically because main() calls get_app() unconditionally every
    rerun, so a live Chroma/SQLite connection to that directory already
    exists by the time this runs -- a filesystem-level delete-then-
    recreate crashed with `PermissionError: [WinError 32] ... used by
    another process` on Windows, confirmed live, since Windows won't
    delete a file another handle in the same process still has open.
    gc.collect() here is still worth keeping: it's what actually lets
    the OLD HybridRetriever (and its Chroma connection) get finalized
    promptly once its last reference is dropped, rather than leaving
    that to whenever the collector next runs on its own.
    """
    with st.spinner("Reindexing corpus (chunking, embedding, rebuilding Chroma + BM25)..."):
        build_indexes()
    _load_corpus_summary.clear()
    get_retriever.clear()
    get_app.clear()
    gc.collect()
    st.rerun()


@st.cache_data(show_spinner=False)
def _load_corpus_summary() -> list[dict]:
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_line = next((line for line in text.splitlines() if line.strip().startswith("#")), "")
        title = title_line.lstrip("#").strip() or path.stem
        docs.append({"filename": path.name, "title": title, "words": len(text.split())})
    return docs


@st.cache_data(show_spinner=False)
def _load_specs_db_summary() -> dict[str, list[str]]:
    if not SPECS_DB_PATH.exists():
        return {}

    uri = f"file:{SPECS_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    summary = {}
    for table in ("gpus", "ssds"):
        try:
            rows = conn.execute(f"SELECT model FROM {table}").fetchall()
            summary[table] = [r[0] for r in rows]
        except sqlite3.Error:
            summary[table] = []
    conn.close()
    return summary


def render_documents_tab():
    """
    Browser + editor over the document corpus the `docs` tool searches
    -- view a document's exact raw content (so a cited snippet can be
    checked against the real source instead of trusted blindly), add a
    new one, or soft-delete one, each followed by an automatic full
    reindex so the agent's retrieval reflects the change immediately
    rather than silently searching stale data. The specs DB stays
    read-only browsing only (built offline by data/build_specs_db.py,
    out of scope for this editor -- see README).
    """
    st.subheader("📚 Document knowledge base")
    st.caption("The `docs` tool searches these files — hybrid dense+BM25 retrieval, LLM-reranked.")
    docs = _load_corpus_summary()
    if docs:
        st.markdown(f"**{len(docs)} documents, {sum(d['words'] for d in docs):,} words total**")
        st.dataframe(
            [{"File": d["filename"], "Title": d["title"], "Words": d["words"]} for d in docs],
            hide_index=True,
        )
    else:
        st.info("No corpus documents found — add one below, or run `uv run python src/retrieval/ingest.py`.")

    st.divider()
    st.subheader("🔍 View or delete a document")
    st.caption(
        "Raw file content exactly as the doc tool sees it before chunking -- "
        "check this against a citation's snippet if something looks off."
    )
    if docs:
        filenames = [d["filename"] for d in docs]
        selected = st.selectbox("Choose a document", filenames, key="doc_view_select")
        if selected:
            content = (CORPUS_DIR / selected).read_text(encoding="utf-8")
            st.code(content, language="markdown", height=300)

            if len(docs) <= 1:
                st.info(
                    "This is the last remaining document -- deleting it would leave "
                    "an empty corpus, which the embedding/indexing step can't fit "
                    "against. Add a replacement document first."
                )
            else:
                confirm_delete = st.checkbox(
                    f"I understand this removes `{selected}` from the searchable corpus "
                    "(it moves to Recently deleted below, not permanently erased)",
                    key=f"confirm_delete_{selected}",
                )
                if st.button("🗑️ Delete this document", disabled=not confirm_delete):
                    DELETED_DIR.mkdir(exist_ok=True)
                    shutil.move(str(CORPUS_DIR / selected), str(DELETED_DIR / selected))
                    st.success(f"Moved `{selected}` to Recently deleted.")
                    _reindex_and_refresh()
    else:
        st.caption("No documents to view or delete yet.")

    deleted_files = sorted(DELETED_DIR.glob("*.md")) if DELETED_DIR.exists() else []
    if deleted_files:
        with st.expander(f"🗑️ Recently deleted ({len(deleted_files)}) — recoverable"):
            for f in deleted_files:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"`{f.name}`")
                if col2.button("Restore", key=f"restore_{f.name}"):
                    shutil.move(str(f), str(CORPUS_DIR / f.name))
                    st.success(f"Restored `{f.name}`.")
                    _reindex_and_refresh()

    st.divider()
    st.subheader("➕ Add a document")
    st.caption("Start the content with `# Document Title` as the first line, same as the existing corpus files.")
    with st.form("add_document_form", clear_on_submit=True):
        upload = st.file_uploader("Upload a .md file", type=["md"])
        st.caption("— or write one directly —")
        new_filename = st.text_input("Filename (e.g. 13_new_topic.md)")
        new_content = st.text_area("Markdown content", height=200)
        submitted = st.form_submit_button("Add & reindex")

    if submitted:
        if upload is not None:
            raw_name, content = upload.name, upload.read().decode("utf-8")
        elif new_filename.strip() and new_content.strip():
            raw_name = new_filename if new_filename.endswith(".md") else f"{new_filename}.md"
            content = new_content
        else:
            st.error("Upload a .md file, or fill in both filename and content.")
            raw_name = None

        if raw_name:
            filename = _sanitize_md_filename(raw_name)
            if filename is None:
                st.error(
                    "Invalid filename -- use only letters, numbers, underscores, "
                    "and hyphens, ending in `.md`."
                )
            elif (CORPUS_DIR / filename).exists():
                st.error(f"`{filename}` already exists in the corpus.")
            elif not content.strip():
                st.error("Document content is empty.")
            else:
                (CORPUS_DIR / filename).write_text(content, encoding="utf-8")
                st.success(f"Added `{filename}`.")
                _reindex_and_refresh()

    st.divider()
    st.subheader("🗄️ Hardware specs database")
    st.caption("The `specs` tool runs text-to-SQL against these tables.")
    summary = _load_specs_db_summary()
    if summary:
        col1, col2 = st.columns(2)
        col1.metric("GPU models", len(summary.get("gpus", [])))
        col2.metric("SSD models", len(summary.get("ssds", [])))
        with st.expander("GPU models in database"):
            for m in summary.get("gpus", []):
                st.markdown(f"- {m}")
        with st.expander("SSD models in database"):
            for m in summary.get("ssds", []):
                st.markdown(f"- {m}")
    else:
        st.info("No specs database found — run `uv run python data/build_specs_db.py` first.")
