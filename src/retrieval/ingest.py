"""
Ingestion pipeline: load corpus -> chunk -> embed -> persist Chroma vector
store, and build a parallel BM25 index for hybrid retrieval.

Run once (or whenever the corpus changes):
    python src/retrieval/ingest.py
"""

import pickle
import re
import sys
from pathlib import Path

# Absolute import (not `from .embeddings import ...`) + this sys.path
# bootstrap, specifically so this file stays runnable directly
# (`python src/retrieval/ingest.py`, the command README's setup docs
# use) in addition to being imported normally as `retrieval.ingest` --
# a relative import would raise "attempted relative import with no
# known parent package" when the file is executed directly rather than
# imported, since Python doesn't treat a directly-run script as part of
# its package. Idempotent/harmless when already imported normally (src/
# is already on sys.path by then from app.py's own setup).
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from retrieval.embeddings import Qwen3Embeddings

# Needed here (not just in app.py/agent/__init__.py) so SSL_CERT_FILE/
# REQUESTS_CA_BUNDLE from .env are applied when this module downloads
# the embedding model weights on a standalone ingest run, not only when
# it's imported from within the Streamlit app (which already calls
# load_dotenv() itself before anything else runs).
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = str(ROOT / "data" / "chroma_db")
BM25_PATH = ROOT / "data" / "bm25_index.pkl"

COLLECTION_NAME = "maistorage_ai_infra"


def load_corpus() -> list[Document]:
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # first '# ' line is the doc title
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1) if title_match else path.stem
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "title": title},
            )
        )
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # attach a stable chunk id + section heading (best-effort) as metadata
    for i, c in enumerate(chunks):
        heading_match = re.search(r"^##?\s+(.+)$", c.page_content, re.MULTILINE)
        c.metadata["section"] = heading_match.group(1) if heading_match else ""
        c.metadata["chunk_id"] = f"{c.metadata['source']}::{i}"
    return chunks


# Minimal English stopword list. BM25 on a small corpus is sensitive to
# high-frequency function words getting non-trivial IDF weight, which can
# drown out the rare content terms that should actually drive ranking.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "to", "from", "in", "on", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "can", "may", "might",
    "must", "this", "that", "these", "those", "it", "its", "how", "what",
    "when", "where", "why", "which", "who", "whom", "i", "you", "he",
    "she", "we", "they", "not", "no", "than", "too", "very", "s", "t",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def build_indexes():
    print(f"Loading corpus from {CORPUS_DIR} ...")
    docs = load_corpus()
    print(f"  {len(docs)} source documents")

    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks after splitting")

    print("Loading Qwen3-Embedding-0.6B (downloads once via Hugging Face, "
          "cached locally after) ...")
    embeddings = Qwen3Embeddings()

    print(f"Building Chroma vector store at {CHROMA_DIR} ...")
    # Deliberately NOT shutil.rmtree(CHROMA_DIR) then recreate -- confirmed
    # live on Windows: if any live process (e.g. this same app, mid-session,
    # reindexing after a document add/delete) already holds a Chroma
    # connection open against this path, its chroma.sqlite3 file handle is
    # still open, and Windows (unlike POSIX) refuses to delete an open file
    # -- `PermissionError: [WinError 32] ... used by another process`,
    # silently aborting the reindex server-side. Clearing every Python
    # reference and forcing gc.collect() before the rmtree did NOT fix
    # this: chromadb keeps its own internal client registry keyed by
    # persist_directory (to avoid duplicate connections to the same path),
    # so the file stays open regardless of what this module's own
    # references are doing. Going through Chroma's delete_collection() API
    # instead reuses that already-open connection to drop and recreate the
    # collection's data -- no filesystem-level delete of an open file,
    # so it works whether or not another live connection to this same
    # path already exists.
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    try:
        vectordb.delete_collection()
    except Exception:
        pass  # nothing to delete yet (first-ever ingest run)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Vector store persisted with {vectordb._collection.count()} vectors")

    print("Building BM25 sparse index ...")
    tokenized_corpus = [tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"  BM25 index persisted to {BM25_PATH}")

    print("Ingestion complete.")


if __name__ == "__main__":
    build_indexes()
