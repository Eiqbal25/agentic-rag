"""
Dense embedding models for the `docs` retrieval tool.

Qwen3Embeddings (below) is the live default: a real pretrained
transformer bi-encoder (Qwen/Qwen3-Embedding-0.6B, 1024-dim, via
sentence-transformers), running on CPU. LSAEmbeddings (TF-IDF ->
Truncated SVD, i.e. classical Latent Semantic Analysis) is kept
alongside it, unused by the live pipeline, as a self-contained
zero-download fallback -- it was the original default back when this
project was built in a sandbox whose network egress allowlist didn't
include huggingface.co. That constraint doesn't hold in this
environment (verified live: huggingface.co and pypi.org are both
reachable once Python trusts AVG Antivirus's HTTPS-inspection
certificate -- see SSL_CERT_FILE / REQUESTS_CA_BUNDLE in .env.example),
so the real embedding model is no longer blocked and is what actually
runs retrieval now. LSAEmbeddings' own tests (tests/test_embeddings.py)
are untouched and still pass -- nothing about swapping which class
retrieval.py/ingest.py import required deleting the other.
"""

import pickle
from pathlib import Path

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class Qwen3Embeddings(Embeddings):
    """
    LangChain-compatible Embeddings implementation backed by a
    pretrained Qwen3-Embedding-0.6B bi-encoder. Unlike LSAEmbeddings,
    there's no corpus-specific fit/save/load step -- the weights are
    pretrained and downloaded once via sentence-transformers (cached
    under ~/.cache/huggingface/hub afterward, so every run after the
    first just loads from local disk).

    IMPORTANT asymmetry: Qwen3-Embedding models are trained with a
    different encoding path for queries vs. documents -- queries get an
    instruction prefix baked in via sentence-transformers'
    prompt_name="query" mechanism ("Instruct: Given a web search query,
    retrieve relevant passages that answer the query\\nQuery: ..."),
    documents are embedded plain. Getting this backwards doesn't raise
    an error, it just quietly degrades ranking quality -- unlike
    LSAEmbeddings, where embed_query and embed_documents were the exact
    same code path, these two methods here are NOT interchangeable.

    OFFLINE-FIRST LOADING: reproduced live -- even with the model
    already fully cached locally, sentence-transformers/huggingface_hub
    still does an online metadata check by default before falling back
    to cache, and that network round-trip through AVG Antivirus's
    HTTPS-inspecting proxy is fragile (a `RuntimeError: Cannot send a
    request, as the client has been closed` was reproduced from
    huggingface_hub's httpx-based retry/backoff wrapper reusing a client
    that a prior SSL failure had already torn down -- a confusing
    secondary error masking the real one). Since the whole point of
    caching is to not need the network on every app start, this passes
    local_files_only=True (cache-only, no network call at all) first.

    NOTE: setting the HF_HUB_OFFLINE env var instead of this constructor
    argument does NOT work here -- confirmed live -- because
    huggingface_hub reads that env var into a module-level constant at
    IMPORT time (`from sentence_transformers import SentenceTransformer`
    at the top of this file already triggered that import), so setting
    it later inside __init__ has no effect. local_files_only is a
    genuine per-call argument that huggingface_hub checks dynamically,
    not a snapshotted one.

    FALLBACK: if the model genuinely isn't cached locally (e.g. a fresh
    Streamlit Community Cloud container), this does NOT fall back to a
    live Hugging Face Hub download -- reproduced live, a real HF outage
    turned that into a multi-minute retry storm (see model_assets.py's
    docstring for the full story). Instead it downloads the same weights
    from this project's own GitHub Release asset (one plain HTTPS GET,
    no dependency on huggingface.co) and loads from the extracted local
    path.
    """

    def __init__(self, model_name: str = QWEN3_EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            from .model_assets import get_local_model_path

            local_path = get_local_model_path()
            self.model = SentenceTransformer(str(local_path), device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([text], prompt_name="query", normalize_embeddings=True)
        return vec[0].tolist()


class LSAEmbeddings(Embeddings):
    """LangChain-compatible Embeddings implementation backed by a fitted
    TF-IDF + TruncatedSVD pipeline."""

    def __init__(self, n_components: int = 200):
        self.n_components = n_components
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LSAEmbeddings":
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
        )
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # On a very small or highly homogeneous corpus, max_df=0.9
            # can prune every term shared across documents down to zero
            # vocabulary (e.g. 2 near-identical short documents), which
            # sklearn raises as "no terms remain" rather than degrading
            # gracefully. Retrying without max_df pruning trades away a
            # little noise-reduction for not crashing outright -- the
            # right tradeoff for a small/edge-case corpus where losing a
            # few common terms isn't worth losing the whole embedding.
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        # n_components must be < n_features and < n_samples
        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1, len(texts) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_matrix)
        return self

    def _embed(self, texts: list[str]):
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        vecs = normalize(vecs)  # so cosine similarity == dot product
        return vecs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0].tolist()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    @classmethod
    def load(cls, path: str) -> "LSAEmbeddings":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.svd = data["svd"]
        return obj
