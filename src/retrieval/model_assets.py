"""
Fallback source for the Qwen3-Embedding-0.6B weights: a GitHub Release
asset on this project's own repo, instead of Hugging Face's Hub API.

Why this exists: reproduced live on Streamlit Community Cloud -- a fresh
deploy container has no local model cache, so it must fetch the model
over the network on cold start. huggingface_hub's snapshot_download
resolves each file (config.json, tokenizer.json, the weights file, ...)
individually, each with its own HEAD-request etag check and 5-attempt
retry/backoff -- during a genuine Hugging Face outage (also confirmed
live, https://status.huggingface.co), this turned into a multi-minute
retry storm before failing outright, with the app appearing to hang.

The weights themselves are unchanged -- this only changes where they're
fetched from. Bundling them directly into git wasn't practical (the
model is ~1.15GB; GitHub's free Git LFS tier caps out at 1GB/month
bandwidth, which a single redeploy would nearly exhaust), but GitHub
Release assets support files up to 2GB and aren't subject to that LFS
quota, so the same repo you already trust for code can host this too --
one plain HTTPS GET instead of many small etag-checked requests, and no
dependency on huggingface.co being reachable at all.

This is only reached when the model isn't already cached locally (see
Qwen3Embeddings.__init__): local dev machines that already have the
model in their Hugging Face cache never hit this path.
"""

import tarfile
import tempfile
from pathlib import Path

import requests

MODEL_ASSET_URL = (
    "https://github.com/Eiqbal25/agentic-rag/releases/download/"
    "model-assets-v1/qwen3-embedding-0.6b.tar.gz"
)
_CACHE_DIR = Path(tempfile.gettempdir()) / "agentic-rag-models" / "qwen3-embedding-0.6b"


def get_local_model_path() -> Path:
    """
    Returns a local directory containing the Qwen3-Embedding-0.6B model
    files (the same layout SentenceTransformer expects for a local-path
    load), downloading and extracting the GitHub Release asset on first
    call if not already present. Idempotent: once config.json exists in
    the cache dir, later calls (including a later run within the same
    container) return immediately with no network call.
    """
    marker = _CACHE_DIR / "config.json"
    if marker.exists():
        return _CACHE_DIR

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(MODEL_ASSET_URL, stream=True, timeout=(10, 60))
    response.raise_for_status()

    fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "wb") as tmp_file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                tmp_file.write(chunk)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(_CACHE_DIR)
    finally:
        tmp_path.unlink(missing_ok=True)

    return _CACHE_DIR
