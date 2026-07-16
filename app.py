"""
Streamlit demo: Agentic RAG (single agent + tools) over AI infrastructure
knowledge -- documents, a structured hardware specs DB, and live web search.

Run with:
    uv sync
    cp .env.example .env   # fill in GROQ_API_KEY and TAVILY_API_KEY
    uv run streamlit run app.py

Thin entry point -- the actual UI lives in ui/ (one module per page,
plus shared sidebar/resources/styles/config), and the agent/retrieval/
tools logic lives in src/. This file's only job is to get `src/` onto
sys.path before anything under ui/ imports from it, then hand off to
ui.app.main().
"""

# MUST run before anything else imports chromadb (transitively, via
# ui.app -> ... -> retrieval -> langchain_chroma) -- a well-known,
# frequently-hit deployment blocker: Streamlit Community Cloud's base
# image ships a system libsqlite3 older than chromadb requires, causing
# an obscure crash on first deploy ("sqlite3.OperationalError" /
# unsupported version) that has nothing to do with this project's own
# code. The standard fix is swapping in the pysqlite3-binary wheel
# (bundles its own modern SQLite) before chromadb ever imports the
# stdlib sqlite3 module. Guarded in try/except and only installed on
# Linux (see requirements.txt/pyproject.toml's `; sys_platform ==
# "linux"` marker) -- a no-op on local Windows dev, where the system
# sqlite3 is already new enough and no Windows wheel for
# pysqlite3-binary even exists.
try:
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules["pysqlite3"]
except ImportError:
    pass

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
