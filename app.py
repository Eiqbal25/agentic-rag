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

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
