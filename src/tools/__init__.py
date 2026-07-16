"""
The `specs` and `web` tools: text-to-SQL over the hardware specs
database, and live web search via Tavily.
"""

from .specs_tool import SpecsQueryError, query_specs_db, run_specs_query, validate_sql
from .web_tool import web_search

__all__ = [
    "query_specs_db",
    "run_specs_query",
    "validate_sql",
    "SpecsQueryError",
    "web_search",
]
