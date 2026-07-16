"""
Text-to-SQL tool over the specs database (data/specs.db).

This stands in for the "Cloud APIs" / structured live-data source in the
architecture — an LLM sees the schema, writes a SQL query, we validate and
execute it, and return structured rows (not prose) back to the agent.

Safety guardrails (this is the part that matters if anyone asks "isn't
letting an LLM write SQL dangerous?"):
  1. The connection is opened read-only (SQLite URI mode=ro) — even a
     successfully-injected DROP/DELETE/UPDATE cannot execute against a
     read-only connection at the OS/driver level, not just by convention.
  2. Only SELECT statements are accepted; anything else is rejected before
     execution by a keyword check.
  3. A hard LIMIT is appended if the LLM's query doesn't include one, so a
     malformed query can't return an unbounded result set.
  4. The LLM only ever sees the schema (table/column names), never raw
     credentials or file paths.
"""

import re
import sqlite3
from pathlib import Path

from llm_utils import invoke_with_retry, strip_thinking_tags

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "specs.db"

SCHEMA_DESCRIPTION = """\
Table: gpus
  model TEXT, vram_gb REAL, memory_type TEXT, memory_bandwidth_gbps REAL,
  fp16_tflops_dense REAL, nvlink_gbps REAL, tdp_watts REAL,
  price_tier TEXT ('enterprise' or 'consumer'), notes TEXT, source_url TEXT

Table: ssds
  model TEXT, interface TEXT, capacity_tb REAL, seq_read_mbps REAL,
  seq_write_mbps REAL, endurance_dwpd REAL, form_factor TEXT, notes TEXT,
  source_url TEXT
"""

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|REPLACE)\b",
    re.IGNORECASE,
)


def _get_sample_model_names() -> str:
    """
    Pulls the real `model` values out of the DB and hands them to the LLM.

    Why this exists: the model column holds full strings like
    'NVIDIA H100 80GB SXM', not the short names a user types ('H100 SXM').
    Without seeing the real values, the LLM writes `WHERE model =
    'H100 SXM'` -- a syntactically valid, silently-wrong exact match that
    returns 0 rows even though the row exists. Showing the actual stored
    strings (and instructing LIKE over =) fixes this class of bug instead
    of just hiding it.
    """
    try:
        uri = f"file:{DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        gpu_models = [r[0] for r in conn.execute("SELECT model FROM gpus").fetchall()]
        ssd_models = [r[0] for r in conn.execute("SELECT model FROM ssds").fetchall()]
        conn.close()
        return (
            f"Actual values in gpus.model: {gpu_models}\n"
            f"Actual values in ssds.model: {ssd_models}"
        )
    except sqlite3.Error:
        return ""


class SpecsQueryError(Exception):
    pass


def validate_sql(sql: str) -> str:
    """Raises SpecsQueryError if the query isn't a safe read-only SELECT."""
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise SpecsQueryError("Only SELECT statements are allowed.")
    if _FORBIDDEN.search(stripped):
        raise SpecsQueryError("Query contains a forbidden keyword.")
    if ";" in stripped:
        raise SpecsQueryError("Multiple statements are not allowed.")
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        stripped += " LIMIT 20"
    return stripped


def run_specs_query(sql: str) -> list[dict]:
    """Validates and executes a SQL query against the read-only specs DB."""
    safe_sql = validate_sql(sql)
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(safe_sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def build_sql_prompt(question: str) -> str:
    samples = _get_sample_model_names()
    return (
        "You are a text-to-SQL agent for a small SQLite database of AI "
        "infrastructure hardware specs (GPUs and enterprise SSDs).\n\n"
        f"Schema:\n{SCHEMA_DESCRIPTION}\n"
        f"{samples}\n\n"
        f"User question: {question}\n\n"
        "Write ONE read-only SELECT query that answers this question. "
        "IMPORTANT: the `model` column holds full product name strings "
        "(see the actual values above) -- a user asking about \"H100\" or "
        "\"H100 SXM\" means a SUBSTRING of the real value, not an exact "
        "match. Use `WHERE model LIKE '%H100%SXM%'` (or similar wildcard "
        "matching on the relevant keywords) rather than `WHERE model = "
        "'...'`, unless the question gives you the exact full string. "
        "Respond with ONLY the SQL query, no explanation, no markdown "
        "fences."
    )


def _loosen_exact_match_to_like(sql: str) -> str | None:
    """
    Defense-in-depth fallback: if a query used `model = 'X'` (or `model=
    "X"`) and returned zero rows, rewrite it to match each word of X
    independently (`model LIKE '%word1%' AND model LIKE '%word2%' ...`)
    and retry once.

    Why word-by-word rather than one `LIKE '%X%'` on the whole phrase:
    real model strings interspers extra tokens (e.g. the real value is
    'NVIDIA H100 80GB SXM', not 'H100 SXM') -- a single contiguous
    substring match on 'H100 SXM' still fails because '80GB' sits between
    the words in the actual data. Matching each word separately handles
    that correctly. Prompting alone can't be relied on 100% of the time
    to get this right, so this fallback catches the failure mode
    directly instead of hoping the LLM never regresses to exact-match
    syntax.
    """
    match = re.search(r"model\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
    if not match:
        return None
    literal = match.group(1)
    words = [w for w in re.split(r"\s+", literal.strip()) if w]
    if not words:
        return None
    like_clauses = " AND ".join(f"model LIKE '%{w}%'" for w in words)
    loosened = sql[: match.start()] + f"({like_clauses})" + sql[match.end() :]
    return loosened


def query_specs_db(question: str, llm) -> dict:
    """
    Full text-to-SQL pipeline: LLM writes SQL from the question, we
    validate + execute it, and return both the SQL (for the trace) and
    the resulting rows.

    Includes one automatic retry: if an exact `model = '...'` match
    returns zero rows, it's loosened to a LIKE wildcard and re-run once
    (see _loosen_exact_match_to_like). The trace/sql returned reflects
    whichever query actually produced results.
    """
    prompt = build_sql_prompt(question)
    resp = invoke_with_retry(llm, prompt)
    raw_sql = resp.content if hasattr(resp, "content") else str(resp)
    raw_sql = strip_thinking_tags(raw_sql)
    raw_sql = raw_sql.strip().strip("`").replace("sql\n", "", 1).strip()

    try:
        rows = run_specs_query(raw_sql)
    except (SpecsQueryError, sqlite3.Error) as e:
        return {"sql": raw_sql, "rows": [], "error": str(e)}

    if not rows:
        loosened_sql = _loosen_exact_match_to_like(raw_sql)
        if loosened_sql:
            try:
                retry_rows = run_specs_query(loosened_sql)
                if retry_rows:
                    return {"sql": loosened_sql, "rows": retry_rows, "error": None}
            except (SpecsQueryError, sqlite3.Error):
                pass  # fall through and return the original empty result

    return {"sql": raw_sql, "rows": rows, "error": None}
