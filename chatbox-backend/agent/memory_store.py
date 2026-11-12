"""Tenant-scoped memory storage. Model calls never hold a database transaction."""
from psycopg.types.json import Jsonb

from template_store import get_connection
from tenancy import tenant_id


def load_profile() -> dict:
    with get_connection() as conn:
        conn.execute("INSERT INTO memory_profiles DEFAULT VALUES ON CONFLICT (tenant_id) DO NOTHING")
        return conn.execute(
            "SELECT enabled, facts, revision FROM memory_profiles WHERE tenant_id = %s", (tenant_id(),)
        ).fetchone()


def save_profile(facts: list[dict], revision: int) -> bool:
    # Compare-and-swap also prevents an in-flight extraction undoing Clear/Disable.
    with get_connection() as conn:
        return conn.execute(
            """UPDATE memory_profiles SET facts = %s, revision = revision + 1, updated_at = now()
               WHERE tenant_id = %s AND revision = %s AND enabled = true""",
            (Jsonb(facts), tenant_id(), revision),
        ).rowcount == 1


def configure_profile(*, enabled: bool | None = None, clear: bool = False) -> dict:
    load_profile()
    with get_connection() as conn:
        return conn.execute(
            """UPDATE memory_profiles SET enabled = COALESCE(%s, enabled),
               facts = CASE WHEN %s THEN '[]'::jsonb ELSE facts END,
               revision = revision + 1, updated_at = now()
               WHERE tenant_id = %s RETURNING enabled, facts, revision""",
            (enabled, clear, tenant_id()),
        ).fetchone()


def load_summary(session_id: str) -> dict:
    with get_connection() as conn:
        return conn.execute(
            "SELECT summary, through_message_id, through_offset FROM session_summaries WHERE session_id = %s",
            (session_id,),
        ).fetchone() or {"summary": "", "through_message_id": 0, "through_offset": 0}


def summary_batch(session_id: str, cursor: dict, budget: int = 12000) -> tuple[list[dict], int, int]:
    """Read unsummarized text older than the latest 20 messages, without losing long-message tails."""
    message_id, offset = cursor["through_message_id"], cursor["through_offset"]
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT message_id, role, char_length(content) AS length,
                      substring(content FROM CASE WHEN message_id = %s THEN %s + 1 ELSE 1 END FOR %s) AS content
               FROM messages
               WHERE session_id = %s AND message_id >= %s
                 AND (message_id > %s OR char_length(content) > %s)
                 AND message_id < (SELECT min(message_id) FROM
                     (SELECT message_id FROM messages WHERE session_id = %s
                      ORDER BY message_id DESC LIMIT 20) recent)
               ORDER BY message_id ASC LIMIT 40""",
            (message_id, offset, budget, session_id, message_id, message_id, offset, session_id),
        ).fetchall()
    batch = []
    remaining = budget
    for row in rows:
        start = offset if row["message_id"] == message_id else 0
        text = row["content"][:remaining]
        if row["role"] in {"user", "assistant"} and text:
            batch.append({"role": row["role"], "content": text})
        next_offset = start + len(text)
        message_id, offset = row["message_id"], next_offset
        remaining -= len(text)
        if remaining <= 0:
            break
    return batch, message_id, offset


def save_summary(session_id: str, old: dict, summary: str, message_id: int, offset: int) -> bool:
    with get_connection() as conn:
        conn.execute("INSERT INTO session_summaries (session_id) VALUES (%s) ON CONFLICT DO NOTHING", (session_id,))
        return conn.execute(
            """UPDATE session_summaries SET summary = %s, through_message_id = %s,
               through_offset = %s, updated_at = now()
               WHERE session_id = %s AND through_message_id = %s AND through_offset = %s""",
            (summary, message_id, offset, session_id, old["through_message_id"], old["through_offset"]),
        ).rowcount == 1
