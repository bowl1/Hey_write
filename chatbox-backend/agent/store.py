from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from template_store import get_connection, init_db as init_template_db


def init_agent_db() -> None:
    init_template_db()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id text PRIMARY KEY,
                    active_template_id text,
                    current_draft text NOT NULL DEFAULT '',
                    style text NOT NULL DEFAULT 'Formal',
                    language text NOT NULL DEFAULT 'English',
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id text PRIMARY KEY,
                    session_id text NOT NULL,
                    action text NOT NULL,
                    intent text NOT NULL,
                    selected_template_id text,
                    input_state jsonb NOT NULL DEFAULT '{}'::jsonb,
                    output_state jsonb NOT NULL DEFAULT '{}'::jsonb,
                    trace jsonb NOT NULL DEFAULT '[]'::jsonb,
                    reply text NOT NULL DEFAULT '',
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id bigserial PRIMARY KEY,
                    session_id text NOT NULL,
                    role text NOT NULL,
                    content text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS agent_runs_session_idx ON agent_runs (session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id)")


def load_session(session_id: str) -> dict[str, Any] | None:
    init_agent_db()
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT session_id, active_template_id, current_draft, style, language
            FROM agent_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        ).fetchone()


def persist_agent_state(state: dict[str, Any]) -> None:
    init_agent_db()
    session_id = state["session_id"]
    template_meta = state.get("template_meta") or {}
    active_template_id = (
        template_meta.get("selected_template_id")
        or state.get("active_template_id")
    )
    current_draft = state.get("reply") or state.get("current_draft") or ""
    input_state = {
        "action": state.get("action"),
        "intent": state.get("intent"),
        "style": state.get("style"),
        "language": state.get("language"),
        "active_template_id": state.get("active_template_id"),
    }
    output_state = {
        "active_template_id": active_template_id,
        "current_draft": current_draft,
        "template_meta": template_meta,
        "evaluation": state.get("evaluation") or {},
    }

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_sessions (
                session_id, active_template_id, current_draft, style, language
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                active_template_id = EXCLUDED.active_template_id,
                current_draft = EXCLUDED.current_draft,
                style = EXCLUDED.style,
                language = EXCLUDED.language,
                updated_at = now()
            """,
            (
                session_id,
                active_template_id,
                current_draft,
                state.get("style") or "Formal",
                state.get("language") or "English",
            ),
        )
        conn.execute(
            """
            INSERT INTO agent_runs (
                run_id, session_id, action, intent, selected_template_id,
                input_state, output_state, trace, reply
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                state["run_id"],
                session_id,
                state.get("action"),
                state.get("intent"),
                active_template_id,
                Jsonb(input_state),
                Jsonb(output_state),
                Jsonb(state.get("trace") or []),
                current_draft,
            ),
        )
        if state.get("intent"):
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, "user", state["intent"]),
            )
        if current_draft:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, "assistant", current_draft),
            )
