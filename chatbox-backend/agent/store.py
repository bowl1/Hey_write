from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from template_store import get_connection

from .schemas import AgentRunRecord, AgentSession, SessionMessage, SessionSnapshot, SessionSummary


def load_session(session_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT session_id, active_template_id, current_draft, style, language
            FROM agent_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        ).fetchone()


def load_session_snapshot(session_id: str) -> SessionSnapshot | None:
    with get_connection() as conn:
        session = conn.execute(
            """
            SELECT session_id, active_template_id, current_draft, style, language,
                   created_at, updated_at
            FROM agent_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        ).fetchone()
        if not session:
            return None

        messages = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at ASC, message_id ASC
            """,
            (session_id,),
        ).fetchall()
        last_run = conn.execute(
            """
            SELECT run_id, action, intent, selected_template_id,
                   output_state, trace, reply, created_at
            FROM agent_runs
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()

    return SessionSnapshot(
        session=AgentSession.model_validate(session),
        messages=[SessionMessage.model_validate(message) for message in messages],
        last_run=AgentRunRecord.model_validate(last_run) if last_run else None,
    )


def list_session_summaries(limit: int = 20) -> list[SessionSummary]:
    with get_connection() as conn:
        sessions = conn.execute(
            """
            SELECT session_id, active_template_id, current_draft, style, language,
                   created_at, updated_at
            FROM agent_sessions
            WHERE session_id NOT LIKE 'eval-%%'
              AND session_id NOT LIKE 'test-%%'
              AND session_id NOT LIKE 'smoke-%%'
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()

        summaries = []
        for session in sessions:
            messages = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC, message_id ASC
                """,
                (session["session_id"],),
            ).fetchall()
            first_user_message = next(
                (message["content"] for message in messages if message["role"] == "user"),
                "",
            )
            title = first_user_message or session.get("current_draft") or "Untitled session"
            summaries.append(
                SessionSummary.model_validate(
                    {
                        **session,
                        "title": title[:80],
                        "message_count": len(messages),
                        "messages": messages,
                    }
                )
            )
    return summaries


def load_session_history(session_id: str, limit: int = 20) -> list[dict[str, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT message_id, role, content FROM messages
                WHERE session_id = %s
                ORDER BY message_id DESC LIMIT %s
            ) recent ORDER BY message_id ASC
            """,
            (session_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def persist_agent_state(state: dict[str, Any]) -> dict[str, Any]:
    session_id = state["session_id"]
    template_meta = state.get("template_meta") or {}
    active_template_id = (
        template_meta.get("selected_template_id")
        if template_meta.get("used_template")
        else None
    ) or (
        state.get("active_template_id")
        if not state.get("blocked_reason")
        else None
    )
    candidate_draft = state.get("reply") or ""
    accepted = (
        (state.get("evaluation") or {}).get("passed") is True
        and not state.get("blocked_reason")
        and bool(candidate_draft.strip())
    )
    input_state = {
        "action": state.get("action"),
        "intent": state.get("intent"),
        "style": state.get("style"),
        "language": state.get("language"),
        "active_template_id": state.get("active_template_id"),
    }
    with get_connection() as conn:
        saved = conn.execute(
            """
            INSERT INTO agent_sessions (
                session_id, active_template_id, current_draft, style, language
            )
            VALUES (%(session_id)s, %(template_id)s, %(draft)s, %(style)s, %(language)s)
            ON CONFLICT (tenant_id, session_id) DO UPDATE SET
                active_template_id = CASE WHEN %(accepted)s THEN EXCLUDED.active_template_id ELSE agent_sessions.active_template_id END,
                current_draft = CASE WHEN %(accepted)s THEN EXCLUDED.current_draft ELSE agent_sessions.current_draft END,
                style = CASE WHEN %(accepted)s THEN EXCLUDED.style ELSE agent_sessions.style END,
                language = CASE WHEN %(accepted)s THEN EXCLUDED.language ELSE agent_sessions.language END,
                updated_at = now()
            RETURNING active_template_id, current_draft
            """,
            {
                "session_id": session_id,
                "template_id": active_template_id if accepted else None,
                "draft": candidate_draft if accepted else "",
                "style": state.get("style") or "Formal",
                "language": state.get("language") or "English",
                "accepted": accepted,
            },
        ).fetchone()
        output_state = {
            **saved,
            "draft_accepted": accepted,
            "candidate_draft": candidate_draft,
            "template_meta": template_meta,
            "evaluation": state.get("evaluation") or {},
            "blocked_reason": state.get("blocked_reason"),
        }
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
                candidate_draft,
            ),
        )
        if state.get("intent"):
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, "user", state["intent"]),
            )
        if accepted:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, "assistant", candidate_draft),
            )
    return {**saved, "draft_accepted": accepted}
