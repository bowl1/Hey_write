"""Shared-table DDL. Called only by the administrative migration command."""
from contextlib import nullcontext
from template_store import EMBEDDING_DIMENSION
from tenancy import secure_tenant_table


def init_templates(connection) -> None:
    with nullcontext(connection) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS templates (
                    tenant_id uuid NOT NULL DEFAULT nullif(current_setting('app.tenant_id', true), '')::uuid,
                    id text NOT NULL,
                    PRIMARY KEY (tenant_id, id),
                    title text NOT NULL,
                    category text NOT NULL DEFAULT 'general',
                    description text NOT NULL DEFAULT '',
                    tags text[] NOT NULL DEFAULT '{{}}',
                    style text NOT NULL DEFAULT 'Formal',
                    language text NOT NULL DEFAULT 'English',
                    use_cases jsonb NOT NULL DEFAULT '[]'::jsonb,
                    structure jsonb NOT NULL DEFAULT '[]'::jsonb,
                    content text NOT NULL DEFAULT '',
                    enabled boolean NOT NULL DEFAULT true,
                    version integer NOT NULL DEFAULT 1,
                    embedding public.vector({EMBEDDING_DIMENSION}),
                    embedding_text text,
                    embedding_model text,
                    embedding_updated_at timestamptz,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS templates_enabled_idx ON templates (tenant_id, enabled)")
            secure_tenant_table(conn, "templates")
            cur.execute("CREATE INDEX IF NOT EXISTS templates_category_idx ON templates (tenant_id, category)")
            cur.execute("CREATE INDEX IF NOT EXISTS templates_tags_idx ON templates USING gin (tags)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS templates_embedding_idx
                ON templates USING ivfflat (embedding public.vector_cosine_ops)
                WITH (lists = 100)
                """
            )


def init_tables(connection) -> None:
    init_templates(connection)
    with nullcontext(connection) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    tenant_id uuid NOT NULL DEFAULT nullif(current_setting('app.tenant_id', true), '')::uuid,
                    session_id text NOT NULL,
                    PRIMARY KEY (tenant_id, session_id),
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
                    tenant_id uuid NOT NULL DEFAULT nullif(current_setting('app.tenant_id', true), '')::uuid,
                    run_id text NOT NULL,
                    PRIMARY KEY (tenant_id, run_id),
                    session_id text NOT NULL,
                    FOREIGN KEY (tenant_id, session_id) REFERENCES agent_sessions (tenant_id, session_id),
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
                    tenant_id uuid NOT NULL DEFAULT nullif(current_setting('app.tenant_id', true), '')::uuid,
                    message_id bigserial NOT NULL,
                    PRIMARY KEY (tenant_id, message_id),
                    session_id text NOT NULL,
                    FOREIGN KEY (tenant_id, session_id) REFERENCES agent_sessions (tenant_id, session_id),
                    role text NOT NULL,
                    content text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS agent_runs_session_idx ON agent_runs (tenant_id, session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (tenant_id, session_id)")
            for table in ("agent_sessions", "agent_runs", "messages"):
                secure_tenant_table(conn, table)

