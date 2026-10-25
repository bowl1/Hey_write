"""Authenticated workspaces isolated by tenant_id and PostgreSQL RLS."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

import httpx
from fastapi import HTTPException
from psycopg import sql

_tenant = ContextVar("tenant_id", default=None)


def tenant_id() -> str:
    value = _tenant.get()
    if value is None:
        raise RuntimeError("An authenticated tenant context is required")
    return value


def tenant_schema(value: str) -> str:
    """Legacy schema name, retained only for migration tooling and fixtures."""
    return "tenant_" + UUID(value).hex


def configure_tenant_connection(conn) -> None:
    """Bind verified application identity to this transaction, never the session."""
    user = tenant_id()
    conn.execute("SELECT set_config('app.tenant_id', %s, true)", (user,))
    conn.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(
        sql.Identifier("heywrite")))


def validate_runtime_role(conn) -> None:
    """Fail closed rather than silently run shared-table queries without RLS."""
    role = conn.execute("SELECT rolsuper, rolbypassrls FROM pg_catalog.pg_roles WHERE rolname = current_user").fetchone()
    if role["rolsuper"] or role["rolbypassrls"]:
        raise RuntimeError("DATABASE_URL must use a non-superuser role without BYPASSRLS")
    owner = conn.execute("""SELECT 1 FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'heywrite' AND c.relkind = 'r'
          AND pg_has_role(current_user, c.relowner, 'USAGE') LIMIT 1""").fetchone()
    if owner:
        raise RuntimeError("DATABASE_URL must not own shared tables or inherit their owner role")


def secure_tenant_table(conn, table: str) -> None:
    """Deployment-only policy installation for shared tables."""
    target = sql.Identifier("heywrite", table)
    predicate = sql.SQL("tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid")
    conn.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(target))
    conn.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(target))
    for name, mode in (("tenant_access", "PERMISSIVE"), ("tenant_guard", "RESTRICTIVE")):
        conn.execute(sql.SQL("DROP POLICY IF EXISTS {} ON {}").format(sql.Identifier(name), target))
        conn.execute(sql.SQL(
            "CREATE POLICY {} ON {} AS {} FOR ALL TO PUBLIC USING ({}) WITH CHECK ({})"
        ).format(sql.Identifier(name), target, sql.SQL(mode), predicate, predicate))


@contextmanager
def tenant_context(value: str):
    value = str(UUID(value))
    token = _tenant.set(value)
    try:
        yield
    finally:
        _tenant.reset(token)


async def authenticate(authorization: str | None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in to access your workspace", headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "Invalid access token")
    base_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not base_url or not key:
        raise HTTPException(503, "Supabase authentication is not configured")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{base_url}/auth/v1/user",
                headers={"Authorization": f"Bearer {token}", "apikey": key},
            )
        if response.status_code in {400, 401, 403}:
            raise HTTPException(401, "Your session has expired. Please sign in again.",
                                headers={"WWW-Authenticate": "Bearer"})
        response.raise_for_status()
        user = response.json()
        user["id"] = str(UUID(user["id"]))
        if not user.get("email") or not user.get("email_confirmed_at") or user.get("is_anonymous"):
            raise HTTPException(403, "A confirmed email account is required")
        return user
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(503, "Authentication service unavailable") from None


def ensure_workspace() -> None:
    """Check deployment readiness. New accounts require no schema/table creation."""
    from template_store import get_connection
    with get_connection() as conn:
        for table in ("templates", "agent_sessions", "agent_runs", "messages"):
            row = conn.execute("""SELECT c.relrowsecurity AND c.relforcerowsecurity AS secured
                FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'heywrite' AND c.relname = %s AND c.relkind = 'r'""", (table,)).fetchone()
            if not row or not row["secured"]:
                raise RuntimeError("Shared schema is not ready; run migrate_shared.py before serving traffic")
