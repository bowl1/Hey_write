"""Authenticated account workspaces, isolated in PostgreSQL schemas."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

import httpx
import psycopg
from fastapi import HTTPException
from psycopg import sql
from psycopg.rows import dict_row

_tenant = ContextVar("tenant_id", default=None)


def tenant_id() -> str:
    value = _tenant.get()
    if value is None:
        raise RuntimeError("An authenticated tenant context is required")
    return value


def tenant_schema(value: str) -> str:
    return "tenant_" + UUID(value).hex


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
    # A transaction lock serializes first-use provisioning for the same account.
    # No public fallback is included: absent tenant tables cannot reveal legacy data.
    from template_store import get_database_url
    from agent.store import init_agent_db

    schema = tenant_schema(tenant_id())
    with psycopg.connect(get_database_url(), row_factory=dict_row) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (schema,))
        exists = conn.execute("SELECT to_regclass(%s) AS workspace", (f"{schema}.messages",)).fetchone()["workspace"]
        if exists:
            return
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(sql.Identifier(schema)))
        init_agent_db(connection=conn)
