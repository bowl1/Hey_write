"""Administrative workspace provisioning and explicit legacy-data adoption."""

import argparse

import psycopg
from psycopg import sql

from template_store import get_connection, get_database_url, seed_templates_from_files
from tenancy import tenant_context, tenant_schema, ensure_workspace


def copy_legacy(user_id: str) -> None:
    schema = tenant_schema(user_id)
    # Copy all four tables atomically; never overwrite an existing workspace.
    with get_connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (schema,))
        for table in ("templates", "agent_sessions", "agent_runs", "messages"):
            if conn.execute(sql.SQL("SELECT EXISTS (SELECT 1 FROM {}) AS occupied").format(sql.Identifier(table))).fetchone()["occupied"]:
                raise RuntimeError("Legacy adoption requires an empty workspace")
        for table in ("templates", "agent_sessions", "agent_runs", "messages"):
            conn.execute(sql.SQL("INSERT INTO {} SELECT * FROM public.{}").format(sql.Identifier(table), sql.Identifier(table)))
        conn.execute("SELECT setval(pg_get_serial_sequence('messages', 'message_id'), COALESCE(MAX(message_id), 1), MAX(message_id) IS NOT NULL) FROM messages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, help="Confirmed Supabase Auth user UUID")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--copy-legacy", action="store_true", help="Assign all old public data to this account")
    group.add_argument("--seed-templates", action="store_true", help="Seed bundled templates and generate embeddings")
    args = parser.parse_args()
    with psycopg.connect(get_database_url()) as conn:
        if not conn.execute("SELECT 1 FROM auth.users WHERE id = %s AND email_confirmed_at IS NOT NULL", (args.user_id,)).fetchone():
            parser.error("User must be a confirmed account in this Supabase project")
    with tenant_context(args.user_id):
        ensure_workspace()
        if args.copy_legacy:
            copy_legacy(args.user_id)
        elif args.seed_templates:
            seed_templates_from_files(embed=True)
    print("Workspace ready")
