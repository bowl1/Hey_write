"""Administrative, transactional migration from tenant schemas to shared tables.

Run during a write maintenance window using DATABASE_ADMIN_URL, never the web role.
Sources are retained. Conflicts fail rather than silently overwriting tenant data.
"""
import argparse
import os
import re
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from database_schema import init_tables

TABLES = ('templates', 'agent_sessions', 'agent_runs', 'messages')


def copy_workspace(conn, source: str, user: str) -> dict:
    user = str(UUID(user))
    previous = conn.execute('SELECT tenant_id FROM heywrite.schema_imports WHERE source_schema = %s', (source,)).fetchone()
    if previous:
        if str(previous['tenant_id']) != user:
            raise RuntimeError('Source schema was already assigned to another tenant')
        return {'source': source, 'status': 'already imported'}
    for table in TABLES:
        if not conn.execute('SELECT to_regclass(%s) AS name', (f'{source}.{table}',)).fetchone()['name']:
            raise RuntimeError(f'Incomplete source schema: {source}.{table}')
        conn.execute(sql.SQL('LOCK TABLE {} IN SHARE MODE').format(sql.Identifier(source, table)))
        if conn.execute(sql.SQL('SELECT 1 FROM {} WHERE tenant_id = %s LIMIT 1').format(
                sql.Identifier('heywrite', table)), (user,)).fetchone():
            raise RuntimeError(f'Target tenant already has data: {user}; refusing to merge')
    counts = {}
    for table in TABLES:
        columns = conn.execute("""SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'heywrite' AND table_name = %s AND column_name <> 'tenant_id'
            ORDER BY ordinal_position""", (table,)).fetchall()
        names = sql.SQL(', ').join(sql.Identifier(row['column_name']) for row in columns)
        target = sql.Identifier('heywrite', table)
        origin = sql.Identifier(source, table)
        counts[table] = conn.execute(sql.SQL('INSERT INTO {} (tenant_id, {}) SELECT %s::uuid, {} FROM {}').format(
            target, names, names, origin), (user,)).rowcount
        # Verify values as well as counts before committing the import marker.
        mismatch = conn.execute(sql.SQL('''SELECT EXISTS (
            (SELECT {} FROM {} EXCEPT ALL SELECT {} FROM {} WHERE tenant_id = %s)
            UNION ALL
            (SELECT {} FROM {} WHERE tenant_id = %s EXCEPT ALL SELECT {} FROM {})
        ) AS mismatch''').format(names, origin, names, target, names, target, names, origin), (user, user)).fetchone()
        if mismatch['mismatch']:
            raise RuntimeError(f'Copy verification failed: {source}.{table}')
    conn.execute('INSERT INTO heywrite.schema_imports (source_schema, tenant_id) VALUES (%s, %s)', (source, user))
    return {'source': source, 'counts': counts}


def migrate(conn, runtime_role='heywrite_app', copy_tenants=True, public_owner=None):
    privilege = conn.execute('SELECT rolsuper OR rolbypassrls AS admin FROM pg_roles WHERE rolname = current_user').fetchone()
    if not privilege['admin']:
        raise RuntimeError('Migration requires an administrative role with BYPASSRLS to verify every tenant')
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('heywrite-shared-migration-v1', 0))")
    role = conn.execute('SELECT rolsuper, rolbypassrls, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname = %s', (runtime_role,)).fetchone()
    if role and any(role.values()):
        raise RuntimeError('Runtime role must not have administrative privileges')
    if not role:
        conn.execute(sql.SQL('CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT').format(sql.Identifier(runtime_role)))
    if conn.execute('SELECT 1 FROM pg_auth_members WHERE member = (SELECT oid FROM pg_roles WHERE rolname = %s)', (runtime_role,)).fetchone():
        raise RuntimeError('Use a dedicated runtime role without other role memberships')
    conn.execute('CREATE SCHEMA IF NOT EXISTS heywrite')
    conn.execute('REVOKE ALL ON SCHEMA heywrite FROM PUBLIC')
    conn.execute('SET LOCAL search_path TO heywrite, pg_catalog')
    init_tables(conn)
    conn.execute('''CREATE TABLE IF NOT EXISTS heywrite.schema_imports (
        source_schema text PRIMARY KEY, tenant_id uuid NOT NULL UNIQUE,
        imported_at timestamptz NOT NULL DEFAULT now())''')
    conn.execute('''CREATE TABLE IF NOT EXISTS heywrite.default_template_vectors (
        template_id text NOT NULL, content_hash text NOT NULL, model text NOT NULL,
        dimension integer NOT NULL, embedding jsonb NOT NULL,
        PRIMARY KEY (template_id, content_hash, model, dimension))''')
    reports = []
    if copy_tenants:
        schemas = conn.execute("SELECT nspname FROM pg_namespace WHERE nspname LIKE 'tenant_%' ORDER BY nspname").fetchall()
        for row in schemas:
            source = row['nspname']
            if re.fullmatch(r'tenant_[0-9a-f]{32}', source):
                reports.append(copy_workspace(conn, source, str(UUID(source[7:]))))
    if public_owner:
        reports.append(copy_workspace(conn, 'public', public_owner))
    # A transactional sequence restart preserves imported message ordering and IDs.
    next_id = conn.execute('SELECT COALESCE(MAX(message_id), 0) + 1 AS next_id FROM heywrite.messages').fetchone()['next_id']
    conn.execute(sql.SQL('ALTER SEQUENCE heywrite.messages_message_id_seq RESTART WITH {}').format(sql.Literal(next_id)))
    role_name = sql.Identifier(runtime_role)
    conn.execute(sql.SQL('REVOKE ALL ON SCHEMA heywrite FROM {}').format(role_name))
    conn.execute(sql.SQL('GRANT USAGE ON SCHEMA heywrite TO {}').format(role_name))
    for table in TABLES:
        target = sql.Identifier('heywrite', table)
        conn.execute(sql.SQL('REVOKE ALL ON {} FROM PUBLIC, {}').format(target, role_name))
        conn.execute(sql.SQL('GRANT SELECT, INSERT, UPDATE, DELETE ON {} TO {}').format(target, role_name))
    conn.execute(sql.SQL('REVOKE ALL ON heywrite.schema_imports FROM PUBLIC, {}').format(role_name))
    conn.execute(sql.SQL('REVOKE ALL ON heywrite.default_template_vectors FROM PUBLIC, {}').format(role_name))
    conn.execute(sql.SQL('GRANT SELECT, INSERT ON heywrite.default_template_vectors TO {}').format(role_name))
    conn.execute(sql.SQL('REVOKE ALL ON SEQUENCE heywrite.messages_message_id_seq FROM PUBLIC, {}').format(role_name))
    conn.execute(sql.SQL('GRANT USAGE ON SEQUENCE heywrite.messages_message_id_seq TO {}').format(role_name))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-role', default='heywrite_app')
    parser.add_argument('--skip-tenant-copy', action='store_true', help='Initialize a fresh database without importing old tenant schemas')
    parser.add_argument('--public-owner', help='Explicitly assign legacy public tables to this confirmed user UUID')
    args = parser.parse_args()
    url = os.getenv('DATABASE_ADMIN_URL')
    if not url:
        parser.error('DATABASE_ADMIN_URL is required; never use the web DATABASE_URL for migrations')
    with psycopg.connect(url, row_factory=dict_row) as conn:
        if args.public_owner:
            owner = str(UUID(args.public_owner))
            if not conn.execute('SELECT 1 FROM auth.users WHERE id = %s AND email_confirmed_at IS NOT NULL', (owner,)).fetchone():
                parser.error('--public-owner must be a confirmed Supabase account')
        reports = migrate(conn, args.runtime_role, not args.skip_tenant_copy, args.public_owner)
    for report in reports:
        print(report)
    print('Shared schema ready. Configure a password for the runtime role securely and update the web DATABASE_URL.')
