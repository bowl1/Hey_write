"""Run only against a disposable database with pgvector; all fixtures roll back."""
import os
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from import_legacy_tenants import import_legacy, TABLES
from tenancy import configure_tenant_connection, tenant_context, validate_runtime_role, tenant_schema
from tests.migrations import RUNTIME_ROLE, apply_migrations


@unittest.skipUnless(os.getenv('RLS_TEST_DATABASE_URL'), 'Set RLS_TEST_DATABASE_URL to a disposable PostgreSQL database with pgvector')
class RowSecurityTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['RLS_TEST_DATABASE_URL'], row_factory=dict_row)
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        self.role = RUNTIME_ROLE
        apply_migrations(self.conn)
        self.users = [str(uuid4()), str(uuid4())]

    def runtime(self, user):
        self.conn.execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(self.role)))
        validate_runtime_role(self.conn)
        with tenant_context(user):
            configure_tenant_connection(self.conn)

    def test_all_tables_isolate_reads_writes_ids_and_tenant_reassignment(self):
        conn = self.conn
        for user in self.users:
            self.runtime(user)
            conn.execute("INSERT INTO templates (id, title) VALUES ('same', %s)", (user,))
            conn.execute("INSERT INTO agent_sessions (session_id, current_draft) VALUES ('same', %s)", (user,))
            conn.execute("INSERT INTO agent_runs (run_id, session_id, action, intent) VALUES ('same', 'same', 'new_task', %s)", (user,))
            conn.execute("INSERT INTO messages (message_id, session_id, role, content) VALUES (42, 'same', 'user', %s)", (user,))
        for user, other in (self.users, self.users[::-1]):
            self.runtime(user)
            for table in TABLES:
                target = sql.Identifier('heywrite', table)
                rows = conn.execute(sql.SQL('SELECT tenant_id FROM {}').format(target)).fetchall()
                self.assertEqual([str(r['tenant_id']) for r in rows], [user])
                for operation in ('SELECT * FROM {}', 'UPDATE {} SET tenant_id = tenant_id', 'DELETE FROM {}'):
                    result = conn.execute(sql.SQL(operation + ' WHERE tenant_id = %s').format(target), (other,))
                    self.assertEqual(result.rowcount, 0)
                with self.assertRaises(psycopg.errors.InsufficientPrivilege), conn.transaction():
                    conn.execute(sql.SQL('UPDATE {} SET tenant_id = %s').format(target), (other,))
            with self.assertRaises(psycopg.errors.InsufficientPrivilege), conn.transaction():
                conn.execute("INSERT INTO templates (tenant_id, id, title) VALUES (%s, 'forged', 'no')", (other,))
        # A session that exists only for B cannot receive an A message.
        self.runtime(self.users[1])
        conn.execute("INSERT INTO agent_sessions (session_id) VALUES ('only-b')")
        self.runtime(self.users[0])
        with self.assertRaises(psycopg.errors.ForeignKeyViolation), conn.transaction():
            conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('only-b', 'user', 'no')")
        # Missing identity fails closed, including explicit tenant IDs.
        conn.execute("SELECT set_config('app.tenant_id', '', true)")
        for table in TABLES:
            self.assertEqual(conn.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table))).fetchall(), [])
        with self.assertRaises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("INSERT INTO templates (tenant_id, id, title) VALUES (%s, 'missing', 'no')", (self.users[0],))
        self.runtime(self.users[0])
        for table in ('messages', 'agent_runs', 'agent_sessions', 'templates'):
            self.assertEqual(conn.execute(sql.SQL('DELETE FROM {}').format(sql.Identifier(table))).rowcount, 1)
        self.runtime(self.users[1])
        self.assertEqual(conn.execute('SELECT count(*) AS n FROM templates').fetchone()['n'], 1)

    def test_runtime_cannot_change_schema_policies_or_read_import_ledger(self):
        self.runtime(self.users[0])
        for command in ('CREATE TABLE heywrite.forbidden (id int)',
                        'ALTER TABLE heywrite.templates DISABLE ROW LEVEL SECURITY',
                        'SELECT * FROM heywrite.schema_imports', 'TRUNCATE heywrite.templates'):
            with self.assertRaises(psycopg.errors.InsufficientPrivilege), self.conn.transaction():
                self.conn.execute(command)

    def test_migrations_are_idempotent(self):
        apply_migrations(self.conn)
        self.runtime(self.users[0])
        self.conn.execute("INSERT INTO templates (id, title) VALUES ('after-reapply', 'ok')")

    def test_admin_connection_is_rejected_by_application(self):
        with self.assertRaisesRegex(RuntimeError, 'BYPASSRLS'):
            validate_runtime_role(self.conn)

    def test_identity_is_transaction_local(self):
        with tenant_context(self.users[0]):
            configure_tenant_connection(self.conn)
        self.conn.rollback()
        self.assertFalse(self.conn.execute("SELECT current_setting('app.tenant_id', true) AS tenant").fetchone()['tenant'])

    def test_legacy_import_preserves_values_and_overlapping_ids_and_is_repeatable(self):
        conn = self.conn
        for user in self.users:
            source = tenant_schema(user)
            conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(source)))
            for table in TABLES:
                conn.execute(sql.SQL('CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS)').format(
                    sql.Identifier(source, table), sql.Identifier('heywrite', table)))
                conn.execute(sql.SQL('ALTER TABLE {} DROP COLUMN tenant_id').format(sql.Identifier(source, table)))
            conn.execute(sql.SQL("INSERT INTO {} (id, title, content) VALUES ('same', 'legacy', %s)").format(sql.Identifier(source, 'templates')), (user,))
            conn.execute(sql.SQL("INSERT INTO {} (session_id, current_draft) VALUES ('same', %s)").format(sql.Identifier(source, 'agent_sessions')), (user,))
            conn.execute(sql.SQL("INSERT INTO {} (run_id, session_id, action, intent) VALUES ('same', 'same', 'new_task', %s)").format(sql.Identifier(source, 'agent_runs')), (user,))
            conn.execute(sql.SQL("INSERT INTO {} (message_id, session_id, role, content) VALUES (99, 'same', 'user', %s)").format(sql.Identifier(source, 'messages')), (user,))
        reports = import_legacy(conn)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report['counts']['messages'] == 1 for report in reports))
        self.assertTrue(all(report['status'] == 'already imported' for report in import_legacy(conn)))
        for user in self.users:
            self.runtime(user)
            self.assertEqual(conn.execute("SELECT content FROM messages").fetchone()['content'], user)
            next_id = conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('same', 'user', 'new') RETURNING message_id").fetchone()['message_id']
            self.assertGreater(next_id, 99)
        conn.execute('RESET ROLE')
        for user in self.users:
            self.assertEqual(conn.execute(sql.SQL('SELECT count(*) AS n FROM {}').format(sql.Identifier(tenant_schema(user), 'messages'))).fetchone()['n'], 1)

    def test_conflicting_import_rolls_back_without_overwriting(self):
        conn = self.conn
        from import_legacy_tenants import copy_workspace
        user = self.users[0]
        source = tenant_schema(user)
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(source)))
        for table in TABLES:
            conn.execute(sql.SQL('CREATE TABLE {} (LIKE {})').format(sql.Identifier(source, table), sql.Identifier('heywrite', table)))
        conn.execute("INSERT INTO heywrite.templates (tenant_id, id, title) VALUES (%s, 'existing', 'keep')", (user,))
        with self.assertRaisesRegex(RuntimeError, 'already has data'), conn.transaction():
            copy_workspace(conn, source, user)
        self.assertEqual(conn.execute('SELECT title FROM heywrite.templates').fetchone()['title'], 'keep')
