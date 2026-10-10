"""Real PostgreSQL isolation check. All test DDL/data is rolled back."""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

import template_store
import agent.store as store
from tenancy import tenant_context, tenant_id, tenant_schema, ensure_workspace


@unittest.skipUnless(os.getenv("RUN_DB_TESTS") == "1", "Set RUN_DB_TESTS=1 to test PostgreSQL")
class DatabaseIsolationTests(unittest.TestCase):
    def test_templates_sessions_messages_runs_and_vector_search_are_isolated(self):
        conn = psycopg.connect(template_store.get_database_url(), row_factory=dict_row)
        users = [str(uuid4()), str(uuid4())]
        def borrowed():
            conn.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(sql.Identifier(tenant_schema(tenant_id()))))
            return nullcontext(conn)
        try:
            with patch("template_store.get_connection", side_effect=borrowed), \
                 patch("agent.store.get_connection", side_effect=borrowed), \
                 patch("tenancy.psycopg.connect", return_value=nullcontext(conn)):
                for index, user in enumerate(users):
                    with tenant_context(user):
                        ensure_workspace()
                        ensure_workspace()
                        template_store.save_template({"id": "same", "title": f"Tenant {index}", "content": f"private {index}"}, embed=False)
                        store.persist_agent_state({"session_id": "same-session", "run_id": "same-run", "action": "wild", "intent": f"request {index}", "reply": f"draft {index}"})
                vector = [1.0] + [0.0] * (template_store.EMBEDDING_DIMENSION - 1)
                for index, user in enumerate(users):
                    with tenant_context(user):
                        self.assertEqual(template_store.list_templates()[0]["content"], f"private {index}")
                        template = template_store.get_template("same")
                        with patch("template_store.embed_text", return_value=vector):
                            template_store.upsert_template_embedding(template)
                            hits = template_store.search_templates_with_metadata("private")
                        self.assertEqual(len(hits), 1)
                        self.assertEqual(hits[0]["template"]["content"], f"private {index}")
                        snapshot = store.load_session_snapshot("same-session")
                        self.assertEqual(snapshot["session"]["current_draft"], f"draft {index}")
                        self.assertEqual(snapshot["messages"][0]["content"], f"request {index}")
                        self.assertEqual(snapshot["last_run"]["reply"], f"draft {index}")
                        self.assertEqual(store.list_session_summaries()[0]["current_draft"], f"draft {index}")
                # A missing workspace must fail instead of falling back to public.
                with tenant_context(str(uuid4())):
                    with self.assertRaises(psycopg.errors.UndefinedTable):
                        with conn.transaction():
                            borrowed()
                            conn.execute("SELECT * FROM messages")
        finally:
            conn.rollback()
            conn.close()


if __name__ == "__main__":
    unittest.main()
