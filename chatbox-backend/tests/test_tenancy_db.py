"""Real PostgreSQL isolation check. All test DDL/data is rolled back."""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

import template_store
import agent.store as store
from tenancy import tenant_context, ensure_workspace, configure_tenant_connection, validate_runtime_role


@unittest.skipUnless(os.getenv("RLS_TEST_DATABASE_URL"), "Set RLS_TEST_DATABASE_URL to a disposable PostgreSQL database with pgvector")
class DatabaseIsolationTests(unittest.TestCase):
    def test_templates_sessions_messages_runs_and_vector_search_are_isolated(self):
        conn = psycopg.connect(os.environ["RLS_TEST_DATABASE_URL"], row_factory=dict_row)
        users = [str(uuid4()), str(uuid4())]
        def borrowed():
            validate_runtime_role(conn)
            configure_tenant_connection(conn)
            return nullcontext(conn)
        try:
            from migrate_shared import migrate
            role = 'test_app_' + uuid4().hex
            migrate(conn, role, copy_tenants=False)
            conn.execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(role)))
            with patch("template_store.get_connection", side_effect=borrowed), \
                 patch("agent.store.get_connection", side_effect=borrowed):
                for index, user in enumerate(users):
                    with tenant_context(user):
                        ensure_workspace()
                        ensure_workspace()
                        template_store.save_template({"id": "same", "title": f"Tenant {index}", "content": f"private {index}"}, embed=False)
                        store.persist_agent_state({"session_id": "same-session", "run_id": "same-run", "action": "new_task", "intent": f"request {index}", "reply": f"draft {index}", "evaluation": {"passed": True}, "template_meta": {"used_template": True, "selected_template_id": "same"}})
                vector = [1.0] + [0.0] * (template_store.EMBEDDING_DIMENSION - 1)
                for index, user in enumerate(users):
                    with tenant_context(user):
                        listed = template_store.list_templates()
                        self.assertEqual(next(template for template in listed if template["id"] == "same")["content"], f"private {index}")
                        self.assertTrue(any(template["id"].startswith("default:") for template in listed))
                        template = template_store.get_template("same")
                        with patch("template_store.embed_text", return_value=vector), patch("template_store.get_embedding_client", return_value=Mock(embed_documents=lambda texts: [vector for _ in texts])):
                            template_store.upsert_template_embedding(template)
                            hits = template_store.search_templates_with_metadata("private")
                        private_hits = [hit for hit in hits if not hit["template"]["id"].startswith("default:")]
                        self.assertEqual(len(private_hits), 1)
                        self.assertEqual(private_hits[0]["template"]["content"], f"private {index}")
                        snapshot = store.load_session_snapshot("same-session")
                        self.assertEqual(snapshot["session"]["current_draft"], f"draft {index}")
                        self.assertEqual(snapshot["messages"][0]["content"], f"request {index}")
                        self.assertEqual(snapshot["last_run"]["reply"], f"draft {index}")
                        self.assertEqual(store.list_session_summaries()[0]["current_draft"], f"draft {index}")
                        for blocked in (None, "no_template_match"):
                            failed_id = str(uuid4())
                            result = store.persist_agent_state({"session_id": "same-session", "run_id": failed_id, "action": "new_task", "intent": "try again", "reply": "rejected candidate", "evaluation": {"passed": False}, "blocked_reason": blocked, "active_template_id": "wrong-template", "style": "Casual", "language": "Danish"})
                            self.assertFalse(result["draft_accepted"])
                            self.assertEqual(result["current_draft"], f"draft {index}")
                            self.assertEqual(result["active_template_id"], "same")
                            rejected = store.load_session_snapshot("same-session")
                            self.assertEqual(rejected["session"]["current_draft"], f"draft {index}")
                            self.assertEqual(rejected["session"]["style"], "Formal")
                            self.assertEqual(rejected["session"]["language"], "English")
                            self.assertFalse(any(message["content"] == "rejected candidate" for message in rejected["messages"]))
                            history = store.load_session_history("same-session")
                            self.assertEqual(history[0]["content"], f"request {index}")
                            self.assertFalse(any(message["content"] == "rejected candidate" for message in history))
                            row = conn.execute("SELECT reply, output_state FROM agent_runs WHERE run_id = %s", (failed_id,)).fetchone()
                            self.assertEqual(row["reply"], "rejected candidate")
                            self.assertFalse(row["output_state"]["draft_accepted"])
                        fresh = store.persist_agent_state({"session_id": "failed-first-session", "run_id": str(uuid4()), "action": "wild", "intent": "first try", "reply": "rejected", "evaluation": {"passed": False}})
                        self.assertEqual(fresh["current_draft"], "")
                        updated = store.persist_agent_state({"session_id": "same-session", "run_id": str(uuid4()), "action": "wild", "intent": "a successful retry", "reply": "New valid draft", "evaluation": {"passed": True}})
                        self.assertTrue(updated["draft_accepted"])
                        self.assertEqual(updated["current_draft"], "New valid draft")
                # A fresh account sees an empty workspace without any DDL.
                with tenant_context(str(uuid4())):
                    ensure_workspace()
                    borrowed()
                    self.assertEqual(conn.execute("SELECT * FROM messages").fetchall(), [])
        finally:
            conn.rollback()
            conn.close()


if __name__ == "__main__":
    unittest.main()
