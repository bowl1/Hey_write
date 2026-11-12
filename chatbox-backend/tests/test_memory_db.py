"""Memory behavior and isolation against the actual migrations and runtime role."""
import json
import os
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from agent import memory_store
from agent.graph import run_agent
from agent.schemas import AgentRunRequest
from tenancy import tenant_context, configure_tenant_connection, validate_runtime_role
from tests.migrations import apply_migrations, RUNTIME_ROLE


@unittest.skipUnless(os.getenv("RLS_TEST_DATABASE_URL"), "Set RLS_TEST_DATABASE_URL to a disposable pgvector database")
class MemoryDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ["RLS_TEST_DATABASE_URL"], row_factory=dict_row)
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        apply_migrations(self.conn)
        self.conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(RUNTIME_ROLE)))
        self.users = [str(uuid4()), str(uuid4())]
        for module in ("agent.memory_store", "agent.store"):
            patcher = patch(f"{module}.get_connection", side_effect=self.borrow)
            patcher.start()
            self.addCleanup(patcher.stop)

    def borrow(self):
        validate_runtime_role(self.conn)
        configure_tenant_connection(self.conn)
        return nullcontext(self.conn)

    def test_profiles_and_summaries_are_tenant_isolated_and_fail_closed(self):
        for user in self.users:
            with tenant_context(user):
                self.borrow()
                self.conn.execute("INSERT INTO agent_sessions (session_id) VALUES ('same')")
                profile = memory_store.load_profile()
                self.assertTrue(memory_store.save_profile([{"key": "name", "value": user}], profile["revision"]))
                cursor = memory_store.load_summary("same")
                self.assertTrue(memory_store.save_summary("same", cursor, user, 1, 5))
        for user, other in (self.users, self.users[::-1]):
            with tenant_context(user):
                self.assertEqual(memory_store.load_profile()["facts"][0]["value"], user)
                self.assertEqual(memory_store.load_summary("same")["summary"], user)
                for table in ("memory_profiles", "session_summaries"):
                    for operation in ("SELECT * FROM {}", "UPDATE {} SET tenant_id = tenant_id", "DELETE FROM {}"):
                        result = self.conn.execute(sql.SQL(operation + " WHERE tenant_id = %s").format(sql.Identifier(table)), (other,))
                        self.assertEqual(result.rowcount, 0)
                    with self.assertRaises(psycopg.errors.InsufficientPrivilege), self.conn.transaction():
                        self.conn.execute(sql.SQL("UPDATE {} SET tenant_id = %s").format(sql.Identifier(table)), (other,))
        self.conn.execute("SELECT set_config('app.tenant_id', '', true)")
        for table in ("memory_profiles", "session_summaries"):
            self.assertEqual(self.conn.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))).fetchall(), [])
        with self.assertRaises(psycopg.errors.InsufficientPrivilege), self.conn.transaction():
            self.conn.execute("INSERT INTO memory_profiles (tenant_id) VALUES (%s)", (self.users[0],))

    def test_clearing_disabling_and_concurrent_updates_reject_stale_writes(self):
        with tenant_context(self.users[0]):
            initial = memory_store.load_profile()
            self.assertTrue(memory_store.save_profile([{"key": "name", "value": "Anna"}], initial["revision"]))
            self.assertFalse(memory_store.save_profile([], initial["revision"]))
            current = memory_store.load_profile()
            cleared = memory_store.configure_profile(clear=True)
            self.assertEqual(cleared["facts"], [])
            self.assertFalse(memory_store.save_profile(current["facts"], current["revision"]))
            disabled = memory_store.configure_profile(enabled=False)
            self.assertFalse(memory_store.save_profile(current["facts"], disabled["revision"]))
            resumed = memory_store.configure_profile(enabled=True)
            self.assertEqual(resumed["facts"], [])
            self.assertTrue(memory_store.save_profile(current["facts"], resumed["revision"]))

    def test_summary_cursor_handles_long_messages_without_repeating_or_losing_tails(self):
        with tenant_context(self.users[0]):
            self.borrow()
            self.conn.execute("INSERT INTO agent_sessions (session_id) VALUES ('long')")
            content = "a" * 12000 + "IMPORTANT_TAIL" * 200
            first = self.conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('long', 'user', %s) RETURNING message_id", (content,)).fetchone()["message_id"]
            for index in range(20):
                self.conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('long', 'user', %s)", (f"recent {index}",))
            cursor = memory_store.load_summary("long")
            batch, mid, offset = memory_store.summary_batch("long", cursor)
            self.assertEqual(batch, [{"role": "user", "content": content[:12000]}])
            self.assertEqual((mid, offset), (first, 12000))
            self.assertTrue(memory_store.save_summary("long", cursor, "First part", mid, offset))
            self.assertFalse(memory_store.save_summary("long", cursor, "Stale writer", mid, offset))
            cursor = memory_store.load_summary("long")
            batch, mid, offset = memory_store.summary_batch("long", cursor)
            self.assertEqual(batch[0]["content"], content[12000:])
            self.assertTrue(memory_store.save_summary("long", cursor, "Including important tail", mid, offset))
            batch, _, _ = memory_store.summary_batch("long", memory_store.load_summary("long"))
            self.assertEqual(batch, [])

    def test_real_graph_remembers_across_sessions_but_not_across_accounts(self):
        facts = Mock()
        facts.run.side_effect = [json.dumps({"facts": [{"key": "name", "value": "Anna", "evidence": "My name is Anna"}]}), '{"facts":[]}', '{"facts":[]}']
        writing = Mock(); writing.run.return_value = "My name is Anna.\n\nChanges:\n- Added introduction."
        with patch.dict(os.environ, {"AGENT_MEMORY_ENABLED": "true", "AGENT_SUMMARY_ENABLED": "true", "AGENT_LLM_EVALUATOR_ENABLED": "false", "LANGFUSE_TRACING_ENABLED": "false"}), \
             patch("agent.nodes.get_llm_chain", return_value=writing), \
             patch("agent.long_term_memory.get_memory_chain", return_value=facts):
            with tenant_context(self.users[0]):
                first = run_agent(AgentRunRequest(session_id="first", action="wild", intent="My name is Anna"))
                self.assertTrue(first.state.draft_accepted)
                second = run_agent(AgentRunRequest(session_id="second", action="wild", intent="write an introduction using my name"))
                self.assertTrue(second.state.draft_accepted)
                context = json.loads(writing.run.call_args.args[0]["history"])
                self.assertEqual(context["long_term_memory"], [{"key": "name", "value": "Anna"}])
                self.assertEqual(context["recent_messages"], [])
            with tenant_context(self.users[1]):
                run_agent(AgentRunRequest(session_id="second", action="wild", intent="write an introduction"))
                self.assertEqual(json.loads(writing.run.call_args.args[0]["history"])["long_term_memory"], [])
