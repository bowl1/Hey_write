import asyncio
import json
import os
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent.evaluator import evaluate_agent_output, llm_evaluate
from agent.store import list_session_summaries, load_session_snapshot
from main import app
from tenancy import AuthenticatedUser, authenticate

A = "00000000-0000-0000-0000-000000000001"
NOW = datetime(2026, 10, 26, 9, 30, tzinfo=timezone.utc)
CHECKS = ("preserves_original_structure", "completed_user_request", "used_correct_template",
          "contains_changes", "no_unrelated_fabrication")


def connection(*results):
    conn = Mock()
    conn.execute.side_effect = [Mock(fetchone=Mock(return_value=r), fetchall=Mock(return_value=r)) for r in results]
    return nullcontext(conn)


class SupabaseUserTests(unittest.TestCase):
    def authenticate_with(self, payload):
        auth_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
        with patch.dict(os.environ, {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "public-test-key"}), \
             patch("tenancy.httpx.AsyncClient", return_value=auth_client):
            return asyncio.run(authenticate("Bearer token"))

    def test_malformed_identity_is_a_service_error(self):
        for payload in ({"email": "a@example.com"}, {"id": "not-a-uuid", "email": "a@example.com"}, ["not", "an", "object"]):
            with self.assertRaises(HTTPException) as caught:
                self.authenticate_with(payload)
            self.assertEqual(caught.exception.status_code, 503)

    def test_unconfirmed_or_anonymous_accounts_are_rejected(self):
        for payload in ({"id": A, "email": "a@example.com"},
                        {"id": A, "email": "a@example.com", "email_confirmed_at": "2026-01-01T00:00:00Z", "is_anonymous": True}):
            with self.assertRaises(HTTPException) as caught:
                self.authenticate_with(payload)
            self.assertEqual(caught.exception.status_code, 403)


class EvaluatorVerdictTests(unittest.TestCase):
    state = {"action": "wild", "intent": "write a greeting", "reply": "Hello team.\n\nChanges:\n- Added greeting.",
             "template_meta": {"used_template": False}, "trace": []}

    def verdict(self, output):
        chain = Mock(); chain.run.return_value = output
        with patch("agent.evaluator.get_evaluator_chain", return_value=chain), \
             patch.dict(os.environ, {"AGENT_LLM_EVALUATOR_ENABLED": "true"}):
            return llm_evaluate(self.state)

    def test_string_false_is_not_read_as_a_passed_check(self):
        result = self.verdict(json.dumps({"checks": {name: "false" for name in CHECKS}}))
        self.assertFalse(any(result["checks"].values()))
        self.assertEqual(result["reason"], "LLM evaluator completed.")

    def test_malformed_verdict_falls_back_to_deterministic_checks(self):
        for output in ("not json", json.dumps({"reason": "no checks"}), json.dumps({"checks": {"contains_changes": "maybe"}})):
            self.assertIn("error", self.verdict(output))
            chain = Mock(); chain.run.return_value = output
            with patch("agent.evaluator.get_evaluator_chain", return_value=chain), \
                 patch.dict(os.environ, {"AGENT_LLM_EVALUATOR_ENABLED": "true"}):
                evaluation = evaluate_agent_output(self.state)
            self.assertEqual(evaluation["source"], "deterministic")
            self.assertTrue(evaluation["llm_error"])


class SessionResponseTests(unittest.TestCase):
    session = {"session_id": "s1", "active_template_id": None, "current_draft": "Draft",
               "style": "Formal", "language": "English", "created_at": NOW, "updated_at": NOW}
    messages = [{"role": "user", "content": "Write it", "created_at": NOW}]
    last_run = {"run_id": "r1", "action": "new_task", "intent": "Write it", "selected_template_id": "t1",
           "output_state": {"draft_accepted": True}, "trace": [{"node": "start"}], "reply": "Draft", "created_at": NOW}

    def get(self, path, conn):
        async def verify(header):
            return AuthenticatedUser(id=A)
        with patch("main.authenticate", side_effect=verify), patch("main.ensure_workspace"), \
             patch("agent.store.get_connection", return_value=conn):
            return TestClient(app).get(path, headers={"Authorization": "Bearer a"})

    def test_snapshot_keeps_the_frontend_contract(self):
        response = self.get("/agent/session/s1", connection(self.session, self.messages, self.last_run))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"session", "messages", "last_run"})
        self.assertEqual(body["session"]["current_draft"], "Draft")
        self.assertEqual(body["session"]["updated_at"], "2026-10-26T09:30:00Z")
        self.assertEqual(body["messages"], [{"role": "user", "content": "Write it", "created_at": "2026-10-26T09:30:00Z"}])
        self.assertEqual(body["last_run"]["output_state"], {"draft_accepted": True})

    def test_missing_session_and_run_are_reported_as_such(self):
        self.assertEqual(self.get("/agent/session/none", connection(None)).status_code, 404)
        with patch("agent.store.get_connection", return_value=connection(self.session, [], None)):
            self.assertIsNone(load_session_snapshot("s1").last_run)

    def test_summary_title_and_counts(self):
        with patch("agent.store.get_connection", return_value=connection([self.session], self.messages)):
            summary, = list_session_summaries()
        self.assertEqual((summary.title, summary.message_count), ("Write it", 1))
        response = self.get("/agent/sessions", connection([self.session], self.messages))
        self.assertEqual(set(response.json()["sessions"][0]), {
            "session_id", "title", "active_template_id", "current_draft", "style",
            "language", "message_count", "updated_at", "messages"})


if __name__ == "__main__":
    unittest.main()
