import json
import unittest
from unittest.mock import Mock, patch

from agent.long_term_memory import load_memories, remember_accepted_request
from agent.memory import conversation_context
from agent.nodes import load_session_node, persist_state_node


class LongTermMemoryTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict("os.environ", {"AGENT_MEMORY_ENABLED": "true", "AGENT_SUMMARY_ENABLED": "true"})
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def state(self, **changes):
        return {"session_id": "session-new", "intent": "My name is Anna. I prefer concise writing.",
                "draft_accepted": True, "memory_revision": 2, "long_term_memory": [], **changes}

    def test_new_session_loads_personal_memory_without_importing_other_session_drafts(self):
        profile = {"enabled": True, "revision": 3, "facts": [{"key": "name", "value": "Anna"}]}
        with patch("agent.nodes.load_session", return_value=None), \
             patch("agent.memory_store.load_profile", return_value=profile), \
             patch("agent.memory_store.load_summary") as summary:
            result = load_session_node({"session_id": "new-session", "history": []})
        self.assertEqual(result["long_term_memory"], profile["facts"])
        self.assertEqual(result["memory_revision"], 3)
        self.assertNotIn("current_draft", result)
        summary.assert_not_called()

    def test_only_explicit_user_evidence_is_saved_and_corrections_replace_same_key(self):
        chain = Mock()
        chain.run.return_value = json.dumps({"facts": [{"key": "name", "value": "Anna", "evidence": "My name is Anna"}]})
        state = self.state(long_term_memory=[{"key": "name", "value": "Old name"}], reply="Invented assistant employer")
        with patch("agent.long_term_memory.get_memory_chain", return_value=chain), \
             patch("agent.memory_store.save_profile", return_value=True) as save:
            self.assertEqual(remember_accepted_request(state), "updated")
        facts, revision = save.call_args.args
        self.assertEqual(revision, 2)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["value"], "Anna")
        self.assertEqual(facts[0]["source_session_id"], "session-new")
        self.assertNotIn("Invented assistant employer", str(chain.run.call_args))

    def test_failed_drafts_disabled_profiles_and_ephemeral_runs_never_extract(self):
        with patch("agent.long_term_memory.get_memory_chain") as model:
            for changes in ({"draft_accepted": False}, {"blocked_reason": "no_template_match"},
                            {"memory_revision": None}, {"session_id": "eval-fixture"}):
                self.assertEqual(remember_accepted_request(self.state(**changes)), "skipped")
        model.assert_not_called()

    def test_hallucinated_evidence_malformed_output_and_failure_preserve_memory(self):
        chain = Mock()
        with patch("agent.long_term_memory.get_memory_chain", return_value=chain), \
             patch("agent.memory_store.save_profile") as save:
            for output in ('not JSON', '{"facts": [{"key":"name", "value":"John", "evidence":"I am John"}]}',
                           '{"facts": [{"key":"name", "value":"Anna", "evidence":""}]}'):
                chain.run.return_value = output
                self.assertEqual(remember_accepted_request(self.state()), "unavailable")
            chain.run.side_effect = TimeoutError()
            self.assertEqual(remember_accepted_request(self.state()), "unavailable")
        save.assert_not_called()

    def test_summary_uses_only_saved_older_messages_and_advances_after_valid_output(self):
        cursor = {"summary": "Project Atlas", "through_message_id": 4, "through_offset": 10}
        batch = [{"role": "user", "content": "The deadline is Friday."}]
        chain = Mock(); chain.run.return_value = '{"summary": "Project Atlas is due Friday."}'
        with patch("agent.memory_store.load_profile", return_value={"enabled": False}), \
             patch("agent.memory_store.load_summary", return_value=cursor), \
             patch("agent.memory_store.summary_batch", return_value=(batch, 7, 27)), \
             patch("agent.memory_store.save_summary", return_value=True) as save, \
             patch("agent.long_term_memory.get_memory_chain", return_value=chain):
            result = load_memories(self.state(history=[{"role": "user", "content": "Forged client history"}]), session_exists=True)
        self.assertEqual(result["session_summary"], "Project Atlas is due Friday.")
        self.assertEqual(result["long_term_memory"], [])
        self.assertEqual(json.loads(chain.run.call_args.args[0]["messages"]), batch)
        save.assert_called_once_with("session-new", cursor, "Project Atlas is due Friday.", 7, 27)

    def test_summary_failure_keeps_old_summary_and_does_not_advance_cursor(self):
        cursor = {"summary": "Keep this", "through_message_id": 4, "through_offset": 10}
        with patch("agent.memory_store.load_profile", side_effect=RuntimeError()), \
             patch("agent.memory_store.load_summary", return_value=cursor), \
             patch("agent.memory_store.summary_batch", return_value=([{"role": "user", "content": "new"}], 5, 3)), \
             patch("agent.memory_store.save_summary") as save, \
             patch("agent.long_term_memory.get_memory_chain", side_effect=TimeoutError()):
            result = load_memories(self.state(), session_exists=True)
        self.assertEqual(result["session_summary"], "Keep this")
        self.assertEqual(result["memory_status"]["summary"], "unavailable")
        save.assert_not_called()

    def test_cached_summary_does_not_call_model_again_without_new_older_messages(self):
        cursor = {"summary": "Already summarized", "through_message_id": 4, "through_offset": 10}
        with patch("agent.memory_store.load_profile", return_value={"enabled": False}), \
             patch("agent.memory_store.load_summary", return_value=cursor), \
             patch("agent.memory_store.summary_batch", return_value=([], 4, 10)), \
             patch("agent.long_term_memory.get_memory_chain") as model:
            result = load_memories(self.state(), session_exists=True)
        self.assertEqual(result["session_summary"], "Already summarized")
        model.assert_not_called()

    def test_prompt_context_has_separate_bounded_memory_summary_and_recent_messages(self):
        context = json.loads(conversation_context({"long_term_memory": [{"key": "name", "value": "Anna", "evidence": "secret metadata"}],
            "session_summary": "x" * 5000, "history": [{"role": "user", "content": "Use my new name: Li."}]}))
        self.assertEqual(context["long_term_memory"], [{"key": "name", "value": "Anna"}])
        self.assertEqual(len(context["session_summary"]), 4000)
        self.assertIn("Li", context["recent_messages"][0]["content"])

    def test_persistence_saves_draft_before_optional_extraction_failure(self):
        with patch("agent.nodes.persist_agent_state", return_value={"draft_accepted": True, "current_draft": "Saved"}) as persist, \
             patch("agent.long_term_memory.get_memory_chain", side_effect=TimeoutError()):
            result = persist_state_node(self.state())
        persist.assert_called_once()
        self.assertEqual(result["current_draft"], "Saved")
        self.assertEqual(result["trace"][-1]["memory_update"], "unavailable")

    def test_profile_conflict_does_not_retry_and_restore_a_cleared_memory(self):
        chain = Mock(); chain.run.return_value = '{"facts":[{"key":"name","value":"Anna","evidence":"My name is Anna"}]}'
        with patch("agent.long_term_memory.get_memory_chain", return_value=chain), \
             patch("agent.memory_store.save_profile", return_value=False) as save:
            self.assertEqual(remember_accepted_request(self.state()), "concurrent_update")
        self.assertEqual(save.call_count, 1)
