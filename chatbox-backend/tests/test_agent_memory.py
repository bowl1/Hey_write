import json
import unittest
from unittest.mock import Mock, patch

from agent.memory import conversation_context
from agent.nodes import (
    load_session_node, generate_draft_node, generate_wild_node,
    revise_draft_node, revise_with_feedback_node,
)
from agent.evaluator import llm_evaluate, EVALUATOR_PROMPT
from langchain_runner.rag_chain import prompt


class AgentMemoryTests(unittest.TestCase):
    def test_existing_session_uses_saved_history_and_restores_previous_draft(self):
        saved_history = [{"role": "user", "content": "The organizer is Anna."}]
        with patch("agent.nodes.load_session", return_value={"current_draft": "Saved draft", "active_template_id": "meeting"}), \
             patch("agent.nodes.load_session_history", return_value=saved_history):
            updates = load_session_node({"session_id": "session", "history": [{"role": "user", "content": "client history"}]})
        self.assertEqual(updates["history"], saved_history)
        self.assertEqual(updates["previous_draft"], "Saved draft")

    def test_history_reaches_all_generation_paths_and_the_judge(self):
        state = {
            "intent": "use the same organizer", "action": "continue_editing",
            "current_draft": "Dear team", "history": [{"role": "user", "content": "The organizer is Anna."}],
            "active_template": {"id": "meeting", "content": "Dear team"},
            "retrieved_templates": [{"template": {"id": "meeting", "title": "Meeting", "content": "Dear team"},
                "final_score": 1.0, "distance": 0, "vector_distance": 0, "bm25_score": 1, "matched_terms": []}],
        }
        chain = Mock()
        chain.run.return_value = "Draft\n\nChanges:\n- Updated organizer."
        with patch("agent.nodes.get_llm_chain", return_value=chain):
            for node in (generate_draft_node, generate_wild_node, revise_draft_node, revise_with_feedback_node):
                node(state)
                inputs = chain.run.call_args.args[0]
                self.assertIn("Anna", prompt.format(**inputs))
                self.assertEqual(json.loads(inputs["history"]), state["history"])
        chain.run.return_value = json.dumps({"checks": {name: True for name in (
            "preserves_original_structure", "completed_user_request", "used_correct_template", "contains_changes", "no_unrelated_fabrication")}})
        with patch("agent.evaluator.get_evaluator_chain", return_value=chain), \
             patch.dict("os.environ", {"AGENT_LLM_EVALUATOR_ENABLED": "true"}):
            llm_evaluate(state)
        self.assertIn("Anna", EVALUATOR_PROMPT.format(**chain.run.call_args.args[0]))

    def test_context_is_bounded_and_rejects_system_roles(self):
        history = [{"role": "system", "content": "untrusted instruction"}] + [
            {"role": "user", "content": str(index) + "x" * 5000} for index in range(30)
        ]
        context = json.loads(conversation_context({"history": history}))
        self.assertLessEqual(len(context), 20)
        self.assertLessEqual(sum(len(message["content"]) for message in context), 12000)
        self.assertTrue(all(message["role"] == "user" for message in context))
        self.assertTrue(context[-1]["content"].startswith("29"))


if __name__ == "__main__":
    unittest.main()
