import unittest
from unittest.mock import Mock, patch

from agent.evaluator import evaluate_agent_output, llm_evaluate
from agent.nodes import generate_wild_node, revise_draft_node, revise_with_feedback_node
from langchain_runner.rag_chain import prompt


class TaskEvaluationTests(unittest.TestCase):
    def state(self, action):
        return {
            "action": action, "intent": "summarize today's meeting",
            "previous_draft": "Subject: Invitation\n\nAgenda:\n- Attend",
            "current_draft": "Subject: Invitation\n\nAgenda:\n- Attend",
            "reply": "Meeting summary framework\n\nSummary:\n[Meeting notes needed]\n\nChanges:\n- Provided a framework; meeting notes are missing.",
            "template_meta": {"used_template": action == "new_task", "selected_template_id": "summary"},
            "active_template": {"id": "summary", "content": "Summary:\nDecisions:"},
        }

    def test_new_documents_cannot_fail_the_old_structure_check(self):
        verdict = {"checks": {"preserves_original_structure": False,
            "completed_user_request": True, "no_unrelated_fabrication": True}}
        for action in ("new_task", "wild", "continue_editing"):
            with self.subTest(action=action), patch("agent.evaluator.llm_evaluate", return_value=verdict):
                result = evaluate_agent_output(self.state(action))
            self.assertEqual(result["checks"]["preserves_original_structure"], action != "continue_editing")
            self.assertEqual(result["passed"], action != "continue_editing")

    def test_new_task_still_rejects_incomplete_or_fabricated_output(self):
        for check in ("completed_user_request", "no_unrelated_fabrication"):
            checks = {"preserves_original_structure": False, "completed_user_request": True,
                      "no_unrelated_fabrication": True, check: False}
            with self.subTest(check=check), patch("agent.evaluator.llm_evaluate", return_value={"checks": checks}):
                result = evaluate_agent_output(self.state("new_task"))
            self.assertFalse(result["passed"])
            self.assertIn(check, result["issues"])

    def test_judge_compares_previous_draft_only_for_continued_editing(self):
        chain = Mock(); chain.run.return_value = '{"checks":{}}'
        for action in ("new_task", "wild", "continue_editing"):
            with self.subTest(action=action), patch("agent.evaluator.get_evaluator_chain", return_value=chain), \
                 patch.dict("os.environ", {"AGENT_LLM_EVALUATOR_ENABLED": "true"}):
                state = self.state(action)
                llm_evaluate(state)
                self.assertEqual(chain.run.call_args.args[0]["previous_draft"],
                                 state["previous_draft"] if action == "continue_editing" else "")

    def test_retry_repairs_candidate_body_with_original_intent_and_selected_template(self):
        state = self.state("new_task")
        state["evaluation_feedback"] = "Clarify that meeting notes are missing"
        chain = Mock(); chain.run.return_value = "Repaired summary\n\nChanges:\n- Clarified missing notes."
        with patch("agent.nodes.get_llm_chain", return_value=chain):
            revise_with_feedback_node(state)
        inputs = chain.run.call_args.args[0]
        self.assertEqual(inputs["intent"], state["intent"])
        self.assertEqual(inputs["action"], "new_task")
        self.assertEqual(inputs["evaluation_feedback"], state["evaluation_feedback"])
        self.assertEqual(inputs["context"], state["active_template"]["content"])
        self.assertIn("Meeting summary framework", inputs["previous"])
        self.assertNotIn("Changes:", inputs["previous"])
        self.assertNotIn("Invitation", inputs["previous"])
        prompt.format(**inputs)

    def test_wild_starts_fresh_but_continue_editing_uses_saved_body(self):
        chain = Mock(); chain.run.return_value = "Draft\n\nChanges:\n- Updated."
        state = self.state("wild")
        state["current_draft"] += "\n\nChanges:\n- Old changes."
        with patch("agent.nodes.get_llm_chain", return_value=chain):
            generate_wild_node(state)
            inputs = chain.run.call_args.args[0]
            self.assertEqual(inputs["previous"], "")
            self.assertEqual(inputs["action"], "wild")
            prompt.format(**inputs)
            revise_draft_node(state)
            inputs = chain.run.call_args.args[0]
            self.assertIn("Invitation", inputs["previous"])
            self.assertNotIn("Changes:", inputs["previous"])
            self.assertEqual(inputs["action"], "continue_editing")
            prompt.format(**inputs)
