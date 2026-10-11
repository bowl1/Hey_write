import unittest
from tests.evals.run_template_retrieval_eval import assess_case, summarize


class RetrievalEvalTests(unittest.TestCase):
    def hits(self):
        return [{'template': {'id': key, 'title': key}, 'final_score': .8, 'bm25_score': 1}
                for key in ['wrong', 'correct']]

    def test_recall_does_not_hide_wrong_selection(self):
        result = assess_case({'id': 'case', 'language': 'en', 'intent': 'request', 'expected_ids': ['correct']},
                             self.hits(), {'template_meta': {'selected_template_id': 'wrong'}})
        self.assertTrue(result['recall_at_3'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['top1_correct'])

    def test_choice_requires_expected_candidate_and_ambiguity_fixture_requires_both(self):
        case = {'id': 'case', 'language': 'en', 'intent': 'request', 'expected_ids': ['correct']}
        decision = {'template_meta': {'requires_selection': True, 'candidates': [{'id': 'wrong'}]}}
        self.assertFalse(assess_case(case, self.hits(), decision)['passed'])
        case.update(expected_decision='choose', expected_ids=['correct', 'wrong'])
        self.assertFalse(assess_case(case, self.hits(), decision)['passed'])
        decision['template_meta']['candidates'].append({'id': 'correct'})
        self.assertTrue(assess_case(case, self.hits(), decision)['passed'])

    def test_irrelevant_request_must_be_rejected(self):
        case = {'id': 'case', 'language': 'zh', 'intent': 'recipe', 'expected_decision': 'reject'}
        self.assertTrue(assess_case(case, self.hits(), {'blocked_reason': 'no_template_match'})['passed'])
        self.assertFalse(assess_case(case, self.hits(), {'template_meta': {'selected_template_id': 'wrong'}})['passed'])

    def test_summary_separates_decision_accuracy_and_retrieval_recall(self):
        row = {'id': 'bad', 'passed': False, 'top1_correct': False, 'recall_at_3': True}
        self.assertEqual(summarize([row])['decision_accuracy'], 0)
        self.assertEqual(summarize([row])['recall_at_3'], 1)
