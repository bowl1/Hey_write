import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
from template_store import tokenize, bm25_scores, list_public_templates, hybrid_match_score, load_public_template_vectors, prioritize_template_candidates
from agent.nodes import retrieve_template_node, generate_draft_node, finalize_node, evaluator_node, route_after_evaluation, load_active_template_node, revise_draft_node, revise_with_feedback_node


class TemplateSelectionTests(unittest.TestCase):
    def test_chinese_request_has_keyword_evidence(self):
        self.assertIn('会议', tokenize('请写会议邀请函'))
        scores, _ = bm25_scores('请写会议邀请函', list_public_templates())
        self.assertEqual(max(scores, key=scores.get), 'default:meeting_invitation')
        self.assertEqual(hybrid_match_score(.75, 0, False), .75)

    def test_explicit_template_bypasses_retrieval_and_uses_only_selected_content(self):
        template = {'id': 'default:meeting_agenda', 'title': 'Agenda', 'content': 'Selected content', 'enabled': True}
        with patch('agent.nodes.get_template', return_value=template), patch('agent.nodes.search_templates_with_metadata') as search:
            state = retrieve_template_node({'selected_template_id': template['id']})
        search.assert_not_called()
        chain = Mock(); chain.run.return_value = 'Draft'
        with patch('agent.nodes.get_llm_chain', return_value=chain):
            result = generate_draft_node(state)
        self.assertEqual(chain.run.call_args.args[0]['context'], 'Selected content')
        self.assertEqual(result['template_meta']['retrieval_source'], 'manual')

    def test_unavailable_selection_does_not_fall_back_to_other_templates(self):
        with patch('agent.nodes.get_template', return_value=None), patch('agent.nodes.search_templates_with_metadata') as search:
            result = retrieve_template_node({'selected_template_id': 'another-tenant-private-id'})
        search.assert_not_called()
        self.assertEqual(result['blocked_reason'], 'selected_template_unavailable')

    def test_close_scores_wait_for_choice_without_calling_llm(self):
        candidates = [{'template': {'id': key, 'title': key, 'source': 'default', 'content': key},
                       'final_score': score} for key, score in [('a', .85), ('b', .82)]]
        with patch('agent.nodes.get_llm_chain') as llm:
            result = generate_draft_node({'retrieved_templates': candidates})
        llm.assert_not_called()
        self.assertEqual(result['blocked_reason'], 'template_choice_required')
        self.assertEqual(len(result['template_meta']['candidates']), 2)
        self.assertNotIn('reply', finalize_node(result))
        with patch('agent.nodes.evaluate_agent_output') as judge:
            result.update(evaluator_node(result))
        judge.assert_not_called()
        self.assertEqual(route_after_evaluation(result), 'persist_state')

    def test_qualified_private_templates_take_priority_without_promoting_weak_matches(self):
        candidates = [{'template': {'id': key, 'source': source}, 'final_score': score}
                      for key, source, score in [('public', 'default', .9), ('private', 'private', .7), ('weak', 'private', .2)]]
        ranked = prioritize_template_candidates(candidates)
        self.assertEqual([item['template']['id'] for item in ranked], ['private', 'public', 'weak'])

    def test_clear_winner_is_the_only_generation_reference(self):
        candidates = [{'template': {'id': key, 'title': key, 'source': 'default', 'content': key},
                       'final_score': score, 'distance': 0, 'vector_distance': 0, 'bm25_score': 0, 'matched_terms': []}
                      for key, score in [('chosen', .9), ('irrelevant', .7)]]
        chain = Mock(); chain.run.return_value = 'Draft'
        with patch('agent.nodes.get_llm_chain', return_value=chain):
            generate_draft_node({'retrieved_templates': candidates})
        self.assertEqual(chain.run.call_args.args[0]['context'], 'chosen')

    def test_selected_template_is_consistent_across_generation_retry_and_continued_editing(self):
        chosen = {'id': 'default:email', 'title': 'Email', 'source': 'default', 'content': 'EMAIL_ONLY_STRUCTURE'}
        candidates = [{'template': template, 'final_score': score, 'distance': 0,
                       'vector_distance': 0, 'bm25_score': 0, 'matched_terms': []}
                      for template, score in [(chosen, .9),
                        ({'id': 'report', 'title': 'Report', 'source': 'default', 'content': 'REPORT_STRUCTURE'}, .7),
                        ({'id': 'agenda', 'title': 'Agenda', 'source': 'default', 'content': 'AGENDA_STRUCTURE'}, .66)]]
        state = {'intent': 'Write an email'}
        chain = Mock(); chain.run.return_value = 'An email draft'
        with patch('agent.nodes.search_templates_with_metadata', return_value=candidates), \
             patch('agent.nodes.get_llm_chain', return_value=chain), \
             patch('agent.nodes.get_template', return_value=chosen) as lookup:
            state.update(retrieve_template_node(state))
            state.update(generate_draft_node(state))
            selected_id = state['template_meta']['selected_template_id']
            self.assertEqual(selected_id, chosen['id'])
            self.assertEqual(state['active_template_id'], selected_id)
            self.assertEqual(chain.run.call_args.args[0]['context'], chosen['content'])
            state.update(revise_with_feedback_node(state))
            self.assertEqual(chain.run.call_args.args[0]['context'], chosen['content'])
            continued = {'active_template_id': selected_id, 'current_draft': state['reply'], 'intent': 'Shorten it'}
            continued.update(load_active_template_node(continued))
            lookup.assert_called_once_with(selected_id)
            result = revise_draft_node(continued)
            self.assertEqual(result['template_meta']['selected_template_id'], selected_id)
            self.assertEqual(chain.run.call_args.args[0]['context'], chosen['content'])
        for call in chain.run.call_args_list:
            self.assertNotIn('REPORT_STRUCTURE', call.args[0]['context'])
            self.assertNotIn('AGENDA_STRUCTURE', call.args[0]['context'])

    def test_public_embeddings_are_batched_and_reused_after_memory_reset(self):
        saved = []
        def execute(sql, params=None):
            if sql.startswith('INSERT'):
                saved.append({'template_id': params[0], 'content_hash': params[1], 'embedding': params[4].obj})
            return Mock(fetchall=lambda: saved.copy())
        conn = Mock(); conn.execute.side_effect = execute
        client = Mock(); client.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
        templates = [{'id': 'default:a', 'content': 'A'}, {'id': 'default:b', 'content': 'B'}]
        with patch('template_store.get_connection', return_value=nullcontext(conn)), patch('template_store.get_embedding_client', return_value=client), patch('template_store.EMBEDDING_DIMENSION', 2):
            first = load_public_template_vectors(templates)
            second = load_public_template_vectors(templates)
        self.assertEqual(first, second)
        client.embed_documents.assert_called_once()
        self.assertEqual(len(client.embed_documents.call_args.args[0]), 2)


if __name__ == '__main__': unittest.main()
