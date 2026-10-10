import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
from template_store import tokenize, bm25_scores, list_public_templates, hybrid_match_score, load_public_template_vectors


class TemplateSelectionTests(unittest.TestCase):
    def test_chinese_request_has_keyword_evidence(self):
        self.assertIn('会议', tokenize('请写会议邀请函'))
        scores, _ = bm25_scores('请写会议邀请函', list_public_templates())
        self.assertEqual(max(scores, key=scores.get), 'default:meeting_invitation')
        self.assertEqual(hybrid_match_score(.75, 0, False), .75)


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
