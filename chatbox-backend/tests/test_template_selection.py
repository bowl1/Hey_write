import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
from template_store import tokenize, bm25_scores, list_public_templates, hybrid_match_score


class TemplateSelectionTests(unittest.TestCase):
    def test_chinese_request_has_keyword_evidence(self):
        self.assertIn('会议', tokenize('请写会议邀请函'))
        scores, _ = bm25_scores('请写会议邀请函', list_public_templates())
        self.assertEqual(max(scores, key=scores.get), 'default:meeting_invitation')
        self.assertEqual(hybrid_match_score(.75, 0, False), .75)


if __name__ == '__main__': unittest.main()
