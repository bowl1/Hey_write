"""Real embedding retrieval benchmark; isolated database fixtures are rolled back.

Run from chatbox-backend:
  .venv/bin/python -m tests.evals.run_template_retrieval_eval --output /tmp/template-quality.json

Uses real OpenAI embeddings, not generated drafts or a model judging itself.
"""
import argparse
import json
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

import template_store as templates
from tenancy import tenant_context, configure_tenant_connection, validate_runtime_role, ensure_workspace
from agent.nodes import generate_draft_node

CASES_PATH = Path(__file__).with_name('template_retrieval_cases.jsonl')


def assess_case(case, hits, decision):
    expected = set(case.get('expected_ids', []))
    ids = [hit['template']['id'] for hit in hits]
    meta = decision.get('template_meta') or {}
    actual = ('choose' if meta.get('requires_selection') else
              'reject' if decision.get('blocked_reason') else 'select')
    candidates = {item['id'] for item in meta.get('candidates', [])}
    selected = meta.get('selected_template_id') if actual == 'select' else None
    if case.get('expected_decision') == 'reject':
        passed = actual == 'reject'
    elif case.get('expected_decision') == 'choose':
        passed = actual == 'choose' and candidates == expected
    else:
        # A correct option in the top three does not excuse selecting a wrong one.
        passed = (actual == 'select' and selected in expected) or (actual == 'choose' and bool(candidates & expected))
    return dict(id=case['id'], language=case['language'], intent=case['intent'],
                expected_ids=sorted(expected), expected_decision=case.get('expected_decision', 'select_or_choose'),
                decision=actual, selected_id=selected, choice_ids=sorted(candidates), passed=passed,
                top1_correct=bool(ids and ids[0] in expected) if expected else None,
                recall_at_3=bool(expected & set(ids)) if expected else None,
                candidates=[{'id': hit['template']['id'], 'title': hit['template']['title'],
                             'score': hit['final_score'], 'semantic_score': hit.get('vector_score'),
                             'keyword_score': hit['bm25_score']} for hit in hits])


def summarize(results):
    labeled = [row for row in results if row['top1_correct'] is not None]
    return {'cases': len(results), 'passed': sum(row['passed'] for row in results),
            'decision_accuracy': sum(row['passed'] for row in results) / len(results) if results else 0,
            'top1_accuracy': sum(row['top1_correct'] for row in labeled) / len(labeled) if labeled else 0,
            'recall_at_3': sum(row['recall_at_3'] for row in labeled) / len(labeled) if labeled else 0,
            'failed_case_ids': [row['id'] for row in results if not row['passed']]}


def run(cases):
    conn = psycopg.connect(templates.get_database_url(), row_factory=dict_row)
    user = str(uuid4())
    def borrowed():
        validate_runtime_role(conn)
        configure_tenant_connection(conn)
        return nullcontext(conn)
    try:
        with tenant_context(user), patch('template_store.get_connection', side_effect=borrowed):
            ensure_workspace()
            public = templates.get_template('default:meeting_invitation')
            vectors = templates.load_public_template_vectors([public])
            results = []
            for case in cases:
                # Reset fixture data so each case sees an independent library.
                conn.execute('DELETE FROM templates')
                count = {'private': 1, 'ambiguity': 2}.get(case.get('fixture'), 0)
                for index in range(count):
                    item = dict(public, id='eval_private_invitation' + ('_copy' if index else ''))
                    saved = templates.save_template(item, embed=False)
                    assert saved["id"] == item["id"], "Fixture ID must match its expected label"
                    conn.execute('UPDATE templates SET embedding = %s::public.vector WHERE id = %s',
                                 (templates.vector_literal(vectors[public['id']]), saved['id']))
                hits = templates.search_templates_with_metadata(case['intent'])
                # Exercise the actual selection/ambiguity gate without generating a draft.
                chain = Mock(); chain.run.return_value = 'Routing probe'
                with patch('agent.nodes.get_llm_chain', return_value=chain):
                    decision = generate_draft_node({'intent': case['intent'], 'retrieved_templates': hits})
                result = assess_case(case, hits, decision)
                results.append(result)
                print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']}: {result['decision']} {result['selected_id'] or result['choice_ids']}", flush=True)
            return results
    finally:
        conn.rollback()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=CASES_PATH)
    parser.add_argument('--output', type=Path, default=Path('/tmp/heywrite-template-quality.json'))
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if not cases: parser.error('No evaluation cases')
    results = run(cases)
    report = {'configuration': {'embedding_model': templates.EMBEDDING_MODEL, 'min_match_score': templates.MIN_MATCH_SCORE, 'vector_weight': templates.VECTOR_WEIGHT, 'keyword_weight': templates.BM25_WEIGHT, 'choice_score_gap': templates.TEMPLATE_CHOICE_SCORE_GAP}, 'summary': summarize(results), 'by_language': {
        lang: summarize([row for row in results if row['language'] == lang])
        for lang in sorted({row['language'] for row in results})}, 'results': results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    print(f'Report: {args.output}')
    raise SystemExit(0 if all(row['passed'] for row in results) else 1)


if __name__ == '__main__': main()
