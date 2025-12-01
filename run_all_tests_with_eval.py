import json
import sys
import os
from datetime import datetime
from src.multi_agent_system import MultiAgentSystem

OUT_LOG = 'final_run_output.txt'
MIS_FILE = 'misclassifications.json'


def main():
    try:
        with open('tests/test_queries.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('Failed to load tests/test_queries.json:', e)
        sys.exit(1)

    tests = data.get('test_queries') or []
    if not tests:
        print('No test queries found in tests/test_queries.json')
        sys.exit(1)

    # Ensure evaluator/LLM enabled
    os.environ.pop('DISABLE_LLM', None)
    os.environ.pop('DISABLE_EVALUATION', None)
    os.environ.pop('DISABLE_LANGFUSE', None)

    system = MultiAgentSystem(rebuild_vector_stores=False)

    total = len(tests)
    matches = 0
    mis = []

    lines = []
    lines.append(f"Run Time: {datetime.utcnow().isoformat()}Z")
    lines.append(f"LLM Enabled: {os.getenv('DISABLE_LLM') != '1'}")
    lines.append('')

    for i, tq in enumerate(tests, 1):
        query = tq.get('query') if isinstance(tq, dict) else tq
        expected = tq.get('expected_intent') if isinstance(tq, dict) else None
        lines.append('='*80)
        lines.append(f'[{i}/{total}] Query: {query}')
        try:
            result = system.process_query(query)
        except Exception as e:
            lines.append(f'  Error processing query: {type(e).__name__}: {e}')
            continue

        intent = result.get('intent')
        agent = result.get('agent')
        lines.append(f'  Intent: {intent}')
        lines.append(f'  Agent:  {agent}')

        match = None
        if expected is not None:
            match = (intent == expected) or (expected.lower() in str(intent).lower())
            lines.append(f'  Expected: {expected} -> Match: {match}')
            if match:
                matches += 1
        eval_info = result.get('evaluation')
        if eval_info:
            lines.append('  Evaluation:')
            lines.append(f"    Overall: {eval_info.get('overall_score')}/10")
            lines.append(f"    Relevance: {eval_info.get('relevance')}/10")
            lines.append(f"    Completeness: {eval_info.get('completeness')}/10")
            lines.append(f"    Accuracy: {eval_info.get('accuracy')}/10")
        else:
            lines.append('  Evaluation: (skipped or unavailable)')

        # collect misclassifications
        if expected is not None and not match:
            mis.append({
                'id': tq.get('id'),
                'query': query,
                'expected': expected,
                'predicted': intent,
                'agent': agent,
                'evaluation': eval_info
            })

    lines.append('')
    lines.append('='*80)
    lines.append(f'Total: {total}, Matches: {matches}, Accuracy: {matches/total*100:.1f}%')

    # write log
    with open(OUT_LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # write misclassifications
    with open(MIS_FILE, 'w', encoding='utf-8') as f:
        json.dump(mis, f, indent=2)

    print('Done.')
    print(f'Log: {OUT_LOG}')
    print(f'Misclassifications: {MIS_FILE}')


if __name__ == '__main__':
    main()
