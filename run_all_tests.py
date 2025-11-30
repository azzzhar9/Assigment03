import json
import sys

from src.multi_agent_system import MultiAgentSystem


def main():
    try:
        with open('tests/test_queries.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('Failed to load tests/test_queries.json:', e)
        sys.exit(1)

    tests = data.get('test_queries') or data.get('queries') or []
    if not tests:
        print('No test queries found in tests/test_queries.json')
        sys.exit(1)

    system = MultiAgentSystem(rebuild_vector_stores=False)

    total = len(tests)
    matches = 0

    for i, tq in enumerate(tests, 1):
        query = tq.get('query') if isinstance(tq, dict) else tq
        expected = tq.get('expected_intent') if isinstance(tq, dict) else None
        print('\n' + '='*80)
        print(f'[{i}/{total}] Query: {query}')
        try:
            result = system.process_query(query)
        except Exception as e:
            print('  Error processing query:', type(e).__name__, e)
            continue

        intent = result.get('intent')
        agent = result.get('agent')
        print(f'  Intent: {intent}')
        print(f'  Agent:  {agent}')

        if expected is not None:
            match = (intent == expected) or (expected.lower() in str(intent).lower())
            print(f'  Expected: {expected} -> Match: {match}')
            if match:
                matches += 1

        eval_info = result.get('evaluation')
        if eval_info:
            print('  Evaluation:')
            print(f"    Overall: {eval_info.get('overall_score')}/10")
            print(f"    Relevance: {eval_info.get('relevance')}/10")
            print(f"    Completeness: {eval_info.get('completeness')}/10")
            print(f"    Accuracy: {eval_info.get('accuracy')}/10")
        else:
            print('  Evaluation: (skipped or unavailable)')

    print('\n' + '='*80)
    print(f'Total: {total}, Matches: {matches}, Accuracy: {matches/total*100:.1f}%')


if __name__ == '__main__':
    main()
