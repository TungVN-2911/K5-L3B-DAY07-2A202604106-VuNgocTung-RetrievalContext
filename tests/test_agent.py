import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agent import Agent, ProviderError


def call(name, **args):
    return {'candidates': [{'content': {'role': 'model', 'parts': [{'functionCall': {'name': name, 'args': args}}]}}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2}}


class LLM:
    def __init__(self, responses): self.responses, self.allowed = iter(responses), []
    def generate(self, system, contents, tools, allowed):
        self.allowed.append(list(allowed))
        r = next(self.responses)
        if isinstance(r, Exception): raise r
        return r


class Retriever:
    name = 'fake'
    def __init__(self): self.calls = []
    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return [{'id': f'doc:p{i}', 'text': f'chunk {i} about {query}', 'page': i, 'rank': i, 'score': 1 / i,
                 'metadata': {'chapter': 'rag'}} for i in range(1, top_k + 1)]


def agent(responses, **kw):
    llm, r = LLM(responses), Retriever()
    return Agent(llm, r, chapters=['rag', 'mcp'], **{'top_k': 2, **kw}), llm, r


class AgentTest(unittest.TestCase):
    def test_search_then_answer_maps_refs_to_chunk_ids(self):
        a, llm, r = agent([call('search', query='rag'), call('answer', answer='RAG retrieves [C2].', citations=['C2'])])
        out = a.run('What is RAG?')
        self.assertEqual((out['status'], out['citations'], out['answer']), ('answered', ['doc:p2'], 'RAG retrieves [doc:p2].'))
        self.assertEqual((out['searches'], out['llm_calls'], out['usage']['input']), (1, 2, 20))
        self.assertEqual(out['context_ids'], ['doc:p1', 'doc:p2'])

    def test_grouped_refs_are_accepted_and_expanded(self):
        a, _, _ = agent([call('search', query='rag'), call('answer', answer='A [C1, C2]. B [C2].', citations=['C1', 'C2'])])
        out = a.run('q?')
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(out['answer'], 'A [doc:p1][doc:p2]. B [doc:p2].')

    def test_chapter_filter_only_from_allowed_values(self):
        a, _, r = agent([call('search', query='x', chapter='mcp'), call('search', query='y', chapter='bogus'),
                         call('abstain', reason='none')])
        out = a.run('In the MCP chapter, x?')
        self.assertEqual([c[1] for c in r.calls], [{'chapter': 'mcp'}, None])
        self.assertEqual(out['status'], 'abstain')

    def test_invalid_citation_gets_one_retry_then_abstains(self):
        a, _, _ = agent([call('search', query='q'), call('answer', answer='x [C9]', citations=['C9']),
                         call('answer', answer='x', citations=['C1'])])
        out = a.run('q?')
        self.assertEqual(out['status'], 'abstain')
        self.assertEqual([e['event'] for e in out['events']], ['search', 'invalid_answer', 'invalid_answer'])

    def test_search_limit_removes_search_tool(self):
        a, llm, _ = agent([call('search', query='a'), call('search', query='b'), call('abstain', reason='no')], max_searches=2)
        a.run('q?')
        self.assertEqual(llm.allowed, [['search', 'answer', 'abstain']] * 2 + [['answer', 'abstain']])

    def test_repeated_chunks_are_not_resent_and_budget_drops_new_text(self):
        a, _, _ = agent([call('search', query='a'), call('search', query='a'), call('abstain', reason='no')], budget=20)
        out = a.run('q?')
        first, second = [e for e in out['events'] if e['event'] == 'search']
        self.assertEqual([x.get('new') for x in first['results']], [True, None])
        self.assertEqual(first['results'][1]['dropped'], 'budget')
        self.assertFalse(second['results'][0]['new'])
        self.assertLessEqual(out['context_chars'], 20)

    def test_provider_error_is_reported_not_faked(self):
        a, _, _ = agent([ProviderError('Gemini HTTP 500')])
        out = a.run('q?')
        self.assertEqual((out['status'], out['answer'], out['citations']), ('error', '', []))
        self.assertIn('500', out['error'])


if __name__ == '__main__':
    unittest.main()
