import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from ingest import to_record
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

ROW = {'chunk_id': 'doc:p10:fixed500-50:0', 'document_id': 'doc', 'source': 'doc.pdf', 'page': 10,
       'metadata': {'chapter': 'llms', 'printed_page': 9, 'char_start': 0, 'char_end': 500},
       'text': 'A large language model is ...', 'score': 0.81}


class Embedder:
    model_name, dimensions = 'fake', 3
    def embed_query(self, q): return [0.1, 0.2, 0.3]


class Store:
    dimensions = 3
    def __init__(self, rows=None, error=None): self.rows, self.error, self.calls = rows or [], error, []
    def search(self, strategy, vector, top_k, filters):
        self.calls.append((strategy, top_k, filters))
        if self.error: raise self.error
        return self.rows[:top_k]


class AdapterTest(unittest.TestCase):
    def test_maps_store_row_to_chunk_schema(self):
        c = to_record(ROW)
        self.assertEqual((c['id'], c['source_id'], c['page'], c['score']), (ROW['chunk_id'], 'doc', 10, 0.81))
        self.assertEqual(set(c), {'id', 'text', 'source_id', 'source', 'page', 'metadata', 'score'})

    def test_search_passes_strategy_filter_topk_and_ranks(self):
        store = Store([ROW, {**ROW, 'chunk_id': 'doc:p11:fixed500-50:0', 'page': 11, 'score': 0.7}])
        r = DenseRetriever('A', embedder=Embedder(), store=store)
        out = r.search('llm?', filters={'chapter': 'llms'}, top_k=1)
        self.assertEqual(store.calls, [('A', 1, {'chapter': 'llms'})])
        self.assertEqual([(c['rank'], c['strategy']) for c in out], [(1, 'dense')])

    def test_store_errors_propagate(self):
        r = DenseRetriever('A', embedder=Embedder(), store=Store(error=RuntimeError('db down')))
        with self.assertRaises(RuntimeError):
            r.search('x')

    def test_rejects_dimension_mismatch_and_empty_query(self):
        store = Store(); store.dimensions = 768
        with self.assertRaises(RuntimeError):
            DenseRetriever('A', embedder=Embedder(), store=store)
        with self.assertRaises(ValueError):
            DenseRetriever('A', embedder=Embedder(), store=Store()).search('  ')


def chunk(id, text, page=1, chapter='rag'):
    return {'id': id, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf', 'page': page,
            'metadata': {'chapter': chapter}}


CHUNKS = [chunk('a', 'vector database stores embeddings of unstructured data'),
          chunk('b', 'fine-tuning adjusts the weights of a model', 2, 'fine_tuning'),
          chunk('c', 'the database is a database with many tables', 3)]


class Fixed:
    def __init__(self, name, ids): self.name, self.ids = name, ids
    def search(self, query, *, filters=None, top_k=3):
        return [{**chunk(i, i), 'score': 1.0 / r, 'rank': r, 'strategy': self.name} for r, i in enumerate(self.ids[:top_k], 1)]
    def info(self): return {'retriever': self.name}


class BM25Test(unittest.TestCase):
    def test_ranks_by_term_weight_and_skips_zero_scores(self):
        out = BM25Retriever(CHUNKS).search('vector database embeddings', top_k=5)
        self.assertEqual([c['id'] for c in out], ['a', 'c'])
        self.assertEqual([c['rank'] for c in out], [1, 2])

    def test_applies_metadata_filter_and_rejects_unknown_filter(self):
        r = BM25Retriever(CHUNKS)
        self.assertEqual([c['id'] for c in r.search('weights model', filters={'chapter': 'rag'})], [])
        self.assertEqual([c['id'] for c in r.search('weights', filters={'chapter': 'fine_tuning'})], ['b'])
        with self.assertRaises(ValueError):
            r.search('x', filters={'year': 2025})


class HybridTest(unittest.TestCase):
    def test_rrf_rewards_agreement_between_retrievers(self):
        h = HybridRetriever([Fixed('dense', ['x', 'y', 'z']), Fixed('bm25', ['z', 'y', 'w'])], candidates=3, rrf_k=60)
        out = h.search('q', top_k=4)
        self.assertEqual([c['id'] for c in out], ['z', 'y', 'x', 'w'])
        self.assertAlmostEqual(out[0]['score'], 1 / 63 + 1 / 61)
        self.assertAlmostEqual(out[1]['score'], 2 / 62)
        self.assertEqual(out[1]['components'], {'dense': {'rank': 2, 'score': 0.5}, 'bm25': {'rank': 2, 'score': 0.5}})


class RerankTest(unittest.TestCase):
    def test_reorders_candidates_and_keeps_first_stage_rank(self):
        scorer = lambda q, texts: [{'x': 0.1, 'y': 0.9, 'z': 0.5}[t] for t in texts]
        out = RerankRetriever(Fixed('hybrid', ['x', 'y', 'z']), scorer=scorer, candidates=3).search('q', top_k=2)
        self.assertEqual([(c['id'], c['rank'], c['first_stage']['rank']) for c in out], [('y', 1, 2), ('z', 2, 3)])

    def test_rejects_score_count_mismatch(self):
        r = RerankRetriever(Fixed('hybrid', ['x', 'y']), scorer=lambda q, t: [1.0], candidates=2)
        with self.assertRaises(RuntimeError):
            r.search('q')


if __name__ == '__main__':
    unittest.main()
