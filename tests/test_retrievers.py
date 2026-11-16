import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from ingest import to_record
from retrievers import DenseRetriever

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


if __name__ == '__main__':
    unittest.main()
