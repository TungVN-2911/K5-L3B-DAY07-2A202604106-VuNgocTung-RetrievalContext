import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from benchmark import check_absence, check_schema, load_questions


class BenchmarkFileTest(unittest.TestCase):
    def test_committed_benchmark_meets_split_minimums(self):
        counts = check_schema(load_questions())
        self.assertGreaterEqual(counts['dev']['answerable'], 15)
        self.assertGreaterEqual(counts['dev']['unanswerable'], 2)
        self.assertGreaterEqual(counts['test']['answerable'], 5)
        self.assertGreaterEqual(counts['test']['unanswerable'], 2)

    def test_dev_and_test_questions_do_not_repeat(self):
        qs = load_questions()
        dev = {q['question'].casefold() for q in qs if q['split'] == 'dev'}
        self.assertFalse(dev & {q['question'].casefold() for q in qs if q['split'] == 'test'})


class AbsenceCheckTest(unittest.TestCase):
    Q = [{'id': 'u1', 'answerable': False, 'absence_check': {'absent_patterns': [r'BM25']}}]

    def test_flags_pattern_present_in_document(self):
        self.assertEqual(check_absence(self.Q, [(3, 'we use bm25 here')]), {'u1': ["'BM25' p3"]})

    def test_passes_when_pattern_absent(self):
        self.assertEqual(check_absence(self.Q, [(3, 'dense retrieval only')]), {})


if __name__ == '__main__':
    unittest.main()
