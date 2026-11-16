import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from multimodal import QUESTIONS, covered, evidence_hit, first_cover_rank, region_id


def chunk(page, kind, text, label=None):
    return {'id': f'x:p{page}:{kind}', 'page': page, 'text': text, 'metadata': {'kind': kind, 'label': label, 'title': None}}


TABLE = {'page': 100, 'region': 'Table 3.1', 'kind': 'table', 'value': '890 [640-1160]'}
QUOTE = {'page': 35, 'region': 'SPM B.5.2', 'kind': 'text', 'value': '500 GtCO2 for a 50% likelihood … limiting global warming to 1.5°C'}


class RegionTest(unittest.TestCase):
    def test_region_ids(self):
        self.assertEqual(region_id('Figure SPM.7 panel a'), 'Figure SPM.7')
        self.assertEqual(region_id('Table 3.1'), 'Table 3.1')
        self.assertEqual(region_id('Cross-Section Box.2 Figure 1 panel c'), 'Cross-Section Box.2')
        self.assertIsNone(region_id('SPM A.2.1'))


class HitTest(unittest.TestCase):
    def test_visual_evidence_needs_same_page_and_region(self):
        self.assertTrue(evidence_hit(chunk(100, 'table', 'TABLE | Table 3.1 | Key characteristics', 'Table 3.1'), TABLE))
        self.assertFalse(evidence_hit(chunk(100, 'text', 'Table 3.1 values 890'), TABLE))
        self.assertFalse(evidence_hit(chunk(37, 'table', 'TABLE | Table 3.1', 'Table 3.1'), TABLE))
        self.assertFalse(evidence_hit(chunk(100, 'chart', 'CHART | Figure 3.5', 'Figure 3.5'), TABLE))

    def test_text_evidence_needs_every_quoted_part(self):
        text = 'The best estimates ... are 500 GtCO2 for a 50% likelihood of limiting global warming to 1.5°C and 1150'
        self.assertTrue(evidence_hit(chunk(35, 'text', text), QUOTE))
        self.assertFalse(evidence_hit(chunk(35, 'text', '1150 GtCO2 for 2°C'), QUOTE))

    def test_multihop_requires_all_and_alternatives_substitute(self):
        q = {'evidence': [TABLE, QUOTE], 'evidence_required': 'all', 'alternative_evidence': []}
        t = chunk(100, 'table', 'Table 3.1', 'Table 3.1')
        s = chunk(35, 'text', '500 GtCO2 for a 50% likelihood of limiting global warming to 1.5°C')
        self.assertFalse(covered([t], q))
        self.assertTrue(covered([t, s], q))
        self.assertEqual(first_cover_rank([t, chunk(1, 'text', 'x'), s], q), 3)
        alt = {**q, 'evidence': [TABLE], 'alternative_evidence': [QUOTE]}
        self.assertTrue(covered([s], alt))


class BenchmarkFileTest(unittest.TestCase):
    def test_advanced_benchmark_meets_challenge_minimums(self):
        qs = json.loads(QUESTIONS.read_text(encoding='utf-8'))
        test = [q for q in qs if q['split'] == 'test']
        dev = [q for q in qs if q['split'] == 'dev']
        count = lambda rows, g: sum(q['group'] == g for q in rows)
        self.assertGreaterEqual(len(test), 30)
        for group, need in {'table': 6, 'image': 5, 'chart': 5, 'multihop': 4, 'text': 4, 'unanswerable': 6}.items():
            self.assertGreaterEqual(count(test, group), need, group)
        self.assertGreaterEqual(sum(q['answerable'] for q in dev), 15)
        self.assertGreaterEqual(count(dev, 'unanswerable'), 3)
        self.assertEqual(len({q['id'] for q in qs}), len(qs))
        self.assertFalse({q['question'].lower() for q in dev} & {q['question'].lower() for q in test})
        for q in qs:
            if q['answerable']:
                self.assertTrue(q['evidence'] and q['expected_answer'] and q['reason'], q['id'])
            else:
                self.assertTrue(q['absence_check']['absent_patterns'], q['id'])


if __name__ == '__main__':
    unittest.main()
