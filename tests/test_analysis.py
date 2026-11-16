"""Synthetic fixtures ONLY: exercise analysis math and rejection of bad runs."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('analyze', ROOT/'scripts/analyze.py')
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = {'status':'complete', 'shape_order_mnk':[[2,3,4]],
                     'arguments':{'samples':3,'seed':42,'repeats':20}}
        self.rows = []
        for name,factor in [('naive',1),('tiled16',2),('tiled32',4),('cublas',8)]:
            for sample,ms in enumerate([2,4,6]):
                self.rows.append(dict(m=2,n=3,k=4,variant=name,pattern='random',seed=42,
                                      sample=sample,repeats=20,ms=ms/factor,validated=1))

    def write(self):
        (self.root/'metadata.json').write_text(json.dumps(self.meta))
        with (self.root/'2x3x4.csv').open('w',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def reject(self):
        self.write()
        with self.assertRaises(ValueError):
            analyze.load_run(self.root)

    def test_ratios_units_and_spread(self):
        self.write()
        result = {r['variant']:r for r in analyze.load_run(self.root)}
        tiled = result['tiled16']
        self.assertEqual(tiled['median_ms'],2)
        self.assertEqual(tiled['min_ms'],1)
        self.assertEqual(tiled['max_ms'],3)
        self.assertEqual(tiled['speedup_vs_naive'],2)
        self.assertEqual(tiled['percent_cublas'],25)
        self.assertAlmostEqual(tiled['gflops'],48/(2*1e6))

    def test_failed_run_rejected(self):
        self.meta['status'] = 'failed'
        self.reject()

    def test_unvalidated_rejected(self):
        self.rows[0]['validated'] = 0
        self.reject()

    def test_missing_variant_rejected(self):
        self.rows = [r for r in self.rows if r['variant']!='cublas']
        self.reject()

    def test_duplicate_sample_rejected(self):
        self.rows.append(self.rows[0].copy())
        self.reject()

    def test_wrong_dimensions_rejected(self):
        self.rows[0]['m'] = 7
        self.reject()

    def test_invalid_timings_rejected(self):
        for value in ['nan','inf',0,-1]:
            with self.subTest(value=value):
                self.rows[0]['ms'] = value
                self.reject()

    def test_wrong_seed_rejected(self):
        self.rows[0]['seed'] = 17
        self.reject()

    def test_wrong_repeats_rejected(self):
        self.rows[0]['repeats'] = 5
        self.reject()

    def test_single_sample_allowed(self):
        self.meta['arguments']['samples'] = 1
        self.rows = [r for r in self.rows if r['sample']==0]
        self.write()
        for row in analyze.load_run(self.root):
            self.assertEqual(row['q1_ms'],row['median_ms'])
            self.assertEqual(row['q3_ms'],row['median_ms'])


if __name__ == '__main__':
    unittest.main()
