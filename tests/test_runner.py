"""Runner integration with a fake process, not a CUDA correctness test."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root/'fake_benchmark'
        self.out = self.root/'run'

    def run_fake(self, body):
        self.binary.write_text('#!' + sys.executable + '\n' + body)
        self.binary.chmod(0o755)
        return subprocess.run([sys.executable,str(ROOT/'scripts/run_benchmarks.py'),
                               '--binary',str(self.binary),'--output',str(self.out),
                               '--sizes','2','--shapes','--samples','1','--repeats','1'],
                              capture_output=True,text=True)

    def test_success_and_no_overwrite(self):
        body = '''import csv, sys
args = dict(zip(sys.argv[1::2],sys.argv[2::2]))
with open(args['--csv'], 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['m','n','k','variant','pattern','seed','sample','repeats','ms','validated'])
    for name in ['naive','tiled16','tiled32','cublas']:
        w.writerow([2,2,2,name,'random',args['--seed'],0,1,1,1])
'''
        first = self.run_fake(body)
        self.assertEqual(first.returncode,0,first.stderr)
        metadata = json.loads((self.out/'metadata.json').read_text())
        self.assertEqual(metadata['status'],'complete')
        self.assertTrue(metadata['binary_sha256'])
        summary = subprocess.run([sys.executable,str(ROOT/'scripts/analyze.py'),
                                  str(self.out),'--no-plots'],capture_output=True,text=True)
        self.assertEqual(summary.returncode,0,summary.stderr)
        self.assertTrue((self.out/'analysis/summary.md').is_file())
        before = (self.out/'metadata.json').read_bytes()
        second = self.run_fake(body)
        self.assertNotEqual(second.returncode,0)
        self.assertEqual((self.out/'metadata.json').read_bytes(),before)

    def test_subprocess_failure_recorded(self):
        proc = self.run_fake('raise SystemExit(7)\n')
        self.assertNotEqual(proc.returncode,0)
        metadata = json.loads((self.out/'metadata.json').read_text())
        self.assertEqual(metadata['status'],'failed')
        self.assertIn('error',metadata)
        self.assertTrue((self.out/'2x2x2.log').is_file())


if __name__ == '__main__':
    unittest.main()
