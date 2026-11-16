#!/usr/bin/env python3
"""GPU correctness suite: tile edges, rectangles, structured inputs, multiple seeds."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build/matmul_bench')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    shapes = [(1,1,1), (2,3,4), (15,17,19), (16,16,16), (17,31,33),
              (31,32,33), (32,32,32), (33,65,17), (65,33,63), (127,193,65)]
    seeds = [42] if args.quick else [1,42,137]
    count = 0
    for m,n,k in shapes:
        for seed in seeds:
            subprocess.run([binary, '--m', str(m), '--n', str(n), '--k', str(k),
                            '--seed', str(seed), '--validate-only'], check=True)
            count += 1
    for pattern in ['zero', 'ones', 'identity', 'alternating']:
        subprocess.run([binary, '--m', '33', '--n', '65', '--k', '35',
                        '--pattern', pattern, '--validate-only'], check=True)
        count += 1
    print(f'PASS: {count} cases, each checking all four implementations')


if __name__ == '__main__':
    main()
