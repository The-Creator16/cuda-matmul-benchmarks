#!/usr/bin/env python3
"""Create one self-contained run directory; never overwrite earlier measurements."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return {'command': command, 'exit_code': proc.returncode,
                'stdout': proc.stdout, 'stderr': proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'error': str(error)}


def positive(text):
    value = int(text)
    if not 1 <= value <= 16384:
        raise argparse.ArgumentTypeError('must be in [1,16384]')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build/matmul_bench')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', nargs='+', type=positive, default=[64,128,256,512,1024,2048])
    parser.add_argument('--shapes', nargs='*', default=['127x193x65','512x1024x256','1024x256x512'],
                        help='extra MxNxK cases; use --shapes with no values for squares only')
    parser.add_argument('--samples', type=positive, default=11)
    parser.add_argument('--repeats', type=positive, default=20)
    parser.add_argument('--warmup', type=positive, default=5)
    parser.add_argument('--seed', type=positive, default=42)
    args = parser.parse_args()
    shapes = [(n,n,n) for n in args.sizes]
    for text in args.shapes:
        try:
            parts = tuple(positive(x) for x in text.split('x'))
            if len(parts) != 3:
                raise ValueError()
            shapes.append(parts)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error('invalid shape: ' + text)
    shapes = list(dict.fromkeys(shapes))
    random.Random(args.seed).shuffle(shapes)
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        parser.error('build the executable first: ' + str(binary))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for folder in ['src','include','scripts'] for f in sorted((ROOT/folder).glob('*'))
                     if f.is_file()}
    metadata = {
        'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform': platform.platform(), 'python': sys.version,
        'arguments': vars(args), 'shape_order_mnk': shapes,
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source_sha256': source_hashes,
        'environment': {key: os.environ.get(key) for key in
                        ['CUDA_VISIBLE_DEVICES','NVIDIA_TF32_OVERRIDE','CUDA_LAUNCH_BLOCKING']},
        'git_head': capture(['git','rev-parse','HEAD']),
        'git_status': capture(['git','status','--porcelain']),
        'nvcc': capture(['nvcc','--version']),
        'gpu_before': capture(['nvidia-smi','-q']), 'commands': [],
        'timing': 'CUDA events, repeated GEMM device stream interval, excludes copies and allocation',
        'precision': 'FP32 custom FMA / cuBLAS CUBLAS_COMPUTE_32F_PEDANTIC',
    }
    cache = binary.parent/'CMakeCache.txt'
    if cache.exists():
        # Keep relevant compiler configuration without copying the entire local cache.
        metadata['build_settings'] = [line for line in cache.read_text().splitlines()
            if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CUDA_ARCHITECTURES:',
                                'CMAKE_CUDA_COMPILER:', 'CMAKE_CUDA_FLAGS', 'CMAKE_CXX_COMPILER:'))]
    def save():
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    save()
    try:
        for m,n,k in shapes:
            stem = f'{m}x{n}x{k}'
            command = [str(binary), '--m', str(m), '--n', str(n), '--k', str(k),
                       '--samples', str(args.samples), '--repeats', str(args.repeats),
                       '--warmup', str(args.warmup), '--seed', str(args.seed),
                       '--csv', str(output/(stem+'.csv'))]
            metadata['commands'].append(command)
            save()
            print('Running', stem, flush=True)
            with (output/(stem+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'failed'
        metadata['error'] = str(error)
        raise
    finally:
        metadata['gpu_after'] = capture(['nvidia-smi','-q'])
        metadata['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print('Saved validated raw measurements to', output)


if __name__ == '__main__':
    main()
