#!/usr/bin/env python3
"""Summarize real raw measurements; no built-in or fabricated benchmark data."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

VARIANTS = ['naive','tiled16','tiled32','cublas']


def load_run(directory):
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['status'] != 'complete':
        raise ValueError('Refusing to summarize an incomplete/failed run')
    expected = {tuple(s) for s in metadata['shape_order_mnk']}
    samples = metadata['arguments']['samples']
    groups = {}
    for m,n,k in sorted(expected):
        with (directory/f'{m}x{n}x{k}.csv').open(newline='') as file:
            for row in csv.DictReader(file):
                shape = tuple(int(row[x]) for x in ['m','n','k'])
                variant = row['variant']
                if shape != (m,n,k) or variant not in VARIANTS or row['validated'] != '1':
                    raise ValueError('Wrong shape, unknown variant, or unvalidated data')
                ms = float(row['ms'])
                if not math.isfinite(ms) or ms <= 0:
                    raise ValueError('Timing must be finite and positive')
                if int(row['seed']) != metadata['arguments']['seed'] or row['pattern'] != 'random':
                    raise ValueError('Input configuration does not match run metadata')
                if int(row['repeats']) != metadata['arguments']['repeats']:
                    raise ValueError('Repeat count does not match run metadata')
                key = (shape,variant)
                groups.setdefault(key, []).append((int(row['sample']),ms))
    result = []
    for shape in sorted(expected, key=lambda s: (s[0]*s[1]*s[2],s)):
        medians = {}
        for variant in VARIANTS:
            values = groups.get((shape,variant), [])
            if sorted(i for i,_ in values) != list(range(samples)):
                raise ValueError(f'Missing or duplicate samples: {shape}, {variant}')
            medians[variant] = statistics.median(t for _,t in values)
        for variant in VARIANTS:
            times = [t for _,t in groups[(shape,variant)]]
            median = medians[variant]
            q1,_,q3 = statistics.quantiles(times,n=4,method='inclusive') if len(times)>1 else [median]*3
            m,n,k = shape
            result.append(dict(m=m,n=n,k=k,variant=variant,median_ms=median,
                               min_ms=min(times),max_ms=max(times),q1_ms=q1,q3_ms=q3,
                               gflops=2*m*n*k/(median*1e6),
                               speedup_vs_naive=medians['naive']/median,
                               percent_cublas=100*medians['cublas']/median))
    return result


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shapes = list(dict.fromkeys((r['m'],r['n'],r['k']) for r in rows))
    labels = [f'{m}×{n}×{k}' for m,n,k in shapes]
    colors = ['#64748b','#0284c7','#059669','#d97706']
    for field,title,ylabel in [('median_ms','Matrix multiplication latency','Milliseconds (log scale)'),
                               ('gflops','FP32 matrix multiplication throughput','GFLOP/s'),
                               ('speedup_vs_naive','Speedup relative to naive kernel','Speedup (×, log scale)')]:
        fig,ax = plt.subplots(figsize=(max(8,len(shapes)*0.85),4.8),layout='constrained')
        for name,color in zip(VARIANTS,colors):
            data = [next(r for r in rows if (r['m'],r['n'],r['k'])==s and r['variant']==name) for s in shapes]
            values = [r[field] for r in data]
            if field == 'median_ms':
                errors = [[r['median_ms']-r['min_ms'] for r in data],
                          [r['max_ms']-r['median_ms'] for r in data]]
                ax.errorbar(range(len(shapes)),values,yerr=errors,label=name,color=color,
                            marker='o',capsize=3,linewidth=1.5)
            else:
                ax.plot(range(len(shapes)),values,label=name,color=color,marker='o')
        if field in ['median_ms','speedup_vs_naive']:
            ax.set_yscale('log')
        ax.set_xticks(range(len(shapes)),labels,rotation=35,ha='right')
        ax.set_xlabel('M × N × K (ordered by operation count)')
        ax.set_ylabel(ylabel)
        ax.set_title(title + ('\nMedian; error bars show sample min–max' if field=='median_ms' else ''))
        ax.grid(axis='y',alpha=0.25)
        ax.legend()
        fig.savefig(out/(field+'.png'),dpi=180)
        fig.savefig(out/(field+'.svg'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--no-plots',action='store_true')
    args = parser.parse_args()
    rows = load_run(args.run)
    out = args.run/'analysis'
    out.mkdir(exist_ok=True)
    with (out/'summary.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Measured results', '', 'Device-resident FP32 GEMM; no transfer/allocation time.',
             'Latency is the median of batch-average samples. Spread is not a confidence interval.', '',
             '| M×N×K | Variant | Median ms | Min–max ms | GFLOP/s | Speedup vs naive | % cuBLAS throughput |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['m']}×{r['n']}×{r['k']} | {r['variant']} | {r['median_ms']:.5f} | "
                     f"{r['min_ms']:.5f}–{r['max_ms']:.5f} | {r['gflops']:.2f} | "
                     f"{r['speedup_vs_naive']:.2f}× | {r['percent_cublas']:.1f}% |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    if not args.no_plots:
        plot(rows,out)
    print(out/'summary.md')


if __name__ == '__main__':
    main()
