# Benchmark results

[`quick-01`](quick-01/) contains a completed FP32 run on an NVIDIA GeForce RTX
3070 Ti Laptop GPU: square sizes 64, 128 and 256; five samples of ten GEMMs per
variant. Raw CSV files, validation logs, metadata and the generated summary are
retained together.

- [Summary](quick-01/analysis/summary.md)
- [Performance interpretation](../docs/performance_report.md)
- [Measurement protocol](../docs/methodology.md)

Generated directories are ignored by Git by default. The reviewed dataset can
be tracked with `git add -f results/quick-01`. Profiler reports may be distributed
separately because of their size.
