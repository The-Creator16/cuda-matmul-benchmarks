# CUDA Matrix Multiplication Benchmarks

FP32 matrix multiplication in CUDA C++, exploring shared-memory reuse, tile size,
and performance relative to cuBLAS. The project includes numerical validation,
reproducible timing, raw measurements, and a profiling workflow.

## Implementations

All implementations compute row-major **C[M,N] = A[M,K] × B[K,N]**.

| Variant | Implementation |
|---|---|
| `naive` | One output per thread; 16×16 blocks; global-memory reads |
| `tiled16` | Cooperative shared-memory loading; 16×16 tiles |
| `tiled32` | Cooperative shared-memory loading; 32×32 tiles |
| `cublas` | `cublasGemmEx` with pedantic FP32 computation |

Partial tiles and rectangular matrices are supported. Custom kernels use FP32
FMA; cuBLAS uses `CUBLAS_COMPUTE_32F_PEDANTIC`. TF32 is disabled for this comparison.

## Results

Measured on an **NVIDIA GeForce RTX 3070 Ti Laptop GPU**, compute capability 8.6,
with CUDA runtime 13.2. Five samples per variant, ten GEMMs per sample, five warmup
iterations; seed 42. Timings exclude allocation and host/device transfers.

| Square size | Naive (ms) | Tiled16 (ms) | Tiled32 (ms) | cuBLAS (ms) | Tiled16 speedup vs naive |
|---|---:|---:|---:|---:|---:|
| 64 | 0.00604 | 0.00584 | 0.00922 | 0.00983 | 1.04× |
| 128 | 0.01280 | 0.01126 | 0.01546 | 0.00973 | 1.14× |
| 256 | 0.05151 | 0.04116 | 0.05028 | 0.01484 | 1.25× |

Values are medians. Tiled16 outperforms tiled32 at all three measured sizes.
At 256×256, cuBLAS achieves 2261 GFLOP/s versus 815 GFLOP/s for tiled16.
This initial sweep covers small matrices; it does not establish large-matrix
scaling or a profiler-confirmed bottleneck.

[Full analysis](docs/performance_report.md) · [Raw results](results/quick-01/) ·
[Methodology](docs/methodology.md) · [Validation](docs/verification.md)

## Build and test

Requirements: NVIDIA GPU, CUDA Toolkit with cuBLAS, a compatible C++17 compiler,
CMake 3.24+, and Python 3.9+. The documented workflow targets Linux.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

For an explicit RTX 3070 Ti target, use `-DCMAKE_CUDA_ARCHITECTURES=86`.
The GPU suite covers 34 cases across all four implementations, including
rectangles, tile boundaries, multiple seeds and structured inputs.

## Reproduce the recorded experiment

```bash
python3 scripts/run_benchmarks.py --output results/reproduction-01 --sizes 64 128 256 --shapes --samples 5 --repeats 10
python3 scripts/analyze.py results/reproduction-01 --no-plots
```

Output directories must be new. Each run records raw samples, validation logs,
source/binary hashes and environment metadata. To run the larger default sweep:

```bash
python3 scripts/run_benchmarks.py --output results/full-01
```

## Generate figures

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/analyze.py results/quick-01
```

The analysis script creates latency, throughput and speedup plots in PNG/SVG,
plus Markdown and CSV summaries. Latency error bars represent sample min–max.

## Profile

```bash
bash scripts/profile.sh build/matmul_bench results/profile-01 1024
```

See [profiling](docs/profiling.md) for collection and interpretation. Profiler
reports are not included in the recorded experiment. Use uninstrumented event
timings for benchmark comparisons.

## Project structure

| Path | Contents |
|---|---|
| `include/kernels.cuh` | Baseline and shared-memory kernels |
| `include/host.hpp` | Input generation, FP64 reference, numerical checks |
| `src/main.cu` | CUDA/cuBLAS execution, validation and event timing |
| `scripts/` | Experiment runner, validation, analysis and profiling |
| `tests/` | Host and Python tests |
| `docs/` | Methodology, results and verification |
| `results/quick-01/` | Recorded measurements and environment metadata |

GitHub Actions checks host tests and CUDA compilation; GPU runtime validation
requires a CUDA-capable machine. Generated results are ignored by default;
reviewed runs can be tracked with `git add -f results/quick-01`.

## References

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## License

[MIT](LICENSE) — Adilbek Taizhanov
