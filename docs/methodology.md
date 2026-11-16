# Measurement and correctness methodology

## Problem

For row-major matrices A ∈ R^(M×K), B ∈ R^(K×N), compute

C[i,j] = Σ(k=0..K−1) A[i,k] B[k,j].

Alpha is 1 and beta is 0. C is overwritten on every invocation. All dimensions
are positive and limited to 16384 by the CLI. Index products use `size_t`.

## Reference and tolerance

Inputs use an explicitly defined mapping from `std::mt19937` integers to FP32
values in [-1,1), so the data does not depend on `uniform_real_distribution`.

1. Compute FP64 CPU dot products for all output elements when M×N×K ≤ 4,000,000.
2. For larger problems, check 256 unique elements, including corners and central
   positions; the rest are seeded samples. This sampling is not exhaustive.
3. Compute a cuBLAS FP32 reference, check it against the CPU reference, and reject
   any non-finite element in its full output.
4. Check every element of each selected implementation against cuBLAS, and the
   same CPU reference elements independently. Output is prefilled with NaNs before
   each correctness invocation to expose missing writes.
5. Recheck the final result left by the timed sequence against cuBLAS.

An element passes when

`abs(actual − reference) ≤ 1e−3 + 1e−4 × abs(reference)`.

Both values must be finite. CSV records absolute error and the maximum error
normalized by this tolerance, plus CPU check count/error. A scaled error >1 fails.
The tolerance accommodates FP32 accumulation-order differences for this input
range; it is not a universal guarantee for arbitrary conditioning or magnitude.
Do not enlarge the tolerance automatically to conceal a failing kernel.

Zero, ones, identity-like, and alternating-sign patterns supplement random tests.
Dimensions around 16 and 32 exercise partial tiles and barrier participation.
Compute Sanitizer checks remain necessary; numerical tests alone do not prove
absence of memory errors or races.

## cuBLAS row-major mapping

cuBLAS interprets data as column-major. Row-major A[M,K] has the same byte layout
as column-major Aᵀ[K,M]. Likewise B maps to Bᵀ[N,K] and C to Cᵀ[N,M]. Thus invoke:

`cublasGemmEx(N, M, K, B, lda=N, A, ldb=K, C, ldc=N)`

with both operations `CUBLAS_OP_N`. The API call calculates BᵀAᵀ = (AB)ᵀ in the
same output bytes. The code uses `CUBLAS_COMPUTE_32F_PEDANTIC`, with FP32 A/B/C.
There is no TF32 or mixed-precision comparison hidden in the baseline.

## Timing protocol

- One CUDA nonblocking stream, with the cuBLAS handle bound to the same stream
- Context creation, handle creation, allocations, copies and validation outside timing
- Five warmup GEMMs per variant by default
- Eleven rounds by default, with variant order shuffled in each round
- Each sample brackets 20 repeated GEMMs with CUDA events on that stream
- Synchronize the stop event, read elapsed time, divide by repeat count
- Retain all samples; report median and min–max, plus quartiles in summary CSV

This measures the device stream interval, including any device idle gaps caused
by submission overhead. Batching amortizes event overhead but is not a substitute
for CUDA Graphs. Small matrices may be launch-bound. The same buffers are reused
and data may remain in cache. No cache flushing is attempted. Sizes are shuffled
in a reproducible order; between-run thermal drift still needs monitoring.

No runtime allocation/copy is intentionally included in the timed host code;
cuBLAS remains responsible for its own internal algorithm and workspace behavior.
The executable reserves 15% of available memory as headroom before matrix
allocation; it still checks actual allocation/library failures.

## Derived metrics

Use the conventional GEMM operation count (one FMA counts as two operations):

- FLOPs = 2MNK (conventional count, rather than MN(2K−1))
- GFLOP/s = 2MNK / (time_ms × 10^6)
- speedup = median_time_naive / median_time_variant
- % cuBLAS throughput = 100 × median_time_cublas / median_time_variant

The last two ratios use medians for the same shape and run. They are not an
end-to-end CPU-to-GPU speedup and not a speedup against another GPU.

## Analytical memory-reuse model

For one full T×T output tile and one K step of length T:

- Work: 2T³ operations
- Cooperative input loads: 2T² floats = 8T² bytes
- Idealized input-load arithmetic intensity: 2T³/(8T²) = T/4 FLOP/byte

This ignores output stores, padding, cache behavior, synchronization, instructions,
and reuse across blocks. It describes requested loads at this tiling level,
**not measured DRAM traffic**. A naive source-level load count is also not a
DRAM bandwidth measurement: caches and broadcasts can serve repeated loads.
For long K, including one output store per tile gives
`2T²K / (8TK + 4T²)`, approaching T/4 in this simplified model.

T=32 increases input reuse relative to T=16 but also uses 1024 threads versus 256
and 8 KiB versus 2 KiB of shared memory per block. Registers per thread and
architecture-specific limits can restrict resident blocks/warps. More reuse does
not guarantee lower latency. Confirm with profiling.

## Limits

No Tensor Cores, register tiling, asynchronous copies, streams overlapping data
transfers, multi-GPU execution, energy measurement, or roofline calibration are
implemented. The tiled kernels are deliberately small enough to explain and
inspect. cuBLAS can use much more sophisticated scheduling and tiling.
