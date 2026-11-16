# FP32 GEMM performance evaluation

## Experiment

The recorded run uses an NVIDIA GeForce RTX 3070 Ti Laptop GPU (CC 8.6), CUDA
runtime 13.2, driver API version 13.0, and cuBLAS version code 130401.
The run completed on 16 November 2026. Metadata records Fedora 43, Python 3.14.7,
and seed 42. Each variant receives five warmup iterations and five timing samples
of ten GEMMs each. All three matrix dimensions are equal in this sweep.

[Raw data and metadata](../results/quick-01/) accompany the
[full timing table](../results/quick-01/analysis/summary.md).

## Observations

- At size 64, tiled16 takes 0.00584 ms versus 0.00604 ms for naive: a modest 1.04× speedup.
- At size 128, tiled16 reaches 1.14× speedup over naive.
- At size 256, tiled16 takes 0.04116 ms versus 0.05151 ms for naive: 1.25× speedup and 815.12 GFLOP/s.
- Tiled32 is slower than tiled16 at every measured size. At size 64 it is also slower than naive.
- cuBLAS is slower than the custom kernels at size 64 in this run, but leads at sizes 128 and 256. At size 256 it reaches 2261.32 GFLOP/s, or 3.47× the naive throughput.

## Interpretation

Shared-memory tiling provides reuse of input values within a block. Tiled16's
increasing advantage is consistent with amortizing tile-loading and synchronization
costs over more useful work. Timing alone does not establish which hardware resource
limits execution.

Tiled32 uses 1024 threads per block versus 256 for tiled16. For a 64×64 output it
launches only four blocks, versus sixteen for tiled16. This reduces available block
parallelism and can contribute to its poor small-matrix performance. Register usage,
occupancy and scheduler effects require profiling before assigning causality.

At these small sizes, launch and submission costs can be significant. The size-64
cuBLAS result does not imply that the custom kernels generally outperform cuBLAS;
the library already leads by size 128 in the same experiment.

## Correctness

All recorded variants passed full-output comparisons against cuBLAS. CPU FP64
checks cover all outputs for sizes 64 and 128 and 256 sampled outputs for size 256.
The largest logged custom-kernel absolute difference from cuBLAS is 1.33514e-5.
The CTest GPU suite also passed 34 cases. See [verification](verification.md).

## Scope and limitations

These results describe one device-resident, repeated-GEMM run on one laptop GPU.
They exclude allocation and transfer time. Buffers are reused; cache reuse is part
of the workload. Five samples characterize within-run spread, not independent-run
reproducibility. No hardware-counter or sanitizer reports are included.

The next experiments are a larger matrix sweep, repeated runs under controlled
power/thermal conditions, and Nsight measurements of memory traffic, occupancy,
and scheduler stalls. The [methodology](methodology.md) defines the equations and
comparison conditions.
