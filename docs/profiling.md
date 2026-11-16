# GPU profiling

Run the correctness suite before profiling. Use `scripts/profile.sh` for one
representative medium/large size, and optionally repeat with a small size and a
non-aligned size. Save the GPU model, Toolkit version, `ncu --version`, launch
shape, and profiler report with each observation.

`--profile` is a dedicated mode: warm up, synchronize, call `cudaProfilerStart`,
issue one GEMM, synchronize, and call `cudaProfilerStop`. It does not produce a
benchmark CSV or perform validation; the shell script validates first. `ncu
--profile-from-start off` captures only that range. cuBLAS may launch more than
one kernel within a single operation; inspect all relevant launches.

## What to inspect

| Question | Nsight evidence | Interpretation constraint |
|---|---|---|
| Does tiling reduce device-memory traffic? | Memory Workload Analysis, DRAM bytes, L1/L2 traffic/hit rates | Compare bytes for the same GEMM; logical source loads are not DRAM bytes |
| Is the kernel memory- or compute-limited? | Speed Of Light, compute and memory throughput, roofline when available | Low utilization alone is insufficient: launch overhead and dependencies also matter |
| Why does 16 differ from 32? | Launch Statistics, registers/thread, shared memory/block, achieved occupancy, block/warp limits | Occupancy is one constraint, not an objective by itself |
| Is synchronization expensive? | Scheduler/Warp State sections and barrier stalls | Stalls are diagnostic evidence, not additive wall-clock percentages |
| Are memory accesses efficient? | Global load/store sectors and shared-memory bank conflicts | Some accesses are broadcasts; don't equate shared memory with conflict-free access |
| Why is cuBLAS faster/slower? | Kernel count, launch shape, resource use, throughput | An internal kernel name alone does not prove its algorithm |

Section/metric availability depends on the installed Nsight version and GPU:

```bash
ncu --list-sections
ncu --query-metrics
```

The script uses `--set full` to avoid hardcoding architecture-specific metric
names. It can take considerably longer than a normal run. Profiler replay,
cache control, clock control, and instrumentation can alter conditions. Record
those settings; use unprofiled event timings for the performance tables.

If Nsight reports `ERR_NVGPUCTRPERM`, GPU performance-counter access is restricted
by the machine administrator. On a university/cloud machine, request authorized
profiling access. Benchmarking may still work, but do not claim bottlenecks were
confirmed by profiling when counters were unavailable.

## Reasoning sequence

1. State the observation from an uninstrumented run, with shape and spread.
2. Explain the relevant design difference (loads, reuse, barriers, resources).
3. Cite at least one matching profiler observation.
4. State what remains uncertain, particularly if kernels are launch-bound or
   power/thermal conditions differ.

For example, "shared-memory tiling should reduce repeated loads" is a hypothesis.
"Tiled16 reduced measured DRAM read bytes by [measured value] at [shape]" is a
result only after those counters have actually been collected.
