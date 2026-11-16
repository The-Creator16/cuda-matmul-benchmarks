# Verification

## Recorded GPU validation

The supplied CTest log records passing `host_tests` and `gpu_correctness` tests
on 16 November 2026. The GPU suite reports 34 cases across four implementations;
its recorded duration is approximately 11.39 seconds.

The completed `quick-01` run validates each implementation before timing.
[Per-shape logs](../results/quick-01/) record comparison counts and maximum errors.

## Host and tooling checks

During initial development, the CPU-only build, strict compiler-warning check,
12 Python unit/integration tests, Python/Bash syntax checks and plot-generation
smoke test passed. Synthetic fixtures used for tooling tests are not performance
measurements.

## Additional checks

Compute Sanitizer and Nsight reports are not included in this dataset.
The available commands are:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/matmul_bench --m 33 --n 65 --k 35 --validate-only
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/matmul_bench --m 33 --n 65 --k 35 --validate-only
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/matmul_bench --m 33 --n 65 --k 35 --validate-only
```

Numerical tests use tolerances and do not prove correctness for all inputs.
Large-matrix CPU checks are sampled; cuBLAS comparisons cover the full output.
