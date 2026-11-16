#!/usr/bin/env bash
set -euo pipefail
binary=${1:-build/matmul_bench}
out=${2:-results/profile}
size=${3:-1024}
command -v ncu >/dev/null || { echo 'Nsight Compute (ncu) is required' >&2; exit 1; }
if [[ -e "$out" ]]; then
    echo "Output already exists: $out (use a new directory)" >&2
    exit 1
fi
mkdir -p "$out"
ncu --version > "$out/ncu-version.txt"
"$binary" --m "$size" --n "$size" --k "$size" --validate-only > "$out/validation.log" 2>&1
for kernel in naive tiled16 tiled32 cublas; do
    ncu --profile-from-start off --set full --export "$out/$kernel" \
        "$binary" --m "$size" --n "$size" --k "$size" --kernel "$kernel" --profile \
        > "$out/$kernel.log" 2>&1
done
echo "Reports: $out/*.ncu-rep; open with ncu-ui. Do not use profiled timings as benchmark results."
