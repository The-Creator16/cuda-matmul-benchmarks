#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Row-major A[M,K] * B[K,N] -> C[M,N], one output per thread.
__global__ void matmul_naive(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int m, int n, int k) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row >= m || col >= n) return;
    float sum = 0.0f;
    for(int p=0; p<k; p=p+1) {
        sum = fmaf(a[static_cast<std::size_t>(row)*k+p],
                   b[static_cast<std::size_t>(p)*n+col], sum);
    }
    c[static_cast<std::size_t>(row)*n+col] = sum;
}

template<int Tile>
__global__ void matmul_tiled(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int m, int n, int k) {
    __shared__ float as[Tile][Tile];
    __shared__ float bs[Tile][Tile];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*Tile + ty;
    int col = blockIdx.x*Tile + tx;
    float sum = 0.0f;
    for(int base=0; base<k; base=base+Tile) {
        as[ty][tx] = (row < m && base+tx < k)
            ? a[static_cast<std::size_t>(row)*k+base+tx] : 0.0f;
        bs[ty][tx] = (base+ty < k && col < n)
            ? b[static_cast<std::size_t>(base+ty)*n+col] : 0.0f;
        // Even out-of-bounds output threads must reach both barriers.
        __syncthreads();
        #pragma unroll
        for(int p=0; p<Tile; p=p+1) {
            sum = fmaf(as[ty][p], bs[p][tx], sum);
        }
        __syncthreads(); // All consumers finish before the next tile overwrites data.
    }
    if(row < m && col < n) c[static_cast<std::size_t>(row)*n+col] = sum;
}
