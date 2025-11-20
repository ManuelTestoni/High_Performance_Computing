/*
 * BSD 2-Clause License
 * 
 * Copyright (c) 2020, Alessandro Capotondi
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file exercise3.cu
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 3 - CUDA MATMUL
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define gpuErrchk(ans) 
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

extern "C"
{
#include "utils.h"
}

#define TWO02 (1 << 2)
#define TWO04 (1 << 4)
#define TWO08 (1 << 8)


#ifndef N
#define N (1 << 10)
#endif
#ifndef TILE_W
#define TILE_W 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

void gemm(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int n)
{

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[i * n + k] * b[k *n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/**
 * @brief  EX 3 - Complete Matrix Multiplication
 */
__global__ void gemm_kernel(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{
    //TODO: Add GEMM kernel body
    int row = (blockIdx.x * blockDim.x * TILE_W) + (threadIdx.x * TILE_W);
    int col = (blockIdx.y * blockDim.y * TILE_W) + (threadIdx.y * TILE_W);
    int end_row = row+TILE_W < n ? row+TILE_W : n;
    int end_col = col+TILE_W < n ? col+TILE_W : n;

    for(int i = row; i < end_row; ++i) {
            for(int j = col; j < end_col; ++j) {
                    float sum = 0.0;
                    for (int k = 0; k < n; ++k) {
                            sum+= a[i*n+k] * b[k*n+j];
                    }
                    c[i*n+j] = sum;
            }
    }
}

int main(int argc, char *argv[])
{
    int n = N, iret = 0;
    float *a, *b, *c, *g;
    struct timespec rt[2];
    double wt; // walltime

    if (argc > 1)
        n = atoi(argv[1]);

    if (NULL == (a = (float *)malloc(sizeof(*a) * n * n)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (b = (float *)malloc(sizeof(*b) * n * n)))
    {
        printf("error: memory allocation for 'y'\n");
        iret = -1;
    }
    if (NULL == (c = (float *)malloc(sizeof(*c) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }
    if (NULL == (g = (float *)malloc(sizeof(*g) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }

    if (0 != iret)
    {
        free(a);
        free(b);
        free(c);
        free(g);
        exit(EXIT_FAILURE);
    }


    //Init Data
    int _b = rand() % TWO04;
    int _c = rand() % TWO08;
#pragma omp parallel for
    for (int i = 0; i < n * n; i++)
    {
        a[i] = _b / (float)TWO02;
        b[i] = _c / (float)TWO04;
        c[i] = g[i] = 0.0;
    }

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gemm(a, b, g, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    //CUDA Buffer Allocation
    float *d_a, *d_b, *d_c;
    //TODO: Add HERE Cuda data allocation
    gpuErrchk(cudaMalloc((void **)&d_a, sizeof(float) * n*n));
    gpuErrchk(cudaMalloc((void **)&d_b, sizeof(float) * n*n));
    gpuErrchk(cudaMalloc((void **)&d_c, sizeof(float) * n*n));
    clock_gettime(CLOCK_REALTIME, rt + 0);
    //TODO: Add HERE Cuda data transfer
    cudaMemcpy(d_a, a, sizeof(float) * n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * n*n, cudaMemcpyHostToDevice);

    //TODO: Add HERE dimBlock
    //two argument that we pass to the kernel function, this could be 2D or 3D
    dim3 dimBlock(TILE_W, TILE_W);
    dim3 dimGrid((n+TILE_W-1)/TILE_W,(n+TILE_W-1)/TILE_W);
    //TODO: Add HERE dimGrid
    //TODO: Add HERE kernel launch
    gemm_kernel<<<dimGrid, dimBlock>>>(d_a,d_b,d_c,n);
    gpuErrchk(cudaPeekAtLastError());
    //TODO: Add HERE Cuda data transfer
    cudaMemcpy(c, d_c, sizeof(float)* n*n, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    for (int i = 0; i < n * n; i++)
    {
        iret = *(int *)(g + i) ^ *(int *)(c + i);
        assert(iret == 0);
    }
    free(a);
    free(b);
    free(c);
    free(g);
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));

    return 0;
}