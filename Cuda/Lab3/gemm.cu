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
 * @file gemm.cu
 * @author Alessandro Capotondi
 * @date 12 May 2020
 * @brief GEMM Kernel
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define gpuErrchk(ans)                        \
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
#define TILE_W 128
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define SM 64
static void reorder(float *__restrict__ a, float *__restrict__ b, int n)
{
    for (int i = 0; i < SM; i++)
        for (int j = 0; j < SM; j++)
            b[i * SM + j] = a[i * n + j];
}

static void mm(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{
    for (int i = 0; i < SM; i++)
    {
        for (int k = 0; k < SM; k++)
        {
            for (int j = 0; j < SM; j++)
            {
                c[i * n + j] += a[i * n + k] * b[k * SM + j];
            }
        }
    }
}
void gemm_host(float *a, float *b, float *c, int n)
{
    int bk = n / SM;
#pragma omp parallel for collapse(3)
    for (int i = 0; i < bk; i++)
    {
        for (int j = 0; j < bk; j++)
        {
            for (int k = 0; k < bk; k++)
            {
                float b2[SM * SM];
                reorder(&b[SM * (k * n + j)], b2, n);
                mm(&a[SM * (i * n + k)], b2, &c[SM * (i * n + j)], n);
            }
        }
    }
}
__global__ void gemm(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int ib = blockIdx.y;
    int jb = blockIdx.x;

    int it = threadIdx.y;
    int jt = threadIdx.x;

    int a_offset, b_offset, c_offset;

    float Cvalue = 0.0f;
    for (int kb = 0; kb < (n / BLOCK_SIZE); ++kb)
    {
        a_offset = ib * n * BLOCK_SIZE + kb * BLOCK_SIZE;
        b_offset = kb * n * BLOCK_SIZE + jb * BLOCK_SIZE;
        As[it][jt] = a[a_offset + it * n + jt];
        Bs[it][jt] = b[b_offset + it * n + jt];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += As[it][k] * Bs[k][jt];

        __syncthreads();
    }

    c_offset = ib * n * BLOCK_SIZE + jb * BLOCK_SIZE;
    c[c_offset + it * n + jt] = Cvalue;
}

int main(int argc, char *argv[])
{
    int n = N, iret = 0;
    float *a, *b, *c, *g;
    struct timespec rt[2];
    double wt; // walltime

    if (argc > 1)
        n = atoi(argv[1]);

    //TODO Update malloc to cudaMallocHost or cudaMallocManaged (if necessary)
    //cudaMallocHost(&a, sizeof(*a) * n * n);                      // io per pinned
    cudaMallocManaged(&a, sizeof(*a) * n * n);                      // io per UVM
    //cudaMallocHost(&b, sizeof(*b) * n * n);                    // io per pinned
    cudaMallocManaged(&b, sizeof(*b) * n * n);                    // io per UVM
    //cudaMallocHost(&c, sizeof(*c) * n * n);                    // io per pinned
    cudaMallocManaged(&c, sizeof(*c) * n * n);                    // io per UVM
    /*
    if (NULL == (a = (float *)malloc(sizeof(*a) * n * n)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    //TODO Update malloc to cudaMallocHost or cudaMallocManaged (if necessary)    
    if (NULL == (b = (float *)malloc(sizeof(*b) * n * n)))
    {
        printf("error: memory allocation for 'y'\n");
        iret = -1;
    }
    //TODO Update malloc to cudaMallocHost or cudaMallocManaged (if necessary)    
    if (NULL == (c = (float *)malloc(sizeof(*c) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }
    */
    if (NULL == (g = (float *)malloc(sizeof(*g) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }

    if (0 != iret)
    {
        //TODO Update cudaFreeHost or cudaFree (if necessary)
        //free(a);
        //cudaFreeHost(a);                                          // io per pinned
        cudaFree(a);                                          // io per UVM
        //TODO Update cudaFreeHost or cudaFree (if necessary)        
        //free(b);
        //cudaFreeHost(b);                                          // io per pinned
        cudaFree(b);                                          // io per UVM
        //TODO Update cudaFreeHost or cudaFree (if necessary)        
        //free(c);
        //cudaFreeHost(c);                                          // io per pinned
        cudaFree(c);                                          // io per UVM
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
    gemm_host(a, b, g, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    //TODO Remove if unecessary
    //float *d_a, *d_b, *d_c;                                       // tolgo in uvm
    //gpuErrchk(cudaMalloc((void **)&d_a, sizeof(float) * n * n)); // tolgo in uvm
    //gpuErrchk(cudaMalloc((void **)&d_b, sizeof(float) * n * n));
    //gpuErrchk(cudaMalloc((void **)&d_c, sizeof(float) * n * n));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    //TODO Remove if unecessary
    //gpuErrchk(cudaMemcpy(d_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));
    //gemm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n); io per uvm
    gemm<<<dimGrid, dimBlock>>>(a, b, c, n); // io per uvm
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); // sincronizzo qui per UVM
    //TODO Remove if unecessary
    //gpuErrchk(cudaMemcpy(c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost)); tolgo per uvm
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM-v1 (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    for (int i = 0; i < n * n; i++)
    {
        iret = *(int *)(g + i) ^ *(int *)(c + i);
        assert(iret == 0);
    }

    //TODO Update cudaFreeHost or cudaFree (if necessary)
    //free(a);
    //cudaFreeHost(a);                                          // io per pinned
    cudaFree(a);                                          // io per UVM
    //TODO Update cudaFreeHost or cudaFree (if necessary)
    //free(b);
    //cudaFreeHost(b);                                          // io per pinned
    cudaFree(b);                                          // io per UVM
    //TODO Update cudaFreeHost or cudaFree (if necessary)
    //free(c);
    //cudaFreeHost(c);                                          // io per pinned
    cudaFree(c);                                          // io per UVM
    free(g);
    //TODO Remove if unecessary
    //gpuErrchk(cudaFree(d_a));             // tolgo in uvm
    //TODO Remove if unecessary
    //gpuErrchk(cudaFree(d_b));             // tolgo in uvm
    //TODO Remove if unecessary
    //gpuErrchk(cudaFree(d_c));             // tolgo in uvm
    return 0;
}
