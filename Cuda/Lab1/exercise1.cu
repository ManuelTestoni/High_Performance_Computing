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
 * @file exercise1.cu
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 1
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>


#ifndef N
#define N 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

/**
 * @brief  EX 1 - Launch CUDA kernel to "print" helloworld
 *
 * a) Detect global thread id. (tip: use threadIdx.x, blockDim.x, and blockIdx.x)
 * b) Explore thread execution and schedule changing: N={32} and M={4,8,16,32,64}
 * c) Add corner case management
 * 
 * @return void
 */

__global__ void helloworld(void)
{
    //mapping at the start of every blocks using it's first thread, meanwhile with thread id i'll select the specific thread
    int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid < N) {
        printf("Hello world, I am global thread %d (threadIdx=%d, blockIdx=%d, blockDim=%d)\n", gid, threadIdx.x, blockIdx.x, blockDim.x);

    }
}


int main(int argc, const char **argv)
{
    helloworld<<<(N / (BLOCK_SIZE-1)/ BLOCK_SIZE), BLOCK_SIZE>>>();

    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();
    return 0;
}
