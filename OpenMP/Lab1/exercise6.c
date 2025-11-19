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
 * @file exercise6.c
 * @author Alessandro Capotondi
 * @brief Exercise 6
 * 
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

#if !defined(W)
#define W (1 << 15)
#endif

/**
 * @brief EX 6 - Task Parallelism w/tasks
 * 
 * Distribute workload among 4 threads using task directive
 * Same program as EX 5, but we had to manually unroll the loop to use sections
 * 
 * a) Create a parallel region with 4 threads. Use SINGLE directive to allow
 * only one thread to execute the loop. Use TASK directive to outline tasks.
 * b) Parallelize using SECTION
 * 
 * @return void
 */
void exercise()
{
 #pragma omp parallel num_threads(NTHREADS)
{
    #pragma omp sections
    {
        #pragma omp section
        {
            DEBUG_PRINT("%hu: I am executing iteration 0!\n", omp_get_thread_num());
            work((0 + 1) * W);
        }

        #pragma omp section
        {
            DEBUG_PRINT("%hu: I am executing iteration 1!\n", omp_get_thread_num());
            work((1 + 1) * W);
        }

        #pragma omp section
        {
            DEBUG_PRINT("%hu: I am executing iteration 2!\n", omp_get_thread_num());
            work((2 + 1) * W);
        }

        #pragma omp section
        {
            DEBUG_PRINT("%hu: I am executing iteration 3!\n", omp_get_thread_num());
            work((3 + 1) * W);
        }
    }
}
}

                                                                                                                                                89,0-1        Bot
