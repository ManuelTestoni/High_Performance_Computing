
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
 * @file exercise4.c
 * @author Alessandro Capotondi
 * @brief Exercise 4
 * 
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

#if !defined(W)
#define W (1000) //1, 10, 100, 1000
#endif

/**
 * @brief EX 4 - Data parallelism: balanced small parallel loop ** 4 THREADS **
 * 
 * Parallelize loop with static and dynamic scheduling, for chunks of 32, 16, 8, 4, 1
 * and for W 1, 10, 100, 1000
 * 
 * @return void
 */
void exercise()
{
    #pragma omp parallel for schedule(dynamic,32)
    for (int i = 0; i < 1024 * 256; i++)
    {
        DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
        work(W);
    }
}
                                                                                                                                                61,1          Bot
