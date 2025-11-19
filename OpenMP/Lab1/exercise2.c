
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
 * @file exercise2.c
 * @author Alessandro Capotondi
 * @brief Exercise 2
 * 
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

/**
 * @brief EX 2 - Data parallelism: parallel loop ** 1 to 16 THREADS **
 *
 * a) Parallelize loop w/static scheduling
 * b) Parallelize loop w/dynamic scheduling for chunks = NITER/NTHR (same as static) and 1 (finest granularity)
 * c) Same as 2a + 2b, with 1<<20 loop iterations and work(10)
 * 
 * @return void
 */
void exercise()
{
    for (int i = 0; i < 1 << 10; i++)
    {
        DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
        work(1 << 10);
        DEBUG_PRINT("%hu: Done with iteration %hu!\n", omp_get_thread_num(), i);
    }
}
                                                                                                                                                58,1          Bot
