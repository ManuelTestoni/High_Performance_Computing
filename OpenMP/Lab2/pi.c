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
 * @file exercise7.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 8
 * 
 * Pi calculation
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"
#if !defined(ITERS)
#define ITERS (4)
#endif

#define NSTEPS 134217728

void exercise(){
    long i;
    double dx = 1.0 / NSTEPS;
    double pi = 0.0;

    double start_time = omp_get_wtime();

    #pragma omp parallel for shared(pi)
    for (i = 0; i < NSTEPS; i++)
    {
        double x = (i + 0.5) * dx;
        double tmp = 1.0 / (1.0 + x * x);
        #pragma omp atomic
        pi += tmp;
    }
    pi *= 4.0 * dx;

    double run_time = omp_get_wtime() - start_time;
    double ref_pi = 4.0 * atan(1.0);
    printf("pi with %d steps is %.10f in %.6f seconds (error=%e)\n",
           NSTEPS, pi, run_time, fabs(ref_pi - pi));
}

int
main(int argc, char** argv)
{
    for(int i=0; i<ITERS; i++){
        printf("\n\n");
        printf("============================\n");
        printf("Test - Iteration %d...\n", i);
        printf("============================\n");
        start_stats();
        exercise();
        collect_stats();
    }
    printf("\n\n");
    printf("============================\n");
    printf("Statistics\n");
    printf("============================\n");
    print_stats();
    return 0;
}
