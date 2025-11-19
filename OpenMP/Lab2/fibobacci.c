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
 * @file fibonacci.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Recursive computation of Fibonacci
 * 
 * @see https://en.wikipedia.org/wiki/Fibonacci_number
 * @see http://algo.ing.unimo.it/people/andrea/Didattica/HPC/index.html
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#define F_30 832040LL
#define F_40 102334155LL
#define F_50 12586269025LL
#define F_60 1548008755920LL

static int N;
static int CUTOFF;

#define SEPARATOR "------------------------------------\n"

// Parse command line arguments to set solver parameters
void parse_arguments(int argc, char *argv[]);

// Fibonacci Golden Model - DO NOT CHANGE!
unsigned long long fibonacci_g(unsigned long long n)
{
    if (n < 2)
        return n;
    return fibonacci_g(n - 2) + fibonacci_g(n - 1);
}

// Run the Fibonacci
unsigned long long fib(unsigned long long n)
{
    if (n < 2)
        return n;
    int x,y;
    #pragma omp task shared(x)
    {
        x = fib(n-1);
    }
    #pragma omp task shared(y)
    {
        y = fib(n-2);
    }
    #pragma omp taskwait
    return x+y;
}

int main(int argc, char *argv[])
{
    parse_arguments(argc, argv);

    printf(SEPARATOR);
    printf("Number:       %d\n", N);
    printf("Cutoff:       %d\n", CUTOFF);
    printf(SEPARATOR);
    unsigned long long f_n;

    // Run Jacobi solver
    start_timer();

    #pragma omp parallel shared(f_n)
    {
        #pragma omp single 
        {
            f_n = fib(N);
        }
    }
    stop_timer();

    // Check error of final solution
    unsigned long long g_n;
    if (N == 30)
        g_n = F_30;
    else if (N == 40)
        g_n = F_40;
    else if (N == 50)
        g_n = F_50;
    else if (N == 60)
        g_n = F_60;
    else
        g_n = fibonacci_g(N);

    unsigned long long err = f_n - g_n;

    printf(SEPARATOR);
    printf("F(%d) = %llu\n", N, f_n);
    printf("Error = %llu\n", err);
    printf("Runtime = %lf ms\n", elapsed_ns() / 1E6);
    printf(SEPARATOR);

    return 0;
}

int parse_int(const char *str)
{
    char *next;
    int value = strtoul(str, &next, 10);
    return strlen(next) ? -1 : value;
}

double parse_double(const char *str)
{
    char *next;
    double value = strtod(str, &next);
    return strlen(next) ? -1 : value;
}

void parse_arguments(int argc, char *argv[])
{
    // Set default values
    N = 30;
    CUTOFF = 20;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--number") || !strcmp(argv[i], "-n"))
        {
            if (++i >= argc || (N = parse_int(argv[i])) < 0)
            {
                printf("Invalid matrix order\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--cutoff") || !strcmp(argv[i], "-c"))
        {
            if (++i >= argc || (CUTOFF = parse_int(argv[i])) < 0)
            {
                printf("Invalid seed\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            printf("\n");
            printf("Usage: ./jacobi [OPTIONS]\n\n");
            printf("Options:\n");
            printf("  -h  --help               Print this message\n");
            printf("  -c  --cutoff       C     Set task cutoff\n");
            printf("  -n  --number       N     Set the Fibonacci number\n");
            printf("\n");
            exit(0);
        }
        else
        {
            printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
            exit(1);
        }
    }
}