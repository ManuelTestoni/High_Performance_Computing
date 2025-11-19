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
 * @file jacobi.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief This code solves the steady state heat equation on a rectangular region.
 * This code solves the steady state heat equation on a rectangular region.
 *  The sequential version of this program needs approximately
 *  18/epsilon iterations to complete. 
 *  The physical region, and the boundary conditions, are suggested
 *  by this diagram;
 *                 W = 0
 *           +------------------+
 *           |                  |
 *  W = 100  |                  | W = 100
 *           |                  |
 *           +------------------+
 *                 W = 100
 *  The region is covered with a grid of M by N nodes, and an N by N
 *  array W is used to record the temperature.  The correspondence between
 *  array indices and locations in the region is suggested by giving the
 *  indices of the four corners:
 *                I = 0
 *        [0][0]-------------[0][N-1]
 *           |                  |
 *    J = 0  |                  |  J = N-1
 *           |                  |
 *      [M-1][0]-----------[M-1][N-1]
 *                I = M-1
 *  The steady state solution to the discrete heat equation satisfies the
 *  following condition at an interior grid point:
 *    W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )
 *  where "Central" is the index of the grid point, "North" is the index
 *  of its immediate neighbor to the "north", and so on.
 * 
 *  Given an approximate solution of the steady state heat equation, a
 *  "better" solution is given by replacing each interior point by the
 *  average of its 4 neighbors - in other words, by using the condition
 *  as an ASSIGNMENT statement:
 *    W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )
 *  If this process is repeated often enough, the difference between successive 
 *  estimates of the solution will go to zero.
 *  This program carries out such an iteration, using a tolerance specified by
 *  the user, and writes the final estimate of the solution to a file that can
 *  be used for graphic processing.
 * icensing:
 *  This code is distributed under the GNU LGPL license. 
 * odified:
 *  18 October 2011
 * uthor:
 *  Original C version by Michael Quinn.
 *  This C version by John Burkardt.
 * eference:
 *  Michael Quinn,
 *  Parallel Programming in C with MPI and OpenMP,
 *  McGraw-Hill, 2004,
 *  ISBN13: 978-0071232654,
 *  LC: QA76.73.C15.Q55.
 * ocal parameters:
 *  Local, double DIFF, the norm of the change in the solution from one iteration
 *  to the next.
 *  Local, double MEAN, the average of the boundary values, used to initialize
 *  the values of the solution in the interior.
 *  Local, double U[M][N], the solution at the previous iteration.
 *  Local, double W[M][N], the solution computed at the latest iteration.
 * 
 * 
 * @see https://en.wikipedia.org/wiki/Jacobi_method
 * @see http://algo.ing.unimo.it/people/andrea/Didattica/HPC/index.html
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <inttypes.h>

#include "utils.h"

static int N;
static int MAX_ITERATIONS;
static int SEED;
static double CONVERGENCE_THRESHOLD;
static FILE *data;

#define SEPARATOR "------------------------------------\n"
// Return the current time in seconds since the Epoch
double get_timestamp();

// Parse command line arguments to set solver parameters
void parse_arguments(int argc, char *argv[]);

// Run the Jacobi solver
// Returns the number of iterations performed
int run(double *A, double *xtmp)
{
    int iter = 0, iterations_print = 1;
    double err = 0.0;

    for (int i = 0; i < N; i++)
    {
        memcpy(&xtmp[i * N], &A[i * N], (size_t)(N) * sizeof(double));
        // for (int j = 1; j < N - 1; j++)
        // {
        //     A[i * N + j] = xtmp[i * N + j];
        // }
    }

    do
    {
        err = 0.0;
#ifdef INPLACE
        // In-place wavefront (Gauss-Seidel-like) update along anti-diagonals
        // Diagonal index s = i + j, interior ranges from 2 to 2*(N-2)
        for (int s = 2; s <= 2 * (N - 2); ++s) {
            int i_start = s - (N - 2);
            if (i_start < 1) i_start = 1;
            int i_end = s - 1;
            if (i_end > N - 2) i_end = N - 2;

            //#pragma omp parallel for if(i_end - i_start > 512)
            for (int i = i_start; i <= i_end; ++i) {
                int j = s - i;
                double prev = A[i * N + j];
                double newv = 0.25 * (
                    A[(i - 1) * N + j] +
                    A[(i + 1) * N + j] +
                    A[i * N + j - 1] +
                    A[i * N + j + 1]
                );
                A[i * N + j] = newv;
                err = fmax(err, fabs(prev - newv));
            }
        }
#else
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                xtmp[i * N + j] = 0.25*(A[(i - 1) * N + j] + A[(i + 1) * N + j] + A[i * N + j - 1] + A[i * N + j + 1]);
                err = fmax(err, fabs(xtmp[i * N + j] - A[i * N + j]));
            }
        }

        // Copy back only interior to preserve boundary conditions
        for (int i = 1; i < N - 1; i++)
        {
            memcpy(&A[i * N + 1], &xtmp[i * N + 1], (size_t)(N - 2) * sizeof(double));
            // for (int j = 1; j < N - 1; j++)
            // {
            //     A[i * N + j] = xtmp[i * N + j];
            // }
        }
#endif
        iter++;

#ifdef DEBUG
        if (iter == iterations_print)
        {
            printf("  %8d  %f\n", iter, err);
            iterations_print = 2 * iterations_print;
        }
#endif
    } while (err > CONVERGENCE_THRESHOLD && iter < MAX_ITERATIONS);

    return iter;
}

// Output results: compute checksum and optionally dump to file in DEBUG mode
static void output_results(const double *A, int N) {
#ifdef DEBUG
    const char *out_path = "jacobi-result.bin";
    FILE *out = fopen(out_path, "wb");
    if (!out) {
        printf("ERROR: could not open output file '%s'\n", out_path);
    } else {
        size_t to_write = (size_t)N * (size_t)N;
        size_t written = fwrite(A, sizeof(double), to_write, out);
        if (written != to_write) {
            printf("WARNING: wrote only %zu of %zu elements to '%s'\n", written, to_write, out_path);
        }
        fclose(out);
    }
    printf("Output file     = %s\n", out_path);
#endif
    printf(SEPARATOR);
}

// Check results against reference file if available
static void check_reference(const double *A, int N) {
    const char *ref_path = NULL;
    if (N == 500) {
        ref_path = "data/jacobi-500-res.bin";
    } else if (N == 100) {
        ref_path = "data/jacobi-100-res.bin";
    }

    if (!ref_path) return;

    FILE *ref = fopen(ref_path, "rb");
    if (!ref) {
        printf("ERROR: could not open reference file '%s'\n", ref_path);
        return;
    }

    double ref_val;
    double max_diff = 0.0;
    int max_idx = -1;
    size_t total = (size_t)N * (size_t)N;
    size_t read_count = 0;

    for (size_t idx = 0; idx < total; ++idx) {
        if (fread(&ref_val, sizeof(double), 1, ref) != 1) {
            printf("ERROR: unexpected EOF while reading reference file\n");
            break;
        }
        double diff = fabs(ref_val - A[idx]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = (int)idx;
        }
        ++read_count;
    }
    fclose(ref);

    if (read_count != total) {
        printf("WARNING: reference file size mismatch (read %zu of %zu elements)\n", read_count, total);
    }
    printf("Reference file   = %s\n", ref_path);
    printf("Max diff error   = %.4f (idx=%d)\n",         if (diff > max_diff) {
, max_idx);
    printf(SEPARATOR);
}

int main(int argc, char *argv[])
{
    parse_arguments(argc, argv);

    double *A = malloc(N * N * sizeof(double));
    double *xtmp = malloc(N * N * sizeof(double));

    printf(SEPARATOR);
    printf("Matrix size:            %dx%d\n", N, N);
    printf("Maximum iterations:     %d\n", MAX_ITERATIONS);
    printf("Convergence threshold:  %lf\n", CONVERGENCE_THRESHOLD);
    printf(SEPARATOR);

    for (int ii = 0; ii < N; ii++)
    {
        for (int jj = 0; jj < N; jj++)
        {
            double f;
            if ( fread(&f, sizeof(double), 1, data) != 1) {
               printf("ERROR: unexpected EOF while reading reference file\n");
               exit(1);
            }
            A[ii * N + jj] = f;
        }
    }

    // Run Jacobi solver
    start_timer();
    int itr = run(A, xtmp);
    stop_timer();

    printf("Iterations     = %d\n", itr);
    printf("Solver runtime = %lf ms\n", elapsed_ns() / 1E6);
    if (itr == MAX_ITERATIONS)
        printf("WARNING: solution did not converge\n");

    output_results(A, N);
    check_reference(A, N);

    free(A);
    free(xtmp);
    fclose(data);
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
    N = 500;
    MAX_ITERATIONS = 2000;
    CONVERGENCE_THRESHOLD = 0.001;
    SEED = 0;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--convergence") || !strcmp(argv[i], "-c"))
        {
            if (++i >= argc || (CONVERGENCE_THRESHOLD = parse_double(argv[i])) < 0)
            {
                printf("Invalid convergence threshold\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
        {
            if (++i >= argc || (MAX_ITERATIONS = parse_int(argv[i])) < 0)
            {
                printf("Invalid number of iterations\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--norder") || !strcmp(argv[i], "-n"))
        {
            if (++i >= argc || (N = parse_int(argv[i])) < 0)
            {
                printf("Invalid matrix order\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            printf("\n");
            printf("Usage: ./jacobi [OPTIONS]\n\n");
            printf("Options:\n");
            printf("  -h  --help               Print this message\n");
            printf("  -c  --convergence  C     Set convergence threshold\n");
            printf("  -i  --iterations   I     Set maximum number of iterations\n");
            printf("  -n  --norder       N     Set maxtrix order (500 or 1000)\n");
            printf("\n");
            exit(0);
        }
        else
        {
            printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
            exit(1);
        }
    }

    if (N == 1000)
        data = fopen("data/jacobi-1000.bin", "rb");
    else if (N == 500)
        data = fopen("data/jacobi-500.bin", "rb");
    else
    {
        printf("Invalid matrix order\n");
        exit(1);
    }
}