#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "utils.h"

#ifndef N
#define N (1 << 10)
#endif

#pragma omp declare target
#define SM 64

static void reorder2(float *restrict a, float *restrict b, int n)
{
    for (int i = 0; i < SM; i++)
        for (int j = 0; j < SM; j++)
            b[i * SM + j] = a[i * n + j];
}

static void kernel(float *restrict a, float *restrict b, float *restrict c, int n)
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

void gemm_accel(float *restrict a, float *restrict b, float *restrict c, int n)
{
    int bk = n / SM;
    float b2[SM * SM];
    #pragma omp target data map(to:n, bk, a[0,n*n], b[0,n*n]) map(from:c[0,n*n])
    #pragma omp target teams num_teams(bk/NTHREADS_GPU) thread_limit(NTHREADS_GPU) map(to:n, bk, a[0,n*n], b[0,n*n]) map(from: c[0,n*n])
    #pragma omp distribute parallel for num_threads(NTHREADS_GPU) collapse(3) dist_schedule(static, NTHREADS_GPU)
    for (int i = 0; i < bk; i++)
    {
        for (int j = 0; j < bk; j++)
        {
            for (int k = 0; k < bk; k++)
            {
                reorder2(&b[SM * (k * n + j)], b2, n);
                kernel(&a[SM * (i * n + k)], b2, &c[SM * (i * n + j)], n);
            }
        }
    }
}

#pragma omp end declare target

void gemm_opt(float *restrict a, float *restrict b, float *restrict c, int n)
{
    int bk = n / SM;
    {
        float b2[SM * SM];
        for (int i = 0; i < bk; i++)
        {
            for (int j = 0; j < bk; j++)
            {
                for (int k = 0; k < bk; k++)
                {
                    reorder2(&b[SM * (k * n + j)], b2, n);
                    kernel(&a[SM * (i * n + k)], b2, &c[SM * (i * n + j)], n);
                }
            }
        }
    }
}

void gemm(float *restrict a, float *restrict b, float *restrict c, int n)
{
    int i, j, k;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[i + k * n] * b[k + j * n];
            }
            c[i * n + j] += sum;
        }
    }
}

int main(int argc, char *argv[])
{
    int i, n = N,
           iret = 0;
    float *a, *b, *c, *g;
    struct timespec rt[2];
    double wt; // walltime

    if (argc > 1)
        n = atoi(argv[1]);

    /*
   * 0. prepare x, y, and z
   *
   * y := a * x + y (on host)
   * z := a * x + z (on accel)
   */
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

    if (n <= 1024)
    {
        clock_gettime(CLOCK_REALTIME, rt + 0);
        gemm(a, b, c, n);
        clock_gettime(CLOCK_REALTIME, rt + 1);
        wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        printf("gemm on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));
    }

    if (n <= 4096)
    {
        clock_gettime(CLOCK_REALTIME, rt + 0);
        gemm_opt(a, b, c, n);
        clock_gettime(CLOCK_REALTIME, rt + 1);
        wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        printf("gemm_opt on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));
    }

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gemm_accel(a, b, c, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM-opt1 on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));

    for (i = 0; i < n; i++)
    {
        iret = *(int *)(g + i) ^ *(int *)(c + i);
        assert(iret == 0);
    }
    free(a);
    free(b);
    free(c);
    free(g);

    return 0;
}