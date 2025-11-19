/**
 * @file saxpy.c
 *
 * @brief saxpy performs the \c axpy computation in single-precision on both
 * host and accelerator. The performance (in MFLOPS) on host and accelerator is
 * compared and the numerical results are also verified for consistency.
 *
 * The \c axpy computation is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are vectors each with n elements.
 *
 * Please note that in this version only <em>one GPU thread</em> is used.
 *
 * Offload to GPU:
 *
 * gcc -fopenmp -foffload=nvptx-none saxpy.c
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "utils.h"

#define TWO02 (1 << 2)
#define TWO04 (1 << 4)
#define TWO08 (1 << 8)
#ifndef N
#define N (1 << 26)
#endif

int main(int argc, char *argv[])
{
  int i, n = N,
         iret = 0;
  float a = 101.0f / TWO02,
        b, c,
        *x, *y, *z;
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
  if (NULL == (x = (float *)malloc(sizeof(*x) * n)))
  {
    printf("error: memory allocation for 'x'\n");
    iret = -1;
  }
  if (NULL == (y = (float *)malloc(sizeof(*y) * n)))
  {
    printf("error: memory allocation for 'y'\n");
    iret = -1;
  }
  if (NULL == (z = (float *)malloc(sizeof(*z) * n)))
  {
    printf("error: memory allocation for 'z'\n");
    iret = -1;
  }
  if (0 != iret)
  {
    free(x);
    free(y);
    free(z);
    exit(EXIT_FAILURE);
  }
  b = rand() % TWO04;
  c = rand() % TWO08;
  for (i = 0; i < n; i++)
  {
    x[i] = b / (float)TWO02;
    y[i] = z[i] = c / (float)TWO04;
  }
  /*
   * 1. saxpy on host
   */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  for (i = 0; i < n; i++)
  {
    y[i] = a * x[i] + y[i];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("saxpy on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n / (1.0e6 * wt));
  /*
   * 2. saxpy;
   */
  #pragma omp target data map(to:x[0,n]) map(tofrom:z[0,n])
  #pragma omp target teams num_teams(n/NTHREADS_GPU) thread_limit(NTHREADS_GPU) map(to:x[0,n]) map(tofrom:z[0,n])
  #pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
  for (i = 0; i < n; i++)
  {
    z[i] = a * x[i] + z[i];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("saxpy on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n / (1.0e6 * wt));
  /*
   * 3. verify numerical consistency
   */
  for (i = 0; i < n; i++)
  {
    iret = *(int *)(y + i) ^ *(int *)(z + i);
    assert(iret == 0);
  }
  return 0;
}
             