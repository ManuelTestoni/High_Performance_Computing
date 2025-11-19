#if !defined(W)
#define W (1 << 15)
#endif

/* Dummy Tasks */
void task1();
void task2();
void task3();
void task4();


/**
 * @brief EX 5 - Task Parallelism w/sections
 * 
 * a) Create a parallel region with 4 threads. Use thread IDs to execute
 *    different WORK functions on different threads.
 * b) Create a parallel region with 4 threads. Achieve the same work partitioning
 *    as a) using SECTIONS.
 * 
 * @return void
 */
void exercise()
{
    #pragma omp parallel sections       
    {
        #pragma omp section
                task1();
        #pragma omp section
                task2();
        #pragma omp section
                task3();
        #pragma omp section
                task4();
    }
}

void task1()
{
    printf("Hi sir, I'm Thread %d: ", omp_get_thread_num());
    DEBUG_PRINT("%hu: exec task1!\n", omp_get_thread_num());
    work((1 * W));
}

void task2()
{
    printf("Hi sir, I'm Thread %d: ", omp_get_thread_num());
    DEBUG_PRINT("%hu: exec task2!\n", omp_get_thread_num());
    work((2 * W));
}

void task3()
{
    printf("Hi sir, I'm Thread %d: ", omp_get_thread_num());
    DEBUG_PRINT("%hu: exec task3!\n", omp_get_thread_num());
    work((3 * W));
}

void task4()
{
    printf("Hi sir, I'm Thread %d: ", omp_get_thread_num());
    DEBUG_PRINT("%hu: exec task4!\n", omp_get_thread_num());
    work((4 * W));
}
                                                                                                                                                102,1         Bot
