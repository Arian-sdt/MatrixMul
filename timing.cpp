#define _POSIX_C_SOURCE 199309L
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "matrixMul.h"

#define MATRIX_SIZE 8192
#define WARM_UP_SIZE 256
#define BLOCK_SIZE 16
#define NUM_THREADS 2

#define TIMING_RESULT(DESCR, CODE) do { \
    struct timespec start, end; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); \
    CODE; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end); \
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; \
    printf("%25s took %7.1f ms\n", descr, elapsed * 1000); \
} while(0)


void matrix_speed(const char* descr, void mat_mul(float*, float*, float*, uint64_t), uint64_t size) {
    float* m1 = create_matrix(size);
    assert(m1 != NULL);
    float* m2 = create_matrix(size);
    assert(m2 != NULL);
    float* m_result = create_matrix(size);
    assert(m_result != NULL);

    populate_matrix(m1, size);
    populate_matrix(m2, size);

    TIMING_RESULT(descr, mat_mul(m1, m2, m_result, size));

    free(m1);
    free(m2);
    free(m_result);
}

void matrix_blocked_speed(const char* descr, void mat_mul(float*, float*, float*, uint64_t, uint64_t), uint64_t size, uint64_t blockSize) {
    float* m1 = create_matrix(size);
    assert(m1 != NULL);
    float* m2 = create_matrix(size);
    assert(m2 != NULL);
    float* m_result = create_matrix(size);
    assert(m_result != NULL);

    populate_matrix(m1, size);
    populate_matrix(m2, size);

    TIMING_RESULT(descr, mat_mul(m1, m2, m_result, size, blockSize));

    free(m1);
    free(m2);
    free(m_result);
}

void matrix_threaded_speed(const char* descr, void mat_mul(float*, float*, float*, uint64_t, uint64_t), uint64_t size, uint64_t numThreads) {
    float* m1 = create_matrix(size);
    assert(m1 != NULL);
    float* m2 = create_matrix(size);
    assert(m2 != NULL);
    float* m_result = create_matrix(size);
    assert(m_result != NULL);

    populate_matrix(m1, size);
    populate_matrix(m2, size);

    TIMING_RESULT(descr, mat_mul(m1, m2, m_result, size, numThreads));

    free(m1);
    free(m2);
    free(m_result);
}

void matrix_threaded_blocked_speed(const char* descr, void mat_mul(float*, float*, float*, uint64_t, uint64_t, uint64_t), uint64_t size, uint64_t numThreads, uint64_t blockSize) {
    float* m1 = create_matrix(size);
    assert(m1 != NULL);
    float* m2 = create_matrix(size);
    assert(m2 != NULL);
    float* m_result = create_matrix(size);
    assert(m_result != NULL);

    populate_matrix(m1, size);
    populate_matrix(m2, size);

    TIMING_RESULT(descr, mat_mul(m1, m2, m_result, size, numThreads, blockSize));

    free(m1);
    free(m2);
    free(m_result);
}

int main(void) {

    matrix_speed("warmup", c_matrix_mul, WARM_UP_SIZE);

    matrix_speed("c_matrix_mul", c_matrix_mul, MATRIX_SIZE);
    matrix_speed("c_matrix_simd_mul", c_matrix_simd_mul, MATRIX_SIZE);
    matrix_speed("assembly_matrix_mul", assembly_matrix_mul, MATRIX_SIZE);
    matrix_speed("assembly_matrix_simd_mul", assembly_matrix_simd_mul, MATRIX_SIZE);
    matrix_blocked_speed("c_matrix_blocked_mul", c_matrix_blocked_mul, MATRIX_SIZE, BLOCK_SIZE);
    matrix_threaded_speed("threaded_matrix_mul", threaded_matrix_mul, MATRIX_SIZE, NUM_THREADS);
    matrix_threaded_blocked_speed("threaded_blocked_matrix_mul", threaded_blocked_matrix_mul, MATRIX_SIZE, NUM_THREADS, BLOCK_SIZE);

    return 0;
}