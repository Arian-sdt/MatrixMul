#ifndef MINI_PROJECT_H
#define MINI_PROJECT_H

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

void assembly_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size);

void assembly_matrix_simd_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size);

#ifdef __cplusplus
}
#endif

float* create_matrix(uint64_t size);

void populate_matrix(float* matrix ,uint64_t size);

void c_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size);

void c_matrix_simd_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size); 

void c_matrix_blocked_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size, uint64_t blockSize);

void threaded_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size, uint64_t num_threads);

void threaded_blocked_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size, uint64_t num_threads, uint64_t block_size); 

#endif


