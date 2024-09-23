#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "matrixMul.h"


void displayMatrix(float* matrix, uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        for (uint64_t j = 0; j < size; j++) {
            printf(" %.8f ", matrix[i * size + j]);
            
        }
        printf("\n");
    }
}

int main(void) {

    uint64_t size = 512;
    uint64_t block_size = 8; 
    uint64_t num_threads = 8;

    uint64_t length = size * size; 

    float* matrix1 = create_matrix(size);
    float* matrix2 = create_matrix(size);
    float* matrix_out = create_matrix(size);

    assert(matrix1 != NULL);
    assert(matrix2 != NULL);
    assert(matrix_out != NULL);

    populate_matrix(matrix1, size);
    populate_matrix(matrix2, size);

    displayMatrix(matrix1, size);
    printf("\n\n");
    displayMatrix(matrix2, size);
    printf("\n\n");

    c_matrix_mul(matrix1, matrix2, matrix_out, size);
    printf("c_matrix_mul result:\n");
    displayMatrix(matrix_out, size);

    memset(matrix_out, 0,  length * sizeof(float));

    c_matrix_simd_mul(matrix1, matrix2, matrix_out, size);
    printf("c_matrix_simd_mul result:\n");
    displayMatrix(matrix_out, size);

    memset(matrix_out, 0,  length * sizeof(float));
    
    assembly_matrix_mul(matrix1, matrix2, matrix_out, size);
    printf("assembly_matrix_mul result:\n");
    displayMatrix(matrix_out, size);

    memset(matrix_out, 0,  length * sizeof(float));

    assembly_matrix_simd_mul(matrix1, matrix2, matrix_out, size);
    printf("assembly_matrix_simd_mul result:\n");
    displayMatrix(matrix_out, size);

    memset(matrix_out, 0,  length * sizeof(float));

    c_matrix_blocked_mul(matrix1, matrix2, matrix_out, size, block_size);
    printf("c_matrix_blocked_mul result:\n");
    displayMatrix(matrix_out, size);

    memset(matrix_out, 0,  length * sizeof(float));

    threaded_matrix_mul(matrix1, matrix2, matrix_out, size, num_threads);
    printf("threaded_matrix_mul result:\n");
    displayMatrix(matrix_out, size);

    memset(matrix_out, 0,  length * sizeof(float));
    
    threaded_blocked_matrix_mul(matrix1, matrix2, matrix_out, size, num_threads, block_size);
    printf("threaded_blocked_matrix_mul result:\n");
    displayMatrix(matrix_out, size);


    free(matrix1);
    free(matrix2);
    free(matrix_out);

    return 0;
}
