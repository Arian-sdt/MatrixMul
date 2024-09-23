#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <thread>
#include <future>
#include <algorithm>
#include <memory>
#include <vector>
#include "vectorclass.h"
#include "matrixMul.h"

struct ThreadData {
    float* firstMatrix;
    float* secondMatrix;
    float* result;
    uint64_t size;
    uint64_t start_row;
    uint64_t end_row;
    uint64_t block_size;
};


float* create_matrix(uint64_t size){

    uint64_t length = size * size;

    float* matrix = (float*)malloc((length) * sizeof(float));
    if (matrix == NULL) {
        return NULL;
    }

    for (uint64_t i = 0; i < length; i++){
        matrix[i] = 0; 
    }

    return matrix;
}

void populate_matrix(float* matrix ,uint64_t size){

    uint64_t length = size * size;

    for (uint64_t i = 0; i < (length); i++) {
        matrix[i] = ((float)rand() / RAND_MAX - 0.25);
    }

}

void c_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size){

    for (uint64_t i = 0; i < size; i++) {
        for (uint64_t j = 0; j < size; j++) {
            for (uint64_t k = 0; k < size; k++) {
                result[i * size + j] += firstMatrix[i * size + k] * secondMatrix[k * size + j];
            }
        }
    }

}

void c_matrix_simd_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size){
    
    Vec8f vecB, vecA;
    float temparr[8];

    for (uint64_t i = 0; i < size; i++) {
        for (uint64_t j = 0; j < size; j++) {
            Vec8f sum = Vec8f(0.0);

            for (uint64_t k = 0; k < size; k += 8) {
                vecA.load(firstMatrix + i * size + k);

                for (int x = 0; x < 8; x++) {
                    temparr[x] = secondMatrix[(k + x) * size + j];
                }

                vecB.load(temparr);
                sum += vecA * vecB;
            }

            result[i * size + j] = horizontal_add(sum);
        }
    }
    
}

void c_matrix_blocked_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size, uint64_t blockSize) {

    for (uint64_t ii = 0; ii < size; ii += blockSize) {
        for (uint64_t jj = 0; jj < size; jj += blockSize) {
            for (uint64_t kk = 0; kk < size; kk += blockSize) {
                for (uint64_t i = ii; i < ii + blockSize && i < size; ++i) {
                    for (uint64_t j = jj; j < jj + blockSize && j < size; ++j) {
                        float sum = result[i * size + j];
                        for (uint64_t k = kk; k < kk + blockSize && k < size; ++k) {
                            sum += firstMatrix[i * size + k] * secondMatrix[k * size + j];
                        }
                        result[i * size + j] = sum;
                    }
                }
            }
        }
    }
}



void* matrix_mul_worker(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    for (uint64_t i = data->start_row; i < data->end_row; ++i) {
        for (uint64_t j = 0; j < data->size; ++j) {
            float sum = 0;
            for (uint64_t k = 0; k < data->size; ++k) {
                sum += data->firstMatrix[i * data->size + k] * data->secondMatrix[k * data->size + j];
            }
            data->result[i * data->size + j] = sum;
        }
    }

    return nullptr;
}

void threaded_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size, uint64_t num_threads) {
    std::vector<std::future<void*>> futures;
    uint64_t rows_per_thread = size / num_threads;
    uint64_t start_row = 0;

    for (uint64_t i = 0; i < num_threads; ++i) {
        uint64_t end_row = start_row + rows_per_thread;
        if (i == num_threads - 1) {  // Last thread may need to handle the remaining rows
            end_row = size;
        }

        auto threadData = std::make_shared<ThreadData>(ThreadData{firstMatrix, secondMatrix, result, size, start_row, end_row, 0});

        futures.push_back(std::async(std::launch::async, [threadData]() -> void* {
            return matrix_mul_worker(threadData.get());
        }));
        start_row = end_row;
    }

    for (auto& future : futures) {
        future.get();
    }
}

void* matrix_mul_worker2(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    float* firstMatrix = data->firstMatrix;
    float* secondMatrix = data->secondMatrix;
    float* result = data->result;
    uint64_t size = data->size;
    uint64_t start_row = data->start_row;
    uint64_t end_row = data->end_row;
    uint64_t block_size = data->block_size;

    for (uint64_t block_i = start_row; block_i < end_row; block_i += block_size) {
        for (uint64_t block_j = 0; block_j < size; block_j += block_size) {
            for (uint64_t block_k = 0; block_k < size; block_k += block_size) {
                for (uint64_t i = block_i; i < std::min(block_i + block_size, end_row); ++i) {
                    for (uint64_t j = block_j; j < std::min(block_j + block_size, size); ++j) {
                        float sum = 0;
                        for (uint64_t k = block_k; k < std::min(block_k + block_size, size); ++k) {
                            sum += firstMatrix[i * size + k] * secondMatrix[k * size + j];
                        }
                        result[i * size + j] += sum;
                    }
                }
            }
        }
    }

    return nullptr;
}

void threaded_blocked_matrix_mul(float* firstMatrix, float* secondMatrix, float* result, uint64_t size, uint64_t num_threads, uint64_t block_size) {
    std::vector<std::future<void*>> futures;
    uint64_t chunk_size = (size + num_threads - 1) / num_threads;
    uint64_t start_row = 0;

    for (uint64_t i = 0; i < num_threads; ++i) {
        uint64_t end_row = start_row + chunk_size;
        if (end_row > size) {
            end_row = size;
        }

        auto threadData = std::make_shared<ThreadData>(ThreadData{firstMatrix, secondMatrix, result, size, start_row, end_row, block_size});

        futures.push_back(std::async(std::launch::async, [threadData]() -> void* {
            return matrix_mul_worker2(threadData.get());
        }));
        start_row = end_row;
    }

    for (auto& future : futures) {
        future.get();
    }
}