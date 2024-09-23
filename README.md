
# Matrix Multiplication Optimization Project

## Introduction

This project compares multiple algorithms for matrix multiplication, focusing on optimizing performance through various techniques such as SIMD vectorization, tiling (blocking), and multi-threading. Implementations were done in C++ and Assembly, with performance measured by time complexity, operation duration, memory locality, and cache usage.

### Key Algorithms:
- **Basic Matrix Multiplication** (C++ and Assembly)
- **SIMD Vector Matrix Multiplication**
- **Tiled (Blocked) Matrix Multiplication**
- **Threaded Matrix Multiplication**
- **Combined Tiled and Threaded Multiplication**

## Performance Insights

The project demonstrated that memory locality is a critical factor in optimization. While SIMD and threading offered some improvements, the tiled matrix multiplication significantly reduced cache misses, making it the most efficient for larger matrices.

## Full Report

For detailed implementation, performance comparisons, and results, please refer to the [report](report.pdf).

## Getting Started 

Note: the parametere sizes are set within the tests.cpp and timing.cpp and might need to be modified to best suit your needs. 

1. Download the vectorclass library
```bash
git clone --branch v2.02.01 https://github.com/vectorclass/version2 vectorclass
```

2. run the program
   to run the tests use
```bash
g++ -pthread -Ivectorclass -Wall -Wpedantic -std=c++17 -march=haswell -O3 matrixMul.cpp tests.cpp matrixMul.S
```

  to run the timing sequence
```bash
g++ -pthread -Ivectorclass -Wall -Wpedantic -std=c++17 -march=haswell -O3 matrixMul.cpp timing.cpp matrixMul.S
```

---
