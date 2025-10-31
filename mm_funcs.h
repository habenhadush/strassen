#ifndef MM_FUNCS_H
#define MM_FUNCS_H
#include "matrix_data.h"
#include "memory_pool.h"
#include <stdlib.h>

void strassen_matmul(const Matrix A, const Matrix B, const Matrix C, int crossover, int num_threads);
#endif