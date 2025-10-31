#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H
#include "matrix_data.h"

// dispaches
extern void (*matrix_add)(const Matrix, const Matrix, const Matrix);
extern void (*matrix_sub)(const Matrix, const Matrix, const Matrix);
extern void (*standared_matmul)(const Matrix, const Matrix, const Matrix, int num_threads);

// fallbacks
void __ss_madd_fallback(const Matrix A, const Matrix B, const Matrix C);
void __ss_msub_fallback(const Matrix A, const Matrix B, const Matrix C);
void __ss_mmul_fallback(const Matrix A, const Matrix B, Matrix C, int num_threads);

// for  avx / avx2 SIMD support 
void __ss_madd_avx(const Matrix A, const Matrix B, const Matrix C);
void __ss_msub_avx(const Matrix A, const Matrix B, const Matrix C);
void __ss_mmul_avx(const Matrix A, const Matrix B, Matrix C, int num_threads);

// extend it here to support others

// init this before  the above matrix multplication opretions
void init_matrix_ops();
#endif 