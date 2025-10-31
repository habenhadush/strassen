#include "matrix_data.h"
#include "matrix_ops.h"
#include <stdlib.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <stdio.h> 
#include <string.h>

inline void 
__ss_madd_fallback(const Matrix A, Matrix B, Matrix C) {
    for (size_t i = 0; i < A.ndim; i++) {
        const double *__restrict a = &A.elements[(A.i_offset + i) * A.stride + A.j_offset];
        const double *__restrict b = &B.elements[(B.i_offset + i) * B.stride + B.j_offset];
        double *__restrict c = &C.elements[(C.i_offset + i) * C.stride + C.j_offset];
#ifdef _OPENMP
#pragma omp simd
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
        for (size_t j = 0; j < A.ndim; j++) {
            c[j] = a[j] + b[j];
        }
    }
}

inline void
__ss_msub_fallback(const Matrix A, const Matrix B, const Matrix C) {
    for (size_t i = 0; i < A.ndim; i++) {
        const double *__restrict a = &A.elements[(A.i_offset + i) * A.stride + A.j_offset];
        const double *__restrict b = &B.elements[(B.i_offset + i) * B.stride + B.j_offset];
        double *__restrict c = &C.elements[(C.i_offset + i) * C.stride + C.j_offset];
#ifdef _OPENMP
#pragma omp simd
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
        for (size_t j = 0; j < A.ndim; j++) {
            c[j] = a[j] - b[j];
        }
    }
}

inline void
__ss_mmul_fallback(const Matrix A, const Matrix B, Matrix C, int num_threads) {
    const int n = A.ndim;
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
#define BLOCK_SIZE_I 64
#define BLOCK_SIZE_J 64
#define BLOCK_SIZE_K 64

#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int ii = 0; ii < n; ii += BLOCK_SIZE_I) {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE_J) {
                const int i_end = (ii + BLOCK_SIZE_I < n) ? ii + BLOCK_SIZE_I : n;
                const int j_end = (jj + BLOCK_SIZE_J < n) ? jj + BLOCK_SIZE_J : n;

                // zering out result elements
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)] = 0.0;
                    }
                }

                // Cache blocking 
                for (int kk = 0; kk < n; kk += BLOCK_SIZE_K) {
                    const int k_end = (kk + BLOCK_SIZE_K < n) ? kk + BLOCK_SIZE_K : n;

                    for (int i = ii; i < i_end; i++) {
                        for (int k = kk; k < k_end; k++) {
                            // prefetch for better cache usage : ref https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
                            __builtin_prefetch(&A.elements[(A.i_offset + i) * A.stride + (A.j_offset + k + 8)], 0, 3);

                            const double a_ik = A.elements[(A.i_offset + i) * A.stride + (A.j_offset + k)];

                            int j = jj;
                            for (; j + 8 <= j_end; j += 8) {
                                // ref : https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
                                __builtin_prefetch(&B.elements[(B.i_offset + k) * B.stride + (B.j_offset + j + 16)], 0, 3);

                                // 8 elements for ILP
                                double *c_ptr = &C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)];
                                const double *b_ptr = &B.elements[(B.i_offset + k) * B.stride + (B.j_offset + j)];

                                c_ptr[0] += a_ik * b_ptr[0];
                                c_ptr[1] += a_ik * b_ptr[1];
                                c_ptr[2] += a_ik * b_ptr[2];
                                c_ptr[3] += a_ik * b_ptr[3];
                                c_ptr[4] += a_ik * b_ptr[4];
                                c_ptr[5] += a_ik * b_ptr[5];
                                c_ptr[6] += a_ik * b_ptr[6];
                                c_ptr[7] += a_ik * b_ptr[7];
                            }

                            // remaining elemetnts
                            for (; j < j_end; j++) {
                                C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)] +=
                                    a_ik * B.elements[(B.i_offset + k) * B.stride + (B.j_offset + j)];
                            }
                        }
                    }
                }
            }
        }
    }
}
