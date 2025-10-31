#include "matrix_data.h"
#include "matrix_ops.h"
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <string.h>

/*reference :
 *[1] https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL&text=__m256&ig_expand=6612
 *[2] https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0?permalink_comment_id=2602532
 */
inline void
__ss_madd_avx(const Matrix A, const Matrix B, const Matrix C) {
    for (size_t i = 0; i < A.ndim; i++) {
        _mm_prefetch(&A.elements[(A.i_offset + i + 1) * A.stride + A.j_offset], _MM_HINT_T0);
        _mm_prefetch(&B.elements[(B.i_offset + i + 1) * B.stride + B.j_offset], _MM_HINT_T0);
        size_t j = 0;
        // Process in 4-element chunks with 32-byte alignment
        for (; j + 3 < A.ndim; j += 4) {
            __m256d a = _mm256_load_pd(&A.elements[(A.i_offset + i) * A.stride + (A.j_offset + j)]); // Aligned load[1]
            __m256d b = _mm256_load_pd(&B.elements[(B.i_offset + i) * B.stride + (B.j_offset + j)]);
            __m256d c = _mm256_add_pd(a, b);
            _mm256_store_pd(&C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)], c); // Aligned store[2]
        }
        // Remainder elements
        for (; j < A.ndim; j++) {
            double a = A.elements[(A.i_offset + i) * A.stride + (A.j_offset + j)];
            double b = B.elements[(B.i_offset + i) * B.stride + (B.j_offset + j)];
            C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)] = a + b;
        }
    }
}

inline void
__ss_msub_avx(const Matrix A, const Matrix B, const Matrix C) {
    for (size_t i = 0; i < A.ndim; i++) {
        _mm_prefetch(&A.elements[(A.i_offset + i + 1) * A.stride + A.j_offset], _MM_HINT_T0);
        _mm_prefetch(&B.elements[(B.i_offset + i + 1) * B.stride + B.j_offset], _MM_HINT_T0);
        size_t j = 0;
        for (; j + 3 < A.ndim; j += 4) {
            __m256d a = _mm256_load_pd(&A.elements[(A.i_offset + i) * A.stride + (A.j_offset + j)]);
            __m256d b = _mm256_load_pd(&B.elements[(B.i_offset + i) * B.stride + (B.j_offset + j)]);
            __m256d c = _mm256_sub_pd(a, b);
            _mm256_store_pd(&C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)], c);
        }
        for (; j < A.ndim; j++) {
            double a = A.elements[(A.i_offset + i) * A.stride + (A.j_offset + j)];
            double b = B.elements[(B.i_offset + i) * B.stride + (B.j_offset + j)];
            C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)] = a - b;
        }
    }
}

inline void
__ss_mmul_avx(const Matrix A, const Matrix B, Matrix C, int num_threads) {
    const int n = A.ndim;
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
// L1 cache size
#define BLOCK_SIZE_I 64
#define BLOCK_SIZE_J 64
#define BLOCK_SIZE_K 64

#pragma omp parallel
    {
        // aligned with L1 cacheline
        double *restrict buffer = (double *)aligned_alloc(64, BLOCK_SIZE_I * BLOCK_SIZE_J * sizeof(double));
        memset(buffer, 0, BLOCK_SIZE_I * BLOCK_SIZE_J * sizeof(double));

        if (buffer == NULL) {
            fprintf(stderr, "ERROR: Failed to allocate aligned buffer\n");
            exit(EXIT_FAILURE);
        }

#pragma omp for schedule(guided)
        for (int ii = 0; ii < n; ii += BLOCK_SIZE_I) {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE_J) {
                const int i_end = (ii + BLOCK_SIZE_I < n) ? ii + BLOCK_SIZE_I : n;
                const int j_end = (jj + BLOCK_SIZE_J < n) ? jj + BLOCK_SIZE_J : n;

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)] = 0.0;
                    }
                }

                // Blocking
                for (int kk = 0; kk < n; kk += BLOCK_SIZE_K) {
                    const int k_end = (kk + BLOCK_SIZE_K < n) ? kk + BLOCK_SIZE_K : n;
                    // loop unrolling
                    for (int i = ii; i < i_end; i++) {
                        for (int k = kk; k < k_end; k++) {
                            // Prefetch cacheline
                            // ref : https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
                            __builtin_prefetch(&A.elements[(A.i_offset + i) * A.stride + (A.j_offset + k + 8)], 0, 3);//[1][2]

                            const double a_ik = A.elements[(A.i_offset + i) * A.stride + (A.j_offset + k)];

                            // Vectorized inner loop with AVX2 and proper alignment
                            int j = jj;
                            for (; j + 4 <= j_end; j += 4) {
                                // Prefetch B elements
                                __builtin_prefetch(&B.elements[(B.i_offset + k) * B.stride + (B.j_offset + j + 16)], 0, 3);

                                __m256d b_kj = _mm256_loadu_pd(&B.elements[(B.i_offset + k) * B.stride + (B.j_offset + j)]);
                                __m256d c_ij = _mm256_loadu_pd(&C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)]);

                                // reference : https://www.eiken.dev/blog/2020/04/optimizing-the-walsh-hadamard-transform-using-simd-intrinsics/
                                c_ij = _mm256_fmadd_pd(_mm256_set1_pd(a_ik), b_kj, c_ij);

                                _mm256_storeu_pd(&C.elements[(C.i_offset + i) * C.stride + (C.j_offset + j)], c_ij);
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
        free(buffer);
    }
}
