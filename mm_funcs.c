/*
REFERENCE AND DISCAIMERS:

REFERENCES:
The algorithms used in  strassen's algorithm are based on the psedocode
provide in the
  Cormen, T. H. (2009) Introduction to algorithms. 3. ed. Cambridge, Mass: MIT Press.
url : https://uub.primo.exlibrisgroup.com/discovery/fulldisplay?docid=alma991001720429707596&context=L&vid=46LIBRIS_UUB:UUB&lang=en&search_scope=MyInst_and_CI&adaptor=Local%20Search%20Engine&tab=Everything&query=any,contains,INTRODUCTION%20TO%20ALGORITHMS&offset=0

Chapter 4 and Chapter 22.

DISCALIMER:
The matrices should be in 2**n form, and padding can be added to extend to any matrices(as a extenssion to this project).
*/

#include "mm_funcs.h"
#include "matrix_ops.h"
#include "memory_pool.h"
#include <assert.h>
#include <immintrin.h>
#if _OPENMP
    #include <omp.h>
#endif
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BLOCK_SIZE 64
#define PDIM 32

static inline Matrix
create_mat_view(Matrix top, int i, int j, int dimenssion) {
    // motivation taken from pytorch / numpy slices of creating view rather expenssive copy
    // create a submatrix without copying / i.e view
    // RMO with stride :  reference Computer Systems A Programmer’s Perspective
    assert(top.i_offset + i + dimenssion <= top.stride);
    assert(top.j_offset + j + dimenssion <= top.stride);
    Matrix view = {
        .elements = top.elements,
        .i_offset = top.i_offset + i,
        .j_offset = top.j_offset + j,
        .ndim = dimenssion,
        .stride = top.stride};
    return view;
}

Matrix allocate_temp_matrix(int size, MemoryPool *pool) {
    Matrix mat;
    mat.elements = memory_pool_allocate(pool, size);
    mat.i_offset = mat.j_offset = 0;
    mat.ndim = size;
    mat.stride = size;
    memset(mat.elements, 0, size * size * sizeof(double));
    return mat;
}

void deallocate_temp_matrix(Matrix mat, MemoryPool *pool) {
    memory_pool_deallocate(pool, mat.elements, mat.ndim);
}

static _Thread_local MemoryPool *thread_pool = NULL;
void strassen_matmul_helper(const Matrix A, const Matrix B, const Matrix C,
                            int crossover, int num_threads) {
    if (A.ndim <= crossover) {
         standared_matmul(A, B, C, 1);
        return;
    }

    int pdim = A.ndim / 2;

    // Create submatrix views
    Matrix A11 = create_mat_view(A, 0, 0, pdim);
    Matrix A12 = create_mat_view(A, 0, pdim, pdim);
    Matrix A21 = create_mat_view(A, pdim, 0, pdim);
    Matrix A22 = create_mat_view(A, pdim, pdim, pdim);

    Matrix B11 = create_mat_view(B, 0, 0, pdim);
    Matrix B12 = create_mat_view(B, 0, pdim, pdim);
    Matrix B21 = create_mat_view(B, pdim, 0, pdim);
    Matrix B22 = create_mat_view(B, pdim, pdim, pdim);

    Matrix C11 = create_mat_view(C, 0, 0, pdim);
    Matrix C12 = create_mat_view(C, 0, pdim, pdim);
    Matrix C21 = create_mat_view(C, pdim, 0, pdim);
    Matrix C22 = create_mat_view(C, pdim, pdim, pdim);

    Matrix S[10], M[7];

    // Only create parallel tasks when there are enough threads and matrix is large
    int use_tasks = (pdim > PDIM) && (num_threads > 1);

    // Only create tasks when necessary, avoid task creation overhead
    if (use_tasks) {
        for (int i = 0; i < 7; i++) {
            M[i] = allocate_temp_matrix(pdim, thread_pool);
        }
        for (int i = 0; i < 10; i++) {
            S[i] = allocate_temp_matrix(pdim, thread_pool);
        }

#pragma omp taskgroup
        {
            // M1 = (A11 + A22) × (B11 + B22)
#pragma omp task final(pdim <= PDIM)  untied 
            {
                matrix_add(A11, A22, S[0]);
                matrix_add(B11, B22, S[1]);
                strassen_matmul_helper(S[0], S[1], M[0], crossover, num_threads);
            }

            // M2 = (A21 + A22) × B11
#pragma omp task final(pdim <= PDIM) untied
            {
                matrix_add(A21, A22, S[2]);
                strassen_matmul_helper(S[2], B11, M[1], crossover, num_threads);
            }

            // M3 = A11 × (B12 - B22)
#pragma omp task final(pdim <= PDIM) untied
            {
                matrix_sub(B12, B22, S[3]);
                strassen_matmul_helper(A11, S[3], M[2], crossover, num_threads);
            }

            // M4 = A22 × (B21 - B11)
#pragma omp task final(pdim <= PDIM) untied
            {
                matrix_sub(B21, B11, S[4]);
                strassen_matmul_helper(A22, S[4], M[3], crossover, num_threads);
            }

            // M5 = (A11 + A12) × B22
#pragma omp task final(pdim <= PDIM) untied
            {
                matrix_add(A11, A12, S[5]);
                strassen_matmul_helper(S[5], B22, M[4], crossover, num_threads);
            }

            // M6 = (A21 - A11) × (B11 + B12)
#pragma omp task final(pdim <= PDIM) untied
            {
                matrix_sub(A21, A11, S[6]);
                matrix_add(B11, B12, S[7]);
                strassen_matmul_helper(S[6], S[7], M[5], crossover, num_threads);
            }

            // M7 = (A12 - A22) × (B21 + B22)
#pragma omp task final(pdim <= PDIM) untied
            {
                matrix_sub(A12, A22, S[8]);
                matrix_add(B21, B22, S[9]);
                strassen_matmul_helper(S[8], S[9], M[6], crossover, num_threads);
            }
        }

#pragma omp taskgroup
        {
#pragma omp task untied
            {
                matrix_add(M[0], M[3], S[0]);
                matrix_sub(S[0], M[4], S[0]);
                matrix_add(S[0], M[6], C11);
            }

#pragma omp task untied
            {
                matrix_add(M[2], M[4], C12);
            }

#pragma omp task untied
            {
                matrix_add(M[1], M[3], C21);
            }

#pragma omp task untied
            {
                matrix_sub(M[0], M[1], S[1]);
                matrix_add(S[1], M[2], S[1]);
                matrix_add(S[1], M[5], C22);
            }
        }

        // Deallocate temporaries
        for (int i = 0; i < 7; i++) {
            deallocate_temp_matrix(M[i], thread_pool);
        }
        for (int i = 0; i < 10; i++) {
            deallocate_temp_matrix(S[i], thread_pool);
        }
    } else {
        // task managemnt and creation become too expenssive
        Matrix S0 = allocate_temp_matrix(pdim, thread_pool);
        Matrix S1 = allocate_temp_matrix(pdim, thread_pool);
        Matrix M0 = allocate_temp_matrix(pdim, thread_pool);

        // M1 = (A11 + A22) × (B11 + B22)
        matrix_add(A11, A22, S0);
        matrix_add(B11, B22, S1);
        strassen_matmul_helper(S0, S1, M0, crossover, 1);

        // M4 = A22 × (B21 - B11)
        matrix_sub(B21, B11, S0);
        Matrix M3 = allocate_temp_matrix(pdim, thread_pool);
        strassen_matmul_helper(A22, S0, M3, crossover, 1);

        // M5 = (A11 + A12) × B22
        matrix_add(A11, A12, S0);
        Matrix M4 = allocate_temp_matrix(pdim, thread_pool);
        strassen_matmul_helper(S0, B22, M4, crossover, 1);

        // M7 = (A12 - A22) × (B21 + B22)
        matrix_sub(A12, A22, S0);
        matrix_add(B21, B22, S1);
        Matrix M6 = allocate_temp_matrix(pdim, thread_pool);
        strassen_matmul_helper(S0, S1, M6, crossover, 1);

        // Calculate C11 = M1 + M4 - M5 + M7
        matrix_add(M0, M3, S0);
        matrix_sub(S0, M4, S0);
        matrix_add(S0, M6, C11);

        deallocate_temp_matrix(M6, thread_pool);

        // M2 = (A21 + A22) × B11
        matrix_add(A21, A22, S0);
        Matrix M1 = allocate_temp_matrix(pdim, thread_pool);
        strassen_matmul_helper(S0, B11, M1, crossover, 1);

        // C21 = M2 + M4
        matrix_add(M1, M3, C21);

        // M3 = A11 × (B12 - B22)
        matrix_sub(B12, B22, S0);
        Matrix M2 = allocate_temp_matrix(pdim, thread_pool);
        strassen_matmul_helper(A11, S0, M2, crossover, 1);

        // C12 = M3 + M5
        matrix_add(M2, M4, C12);

        deallocate_temp_matrix(M4, thread_pool);

        // M6 = (A21 - A11) × (B11 + B12)
        matrix_sub(A21, A11, S0);
        matrix_add(B11, B12, S1);
        Matrix M5 = allocate_temp_matrix(pdim, thread_pool);
        strassen_matmul_helper(S0, S1, M5, crossover, 1);

        // C22 = M1 - M2 + M3 + M6
        matrix_sub(M0, M1, S0);
        matrix_add(S0, M2, S0);
        matrix_add(S0, M5, C22);

        deallocate_temp_matrix(M0, thread_pool);
        deallocate_temp_matrix(M1, thread_pool);
        deallocate_temp_matrix(M2, thread_pool);
        deallocate_temp_matrix(M3, thread_pool);
        deallocate_temp_matrix(M5, thread_pool);
        deallocate_temp_matrix(S0, thread_pool);
        deallocate_temp_matrix(S1, thread_pool);
    }
}

void strassen_matmul(const Matrix A, const Matrix B, const Matrix C,
                     int crossover, int num_threads) {
    init_matrix_ops();
#ifdef _OPENMP
    omp_set_max_active_levels(8);
    omp_set_nested(1);
    omp_set_dynamic(1);
    num_threads = num_threads > omp_get_max_threads() ? omp_get_max_threads() : num_threads;
#endif

    MemoryPool **thread_pools = (MemoryPool **)malloc(num_threads * sizeof(MemoryPool *));
    for (int i = 0; i < num_threads; i++) {
        thread_pools[i] = memory_pool_create();
    }

#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        thread_pool = thread_pools[thread_id];

#pragma omp single nowait
        {
            strassen_matmul_helper(A, B, C, crossover, num_threads);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        memory_pool_destroy(thread_pools[i]);
    }
    free(thread_pools);
}
