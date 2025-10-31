#include "matrix_ops.h"
#include "matrix_data.h"
#include <cpuid.h>
#include <stdio.h>
#include <stdlib.h>
/* Resources : 
 *      ChatGPT : How to make AVX with fallback code portable into different servers ?   
 * My add and sub function were in the mm_funcs.c file, but wasn't compiling in the fries server. 
 * I took the recommendation to condtional compile based on the runtime AVX support.
 */

void (*matrix_add)(const Matrix, const Matrix, const Matrix) = __ss_madd_fallback;
void (*matrix_sub)(const Matrix, const Matrix, const Matrix) = __ss_msub_fallback;
void (*standared_matmul)(const Matrix, const Matrix, const Matrix, int num_threads) = __ss_mmul_fallback;

/*GCC compiler is assumed*/
void init_matrix_ops() {
    if (__builtin_cpu_supports("avx2") || __builtin_cpu_supports("fma")) {
        matrix_add = __ss_madd_avx;
        matrix_sub = __ss_msub_avx;
        standared_matmul = __ss_mmul_avx;
    }
}