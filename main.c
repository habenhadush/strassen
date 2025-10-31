#include "matrix_data.h"
#include "matrix_ops.h"
#include "memory_pool.h"
#include "mm_funcs.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#if _OPENMP
  #include <omp.h>
#endif
#define ALLIGNMENT 64
#define PRODUCTION 1
#define THRESHOLD 1E-6

static double get_wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// generate random test double type matrix between
// min and max to benchmark with negative and postive values
Matrix static rand_mat(size_t dim, int min, int max) {
    double *data = aligned_alloc(32, dim * dim * sizeof(double));
    if (data == NULL) {
        printf("ERROR: allocating memory for matrix size %ld failed.\n", dim);
        exit(-1);
    }
    for (size_t i = 0; i < dim * dim; i++) {
        data[i] = min + (rand() / (double)RAND_MAX) * (max - min);
    }
    return (Matrix){.elements = data,
                    .i_offset = 0,
                    .j_offset = 0,
                    .ndim = dim,
                    .stride = dim};
}

Matrix static inline alloc_mat(size_t dim) {
    size_t size = dim * dim * sizeof(double);
    double *elements = aligned_alloc(ALLIGNMENT, size);
    if (elements == NULL) {
        fprintf(stderr, "Memory allocation failed in alloc_mat(%zu)\n", dim);
        exit(EXIT_FAILURE);
    }

    memset(elements, 0, size);

    return (Matrix){.elements = elements,
                    .i_offset = 0,
                    .j_offset = 0,
                    .ndim = dim,
                    .stride = dim};
}

typedef struct benchmark_result {
    Matrix C;
    double runtime;
    char *al_name;
    int num_threads;
    int crossover;
} benchmark_result;

int validate(benchmark_result schoolbk_al, benchmark_result other) {
    if (schoolbk_al.C.elements == NULL || other.C.elements == NULL) {
        fprintf(stderr, "Error: NULL matrix in validate()\n");
        return 0;
    }
    const double threshold = THRESHOLD;
    int compare = 1;
    double max_diff = 0.0;
    int ndim = schoolbk_al.C.ndim;

    // from valgrind errors logs
    if (ndim != other.C.ndim) {
        fprintf(stderr, "Error: Matrix dimensions don't match: %d vs %d\n", ndim,
                other.C.ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        for (int j = 0; j < ndim; j++) {
            size_t idx = i * ndim + j;
            if (idx >= (size_t)ndim * ndim) {
                fprintf(stderr, "Index out of bounds: %zu\n", idx);
                continue;
            }

            double element1 = schoolbk_al.C.elements[idx];
            double element3 = other.C.elements[idx];

            /* Valgrind error logs debugged with copilot
             * orginal code didn't have this check...
             */
            if (isnan(element1) || isnan(element3) || isinf(element1) ||
                isinf(element3)) {
                fprintf(stderr, "Invalid value detected at [%d,%d]: %f vs %f\n", i, j,
                        element1, element3);
                max_diff = INFINITY;
                compare = 0;
                continue;
            }

            double d2 = fabs(element1 - element3);

            if (d2 > threshold) {
                max_diff = (d2 > max_diff) ? d2 : max_diff;
                compare = 0;
            }
        }
    }
    if (compare == 0) {
        printf("max_diff=%f is  greater than %f\n", max_diff, THRESHOLD);
        printf("Exiting with error!\n");
    } else {
        printf("seems ok! max_diff = %f\n", max_diff);
    }
    return compare;
}

static inline void print_sample(benchmark_result t, int n) {
    printf("Sample for %s\n", t.al_name);
    printf("-----------------------------------------\n");
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%f\t", t.C.elements[i * n + j]);
        }
        printf("\n");
    }
}

static inline void show_results(benchmark_result r, int ps) {
    printf("%-33s: Runtime = %f seconds.\n", r.al_name, r.runtime);
    if (ps) {
        print_sample(r, 16);
    }
}
typedef struct {
    int dim;
    int threads;
    int crossover;
    double standared_runtime;
    float strassen_runtime;
} benchmark_run;

static void compare_als(benchmark_run result, FILE *file,
                        int add_header) {
#if PRODUCTION == 1
    return;
#endif
    if (add_header) {
        fprintf(file, "Size,Crossover,Num_threads,Standared,Strassen\n");
    }
    fprintf(file, "%d,%d,%d,%f,%f\n", result.dim, result.crossover,
            result.threads, result.standared_runtime, result.strassen_runtime);
}

static benchmark_run benchmark(int dim, int crossover, int num_threads) {
    printf("\033[1;32m******************* DIMENSSION=%dX%d, CROSSOVER=%d, THREADS=%d "
           "*******************\033[0m\n",
           dim, dim, crossover, num_threads);
    benchmark_run benchmarks = {
        .crossover = crossover, .dim = dim, .threads = num_threads};

    Matrix A = rand_mat(dim, -10, 10);
    Matrix B = rand_mat(dim, -10, 10);
    Matrix C0 = alloc_mat(dim);
    Matrix C1 = alloc_mat(dim);

    double start_time, end_time;

    init_matrix_ops();
    benchmark_result schookbk_al;
    start_time = get_wall_time();
    standared_matmul(A, B, C0, num_threads);

    end_time = get_wall_time();
    schookbk_al = (benchmark_result){.C = C0,
                                     .al_name = "Standard_Matrix_Multiplication",
                                     .runtime = end_time - start_time,
                                     .num_threads = num_threads,
                                     .crossover = 0};
    show_results(schookbk_al, 0);
    benchmarks.standared_runtime = schookbk_al.runtime;

    start_time = get_wall_time();
    strassen_matmul(A, B, C1, crossover, num_threads);
    end_time = get_wall_time();
    benchmark_result strassen_al =
        (benchmark_result){.C = C1,
                           .al_name = "Strassen's_Matrix_Multiplication",
                           .runtime = end_time - start_time,
                           .num_threads = num_threads,
                           .crossover = crossover};
    show_results(strassen_al, 0);
    validate(schookbk_al, strassen_al);
    benchmarks.strassen_runtime = strassen_al.runtime;

    free(A.elements);
    free(B.elements);
    free(C0.elements);
    free(C1.elements);
    printf("\n");

    return benchmarks;
}

int main(int argc, char *argv[]) {
    FILE *file;
    int add_header = 1;

#if PRODUCTION == 1

    if (argc != 4) {
        printf("\033[1;31mMissing 3 arguments\033[0m\n");
        printf("\tUsage: [ %s <matrix_dim> <crossover> <num_threads>]  \n", argv[0]);
        return -1;
    }
    int mat_dim = atoi(argv[1]);
    int crossover = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    file = stdout;
#else
    if (argc != 5) {
        printf("\033[1;31mMissing 4 arguments\033[0m\n");
        printf("Usage: %s <matrix_dim> <crossover> <num_threads> <out_filename>\n", argv[0]);
        return -1;
    }
    int mat_dim = atoi(argv[1]);
    int crossover = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    file = fopen(argv[4], "r");
    if (file) {
        add_header = 0;
        fclose(file);
    }
    file = fopen(argv[4], "a");
    if (file == NULL) {
        printf("ERROR: Opening file!\n");
        return -1;
    }
#endif
    benchmark_run res = benchmark(mat_dim, crossover, num_threads);
    compare_als(res, file, add_header);
    fclose(file);
}