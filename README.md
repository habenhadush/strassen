# Strassen Matrix Multiplication

A high-performance implementation of Strassen's matrix multiplication algorithm with SIMD optimizations, OpenMP parallelization, and custom memory pooling for efficient computation of large matrices.

## Overview

This project implements both standard and Strassen's divide-and-conquer matrix multiplication algorithms with the following optimizations:
- **SIMD Vectorization**: AVX/AVX2 instructions for accelerated operations
- **Parallel Processing**: OpenMP task-based parallelization
- **Memory Efficiency**: Custom memory pool for reduced allocation overhead
- **Platform Adaptability**: Automatic fallback for systems without AVX support

## Features

- ✅ Standard matrix multiplication with OpenMP parallelization
- ✅ Strassen's algorithm with configurable crossover threshold
- ✅ AVX2/FMA SIMD optimizations for matrix operations (add, subtract, multiply)
- ✅ Automatic CPU feature detection and fallback to non-SIMD code
- ✅ Thread-local memory pooling for temporary matrices
- ✅ Matrix view abstraction (zero-copy submatrix operations)
- ✅ Comprehensive validation and benchmarking framework
- ✅ Production and development modes

## Algorithm

The implementation is based on Strassen's algorithm from:
> Cormen, T. H. (2009). *Introduction to algorithms*. 3rd ed. Cambridge, Mass: MIT Press. Chapter 4.

Strassen's algorithm reduces the complexity of matrix multiplication from O(n³) to approximately O(n^2.807) by using 7 recursive multiplications instead of 8.

## Requirements

- **Compiler**: GCC with C11 support
- **Libraries**: 
  - OpenMP (for parallelization)
  - libm (math library)
- **CPU Features** (optional): AVX2 and FMA for SIMD optimizations
- **OS**: Linux (uses `clock_gettime` for timing)

## Build

Simply run:
```bash
make
```

This will compile all source files and produce the `strassen` executable.

To clean build artifacts:
```bash
make clean
```

### Build Flags

The makefile uses aggressive optimizations:
- `-O3`: High-level optimizations
- `-ffast-math`: Fast floating-point math
- `-march=native`: Optimize for the host CPU architecture
- `-ftree-vectorize`: Enable auto-vectorization
- `-fopenmp`: Enable OpenMP support
- `-flto`: Link-time optimization
- `-mavx2 -mfma`: AVX2 and FMA instructions (for `matrix_ops_avx.c`)

## Usage

### Production Mode (Default)

```bash
./strassen <matrix_dim> <crossover> <num_threads>
```

**Parameters:**
- `matrix_dim`: Matrix dimension (should be a power of 2, e.g., 64, 128, 256, 512, 1024)
- `crossover`: Threshold for switching from Strassen to standard multiplication
- `num_threads`: Number of OpenMP threads to use

**Example:**
```bash
./strassen 1024 64 4
```

This runs both standard and Strassen multiplication on 1024×1024 matrices, switching to standard multiplication at 64×64 submatrices, using 4 threads.

### Development Mode

To enable CSV output for benchmarking, set `PRODUCTION` to `0` in `main.c`:
```c
#define PRODUCTION 0
```

Then rebuild and run:
```bash
make clean && make
./strassen <matrix_dim> <crossover> <num_threads> <output_file.csv>
```

This appends benchmark results to the CSV file for analysis.

## Architecture

### Core Components

#### 1. **Matrix Data Structure** (`matrix_data.h`)
```c
typedef struct {
    double *restrict elements;  // Pointer to matrix data
    int i_offset;               // Row offset for submatrix views
    int j_offset;               // Column offset for submatrix views
    int ndim;                   // Current submatrix dimension
    int stride;                 // Stride of parent matrix
} Matrix;
```

This design enables zero-copy submatrix views, avoiding expensive data copying during recursion.

#### 2. **Matrix Operations** (`matrix_ops.h/c`)
Provides function pointers that dispatch to the best available implementation:
- `matrix_add`: Element-wise addition
- `matrix_sub`: Element-wise subtraction
- `standared_matmul`: Standard O(n³) matrix multiplication

The `init_matrix_ops()` function detects CPU capabilities and sets function pointers to either:
- AVX-optimized implementations (`matrix_ops_avx.c`)
- Fallback implementations (`matrix_ops_fallback.c`)

#### 3. **Strassen Algorithm** (`mm_funcs.h/c`)
Implements the recursive Strassen multiplication with:
- Automatic task-based parallelization for large submatrices
- Configurable crossover threshold to standard multiplication
- Thread-local memory pool allocation for temporaries

#### 4. **Memory Pool** (`memory_pool.h/c`)
Custom memory allocator optimized for matrix operations:
- Pre-allocated blocks organized by size levels
- Thread-local pools to avoid contention
- Reduces overhead compared to repeated `malloc`/`free` calls
- Inspired by Hoard allocator and game engine architecture

### Key Optimizations

1. **SIMD Vectorization**: AVX2 operations process 4 doubles simultaneously
2. **Task Parallelism**: OpenMP tasks enable parallel computation of the 7 Strassen products
3. **Memory Pool**: Eliminates allocation overhead in recursive calls
4. **Matrix Views**: Submatrices are views into parent data (no copying)
5. **Hybrid Approach**: Switches to standard multiplication below crossover threshold

## Performance Tuning

### Crossover Threshold
The crossover parameter controls when to switch from Strassen to standard multiplication. Typical values:
- Small matrices (< 512): Try 32-64
- Medium matrices (512-2048): Try 64-128
- Large matrices (> 2048): Try 128-256

Experiment to find the optimal value for your hardware.

### Thread Count
- Start with your CPU's physical core count
- Avoid using hyperthreads for compute-intensive workloads
- More threads may not always be better due to overhead

### Matrix Size
- Strassen performs best on power-of-2 dimensions
- Padding to the next power of 2 can be added as an extension

## Example Output

```
******************* DIMENSSION=1024X1024, CROSSOVER=64, THREADS=4 *******************
Standard_Matrix_Multiplication    : Runtime = 2.341567 seconds.
Strassen's_Matrix_Multiplication  : Runtime = 1.876234 seconds.
seems ok! max_diff = 0.000000
```

## Validation

The program validates results by comparing Strassen output against standard multiplication:
- Tolerance threshold: `1E-6` (configurable via `THRESHOLD`)
- Checks for NaN and Inf values
- Reports maximum difference if validation fails

## Project Structure

```
├── main.c                    # Entry point, benchmarking, validation
├── matrix_data.h             # Matrix structure definition
├── matrix_ops.h/c            # Operation dispatcher and CPU detection
├── matrix_ops_avx.c          # AVX2/FMA optimized implementations
├── matrix_ops_fallback.c     # Portable fallback implementations
├── mm_funcs.h/c              # Strassen algorithm implementation
├── memory_pool.h/c           # Custom memory allocator
├── makefile                  # Build configuration
└── README.md                 # This file
```

## References

1. Cormen, T. H. (2009). *Introduction to algorithms*. 3rd ed. Cambridge, Mass: MIT Press.
2. Gregory, J. (2018). *Game engine architecture*. 3rd edition. Chapter 6.2: Memory Management.
3. Berger, E. D., McKinley, K. S., Blumofe, R. D., & Wilson, P. R. (2000). *Hoard: A scalable memory allocator for multithreaded applications*. ASPLOS IX.

## Limitations

- Requires matrix dimensions to be powers of 2 (padding can be added as future work)
- Optimized for double-precision floating-point only

## Future Enhancements

- [ ] Automatic padding for arbitrary matrix sizes
- [ ] Support for single-precision (float) operations
- [ ] Cache-oblivious optimizations
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Distributed computing support (MPI)

## Author

Haben Hadush
Department of Information Technology
Uppsala University

## License

Academic/Educational Project
