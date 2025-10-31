#include "memory_pool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#define DEBUG 0
#define CACHE_LINE_SIZE 64

MemoryPool *memory_pool_create() {
    MemoryPool *pool = (MemoryPool *)malloc(sizeof(MemoryPool));
    for (int i = 0; i < MAX_LEVELS; i++) {
        pool->free_lists[i] = NULL;
    }
    return pool;
}

void memory_pool_destroy(MemoryPool *pool) {
    for (int i = 0; i < MAX_LEVELS; i++) {
        MemoryBlock *block = pool->free_lists[i];
        while (block) {
            MemoryBlock *next = block->next;
            free(block);
            block = next;
        }
    }
    free(pool);
}

static inline int log2i(int n) {
    if (n <= 0) {
        fprintf(stderr, "ERROR: Invalid size for log2i: %d\n", n);
        return 0;
    }
    return (int)ceil(log2(n));
}

double *memory_pool_allocate(MemoryPool *pool, int size) {
    int k = log2i(size);
    if (k >= MAX_LEVELS) {
        fprintf(stderr, "Memory pool level exceeded\n");
        exit(EXIT_FAILURE);
    }
    if (pool->free_lists[k]) {
        MemoryBlock *block = pool->free_lists[k];
        pool->free_lists[k] = block->next;
        return block->data;
    } else {
        size_t block_size = size * size * sizeof(double);
        void *ptr;
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, block_size) != 0) {
            perror("posix_memalign");
            exit(EXIT_FAILURE);
        }
        return (double *)ptr;
    }
}

void memory_pool_deallocate(MemoryPool *pool, double *elements, int size) { 
    int k = log2i(size);
    if (k >= MAX_LEVELS)
        return;
    MemoryBlock *block = (MemoryBlock *)elements;
    block->next = pool->free_lists[k];
    pool->free_lists[k] = block;
}
