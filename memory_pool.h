/*
REFERENCE
[1] : Gregory, J. (2018) Game engine architecture. 3rd edition. Boca Raton, FL: A K Peters/CRC Press, an imprint of Taylor and Francis.
Chapter 6.2 : Memory Managment

url: https://uub.primo.exlibrisgroup.com/discovery/fulldisplay?docid=alma991018333674107596&context=L&vid=46LIBRIS_UUB:UUB&lang=en&search_scope=MyInst_and_CI&adaptor=Local%20Search%20Engine&tab=Everything&query=any,contains,Game%20Engine%20Architecture,%20Third%20Edition

[2] : Hoard: a scalable memory allocator for multithreaded applications
url : https://dl-acm-org.ezproxy.its.uu.se/doi/10.1145/384264.379232

[3] : COMP4300 - Game Programming
url:     https://www.youtube.com/watch?v=hngvIDUMD88&list=PL_xRyXins849E1WPuutEApdyTa0Bfxhzq
*/

#include <omp.h>
#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H
#define MAX_LEVELS 32 // Support for matrices up to 2^32

typedef union MemoryBlock {
    union MemoryBlock *next;
    double data[1];
} MemoryBlock;

typedef struct {
    MemoryBlock *free_lists[MAX_LEVELS];
} MemoryPool;

MemoryPool *memory_pool_create();
void memory_pool_destroy(MemoryPool *pool);
double *memory_pool_allocate(MemoryPool *pool, int size);
void memory_pool_deallocate(MemoryPool *pool, double *elements, int size);
#endif
