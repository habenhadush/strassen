#ifndef MATRIX_DATA_H
#define MATRIX_DATA_H

typedef struct {
    double *restrict elements;
    int i_offset;
    int j_offset;
    int ndim;
    int stride;
} Matrix;
#endif