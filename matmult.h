#ifndef __MATMULT_H
#define __MATMULT_H


// m - number of rows of matrix A (rows of matrix C)
// n - number of columns of matrix B (columns of matrix C)
// k - number of columns of matrix A (rows of matrix B)
// A - input matrix 1
// B - input matrix 2
// C - output matrix
void matmult_nat(int m, int n, int k, double **A, double **B, double **C);

// m - number of rows of matrix A (rows of matrix C)
// n - number of columns of matrix B (columns of matrix C)
// k - number of columns of matrix A (rows of matrix B)
// A - input matrix 1
// B - input matrix 2
// C - output matrix
void matmult_mnk(int m, int n, int k, double **A, double **B, double **C);

// m - number of rows of matrix A (rows of matrix C)
// n - number of columns of matrix B (columns of matrix C)
// k - number of columns of matrix A (rows of matrix B)
// A - input matrix 1
// B - input matrix 2
// C - output matrix
void matmult_lib(int m, int n, int k, double **A, double **B, double **C);

#endif