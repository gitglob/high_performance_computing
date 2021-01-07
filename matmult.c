#include <cblas.h>
#include "math.h"
#include "matmult.h"

// Basic version of matrix multiplication
void matmult_nat(int m, int n, int k, double **A, double **B, double **C) {
    matmult_mnk(m, n, k, A, B, C);
}

// Matrix multiplication with the use of CBLAS_DGEMM
void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, *A, k, *B, n, 0, *C, n);
}

// Matrix multiplication in order m->n->k
void matmult_mnk(int m, int n, int k, double **A, double **B, double **C) {
    int mm, nn, kk;
    for (mm = 0; mm < m; mm++) {
        for (nn = 0; nn < n; nn++) {
            C[mm][nn] = 0;
            for (kk = 0; kk < k; kk++) {
                C[mm][nn] += A[mm][kk] * B[kk][nn];
            }
        }
    }
}

// Matrix multiplication in order m->k->n
void matmult_mkn(int m, int n, int k, double **A, double **B, double **C) {
    int mm, nn, kk;
    for (mm = 0; mm < m; mm++) {
        for (kk = 0; kk < k; kk++) {
            for (nn = 0; nn < n; nn++) {
                if (kk == 0) C[mm][nn] = 0;
                C[mm][nn] += A[mm][kk] * B[kk][nn];
            }
        }
    }
}

// Matrix multiplication in order n->m->k
void matmult_nmk(int m, int n, int k, double **A, double **B, double **C) {
    int mm, nn, kk;
    for (nn = 0; nn < n; nn++) {
        for (mm = 0; mm < m; mm++) {
            for (kk = 0; kk < k; kk++) {
                if (kk == 0) C[mm][nn] = 0;
                C[mm][nn] += A[mm][kk] * B[kk][nn];
            }
        }
    }
}

// Matrix multiplication in order n->k->m
void matmult_nkm(int m, int n, int k, double **A, double **B, double **C) {
    int mm, nn, kk;
    for (nn = 0; nn < n; nn++) {
        for (kk = 0; kk < k; kk++) {
            for (mm = 0; mm < m; mm++) {
                if (kk == 0) C[mm][nn] = 0;
                C[mm][nn] += A[mm][kk] * B[kk][nn];
            }
        }
    }
}

// Matrix multiplication in order k->m->n
void matmult_kmn(int m, int n, int k, double **A, double **B, double **C) {
    int mm, nn, kk;
    for (kk = 0; kk < k; kk++) {
        for (mm = 0; mm < m; mm++) {
            for (nn = 0; nn < n; nn++) {
                if (kk == 0) C[mm][nn] = 0;
                C[mm][nn] += A[mm][kk] * B[kk][nn];
            }
        }
    }
}

// Matrix multiplication in order k->n->m
void matmult_knm(int m, int n, int k, double **A, double **B, double **C) {
    int mm, nn, kk;
    for (kk = 0; kk < k; kk++) {
        for (nn = 0; nn < n; nn++) {
            for (mm = 0; mm < m; mm++) {
                if (kk == 0) C[mm][nn] = 0;
                C[mm][nn] += A[mm][kk] * B[kk][nn];
            }
        }
    }
}

// Blocking version of matrix multiplication in order ->m->k->n
void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs) {
    int i, j;       // indices for filling C with 0
    int MM, KK, NN; // iterators over blocks
    int mm, kk, nn; // iterators inside blocks

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            C[mm][nn] = 0.0;

    for (MM = 0; MM < m; MM += bs) {
        for (KK = 0; KK < k; KK += bs) {
            for (NN = 0; NN < n; NN += bs) {

                for (kk = KK; kk < fmin(KK + bs, k); kk++) {
                    for (mm = MM; mm < fmin(MM + bs, m); mm++) {
                        for (nn = NN; nn < fmin(NN + bs, n); nn++)

                            C[mm][nn] += A[mm][kk] * B[kk][nn];
                    }
                }
            }
        }
    }
}