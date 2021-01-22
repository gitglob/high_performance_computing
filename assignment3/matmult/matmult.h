#ifndef __MATMULT_H
#define __MATMULT_H

void matmult_gpulib(int m, int n, int k, double *h_A, double *h_B, double *h_C) ;

void matmult_gpu5(int m, int n, int k, double *h_A, double *h_B, double *h_C);
__global__ void kernel_gpu5(int m, int n, int k, double *a, double *b, double *c, double *Asub, double *Bsub, double *Csub);

void matmult_gpu(int m, int n, int k, double *h_A, double *h_B, double *h_C);
__global__ void kernel_gpu4(int m, int n, int k, double *a, double *b, double *c);

// Test of amount of elements in gpu4 implementation
// hidden under name matmult_blk, because one more parameter was needed
void matmult_blk(int m, int n, int k, double *h_A, double *h_B, double *h_C, int elems);
__global__ void kernel_gpu4_test(int m, int n, int k, double *a, double *b, double *c, int elems);

void matmult_gpu3(int m, int n, int k, double *h_A, double *h_B, double *h_C);
__global__ void kernel_gpu3(int m, int n, int k, double *a, double *b, double *c);

void matmult_gpu2(int m, int n, int k, double *h_A, double *h_B, double *h_C);
__global__ void kernel_gpu2(int m, int n, int k, double *a, double *b, double *c);

void matmult_gpu1(int m, int n, int k, double *h_A, double *h_B, double *h_C);
__global__ void kernel_gpu1(int m, int n, int k, double *a, double *b, double *c);



void matmult_nat(int m, int n, int k, double *A, double *B, double *C);

void matmult_mnk(int m, int n, int k, double *A, double *B, double *C);
void matmult_mkn(int m, int n, int k, double *A, double *B, double *C);
void matmult_nmk(int m, int n, int k, double *A, double *B, double *C);
void matmult_mkm(int m, int n, int k, double *A, double *B, double *C);
void matmult_knm(int m, int n, int k, double *A, double *B, double *C);
void matmult_kmn(int m, int n, int k, double *A, double *B, double *C);

void matmult_lib(int m, int n, int k, double *A, double *B, double *C);


// void matmult_blk(int m, int n, int k, double *A, double *B, double *C, int bs);

#endif