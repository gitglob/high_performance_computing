#ifndef __GPU_JACOBI
#define __GPU_JACOBI

#include <cuda_runtime_api.h>
#include <helper_cuda.h>


void gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, double *temp_pointer, int delta_2, double div_val);
void run_gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter);

void gpu_jacobi_2(double *u, double *u_old, double *f, int N, double *temp_pointer, int delta_2, double div_val);
void run_gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, int dim_grid, int dim_block);

#endif /* __GPU_JACOBI */