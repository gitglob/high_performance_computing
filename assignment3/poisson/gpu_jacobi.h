#ifndef __GPU_JACOBI
#define __GPU_JACOBI

void gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, double ***temp_pointer, int delta_2, double div_val);
void run_gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter);

void gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta_2, double div_val);
void run_gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block);

void gpu_jacobi_4(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double d);
void run_gpu_jacobi_4(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block, double *tolerance);

double warpReduceSum(double value);
void reduction_presum(double *a, int n, double *res);
void gpu_jacobi_5(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double *d);

void gpu_jacobi_5(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double d);
void run_gpu_jacobi_5(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block, double *tolerance);

#endif /* __GPU_JACOBI */