#ifndef __GPU_JACOBI
#define __GPU_JACOBI

void gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, double ***temp_pointer, int delta_2, double div_val);
void run_gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter);

void gpu_jacobi_2(double *u, double *u_old, double *f, int N, double, int delta_2, double div_val);
void run_gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block);

void gpu_jacobi_31(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double *u_old_);
void gpu_jacobi_32(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double *u_old_);
void run_gpu_jacobi_3(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block, double *u_, double *u_old_, double *f_);

#endif /* __GPU_JACOBI */
