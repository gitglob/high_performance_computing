#ifndef __GPU_JACOBI
#define __GPU_JACOBI

int gpu_jacobi(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max);

#endif /* __ALLOC_3D_GPU */