#ifndef __CPU_JACOBI
#define __CPU_JACOBI

int cpu_jacobi(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter);

#endif /* __CPU_JACOBI */