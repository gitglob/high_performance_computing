#include "gpu_jacobi.h"
#include "cpu_jacobi.h"
#include "init.h"
#include "transfer3d_gpu.h"
#include "alloc3d_gpu.h"
#include "alloc3d.h"

#include <cuda_runtime.h>

extern "C" {
    #include <stdio.h>
    #include <stdlib.h>
    #ifdef _OPENMP
    #include <omp.h>
    #endif
}

#define DEVICE_0 0
#define DEVICE_1 1

#define BLOCK_SIZE 16

int main(int argc, char *argv[]) {
    int N = 100;
    int iter = 0;
    int iter_max = 100;
    double start_T = 0;
    int gpu_run = 0;

    N = atoi(argv[1]);    // grid size
    iter_max = atoi(argv[2]);  // max. no. of iterations
    start_T = atof(argv[3]);  // start T for all inner grid points
    gpu_run = atof(argv[4]);  // start T for all inner grid points

    double ***u_old = NULL;
    double ***u = NULL;
    double ***f = NULL;

    double delta = 2.0 / (N - 2);

    u_old = d_malloc_3d(N, N, N);
    u = d_malloc_3d(N, N, N);
    f = d_malloc_3d(N, N, N);

    u_init_jac(u_old, N, start_T);
    u_init_jac(u, N, start_T);
    f_init_jac(f, N);

    double start_time, end_time;

    if (!gpu_run) { // CPU run
        start_time = omp_get_wtime();
        cpu_jacobi(u, u_old, f, N, delta, iter_max, &iter);
        end_time = omp_get_wtime();
        printf("CPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);

        free(u_old);
        free(u);
        free(f);

        return 0;
    }

    double ***u_old_gpu = NULL;
    double ***u_gpu = NULL;
    double ***f_gpu = NULL;

    u_old_gpu = d_malloc_3d_gpu(N, N, N);
    u_gpu = d_malloc_3d_gpu(N, N, N);
    f_gpu = d_malloc_3d_gpu(N, N, N);

    transfer_3d(u_old_gpu, u_old, N, N, N, cudaMemcpyHostToDevice);
    transfer_3d(u_gpu, u, N, N, N, cudaMemcpyHostToDevice);
    transfer_3d(f_gpu, f, N, N, N, cudaMemcpyHostToDevice);

    switch (gpu_run) { // GPU run
        case 0:
            return 0;

        case 1:
            cudaSetDevice(DEVICE_0);
            start_time = omp_get_wtime();
            run_gpu_jacobi_1(u_gpu, u_old_gpu, f_gpu, N, delta, iter_max, &iter);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);
            break;

        case 2:
            cudaSetDevice(DEVICE_0);
            double *u_old_1d_gpu = NULL;
            double *u_1d_gpu = NULL;
            double *f_1d_gpu = NULL;

            int size = N * N * N * sizeof(double);

            cudaMalloc((void**)&u_old_1d_gpu, size_f);
            cudaMalloc((void**)&u_1d_gpu, size_f);
            cudaMalloc((void**)&f_1d_gpu, size_f);

            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            dim3 dim_grid = dim3(N, N, N);
            dim3 dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

            start_time = omp_get_wtime();
            run_gpu_jacobi_2(u_1d_gpu, u_old_1d_gpu, f_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);

            transfer_3d_from_1d(u_old_gpu, u_old_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_from_1d(u_gpu, u_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_from_1d(f_gpu, f_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            cudaFree(u_old_1d_gpu);
            cudaFree(u_1d_gpu);
            cudaFree(f_1d_gpu);
            break;
    }

    free_gpu(u_old_gpu);
    free_gpu(u_gpu);
    free_gpu(f_gpu);

    return 0;
}
