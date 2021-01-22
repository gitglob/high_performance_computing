#include "gpu_jacobi.h"
#include "init.h"
#include "transfer3d_gpu.h"
#include "alloc3d_gpu.h"
#include "alloc3d.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
}

#define DEVICE_0 0
#define DEVICE_1 1

#define BLOCK_SIZE 8

int main(int argc, char *argv[]) {
    int N = 0;
    int iter = 0;
    int iter_max = 100;
    double start_T = 0;
    int gpu_run = 0;
    int output_type = 0;
    double tolerance;

    N = atoi(argv[1]);    // grid size
    iter_max = atoi(argv[2]);  // max. no. of iterations
    start_T = atof(argv[3]);  // start T for all inner grid points
    gpu_run = atof(argv[4]); // 0 -> run CPU, 1/2/3-> run on GPU
    output_type = atof(argv[5]); // 0 -> run CPU, 1/2/3-> run on GPU
    tolerance = atof(argv[6]);

    cudaSetDevice(DEVICE_0);

    double ***u_old = NULL; u_old = d_malloc_3d(N, N, N);
    double ***u = NULL;     u = d_malloc_3d(N, N, N);
    double ***f = NULL;     f = d_malloc_3d(N, N, N);

    u_init_jac(u_old, N, start_T);
    u_init_jac(u, N, start_T);
    f_init_jac(f, N);

    double ***u_old_gpu = NULL; u_old_gpu = d_malloc_3d_gpu(N, N, N);
    double ***u_gpu = NULL;     u_gpu = d_malloc_3d_gpu(N, N, N);
    double ***f_gpu = NULL;     f_gpu = d_malloc_3d_gpu(N, N, N);

    transfer_3d(u_old_gpu, u_old, N, N, N, cudaMemcpyHostToDevice);
    transfer_3d(u_gpu, u, N, N, N, cudaMemcpyHostToDevice);
    transfer_3d(f_gpu, f, N, N, N, cudaMemcpyHostToDevice);

    int grid_size = (int) N / BLOCK_SIZE;
    if (N % BLOCK_SIZE > 0) grid_size++;

    double delta = 2.0 / (N - 2);
    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double ***temp_pointer;
    double start_time, end_time;

    start_time = omp_get_wtime();
    while (*iter < iter_max) {
        gpu_jacobi_1<<<1, 1>>>(u_gpu, u_old_gpu, f_gpu, N, temp_pointer, delta_2, div_val);
        checkCudaErrors(cudaDeviceSynchronize());
        (*iter)++;
    }

    end_time = omp_get_wtime();
    printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);

    free_gpu(u_old_gpu);
    free_gpu(u_gpu);
    free_gpu(f_gpu);

    free(u_old);
    free(u);
    free(f);

    return 0;

//    switch (gpu_run) {
//        case 1: {
//
//        };
//
//            break;
//
//        case 2: {
//            cudaSetDevice(DEVICE_0);
//
//            int size = N * N * N * sizeof(double);
//            double *u_old_1d_gpu = NULL;
//            double *u_1d_gpu = NULL;
//            double *f_1d_gpu = NULL;
//
//            cudaMalloc((void **) &u_old_1d_gpu, size);
//            cudaMalloc((void **) &u_1d_gpu, size);
//            cudaMalloc((void **) &f_1d_gpu, size);
//
//            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            dim3 dim_grid = dim3(grid_size, grid_size, grid_size);
//            dim3 dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
//            start_time = omp_get_wtime();
//            run_gpu_jacobi_2(u_1d_gpu, u_old_1d_gpu, f_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block);
//            end_time = omp_get_wtime();
//            printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);
//
//            transfer_3d_from_1d(u_old_gpu, u_old_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_from_1d(u_gpu, u_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_from_1d(f_gpu, f_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//
//            cudaFree(u_old_1d_gpu);
//            cudaFree(u_1d_gpu);
//            cudaFree(f_1d_gpu);
//        };
//            break;
//
//        case 5: {
//            cudaSetDevice(DEVICE_0);
//
//            int size = N * N * N * sizeof(double);
//
//
//            cudaMalloc((void **) &u_old_1d_gpu, size);
//            cudaMalloc((void **) &u_1d_gpu, size);
//            cudaMalloc((void **) &f_1d_gpu, size);
//
//
//            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//
//            start_time = omp_get_wtime();
//            run_gpu_jacobi_5(u_1d_gpu, u_old_1d_gpu, f_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block,
//                             &tolerance);
//            end_time = omp_get_wtime();
//            printf("GPU %d: iterations done: %d time: %f, tolerance: %f\n", gpu_run, iter, end_time - start_time,
//                   tolerance);
//
//            // debug
//            double *hu = NULL, *ho = NULL, *hf = NULL;
//
//            transfer_3d_from_1d(u_old_gpu, u_old_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_from_1d(u_gpu, u_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_from_1d(f_gpu, f_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//
//            cudaFree(u_old_1d_gpu);
//            cudaFree(u_1d_gpu);
//            cudaFree(f_1d_gpu);
//        };
//            break;
//    }
}
