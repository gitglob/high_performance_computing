#include "gpu_jacobi.h"
#include "cpu_jacobi.h"
#include "init.h"
#include "transfer3d_gpu.h"
#include "alloc3d_gpu.h"
#include "alloc3d.h"
#include "print.h"

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

    if (gpu_run == 3){
        N = 512; // this is hardcoded, it can probably be done another way
    }

    output_type = atof(argv[5]); // 3=.bin, 4=.vtk
    tolerance = atof(argv[6]);

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

    cudaSetDevice(DEVICE_0);
    u_old_gpu = d_malloc_3d_gpu(N, N, N);
    u_gpu = d_malloc_3d_gpu(N, N, N);
    f_gpu = d_malloc_3d_gpu(N, N, N);

    transfer_3d(u_old_gpu, u_old, N, N, N, cudaMemcpyHostToDevice);
    transfer_3d(u_gpu, u, N, N, N, cudaMemcpyHostToDevice);
    transfer_3d(f_gpu, f, N, N, N, cudaMemcpyHostToDevice);

    int grid_size = (int) N / BLOCK_SIZE;
    if (N % BLOCK_SIZE > 0) grid_size++;

    dim3 dim_grid;
    dim3 dim_block;
    if (gpu_run!=3){
      dim_grid = dim3(grid_size, grid_size, grid_size);
      dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    }

    double *u_old_1d_gpu = NULL;
    double *u_1d_gpu = NULL;
    double *f_1d_gpu = NULL;

    switch (gpu_run) { // GPU run
        case 0: {
            return 0;
        };
            break;

        case 1: {
            cudaSetDevice(DEVICE_0);
            start_time = omp_get_wtime();
            run_gpu_jacobi_1(u_gpu, u_old_gpu, f_gpu, N, delta, iter_max, &iter);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);
        };

            break;

        case 2: {
            cudaSetDevice(DEVICE_0);

            int size = N * N * N * sizeof(double);

            cudaMalloc((void **) &u_old_1d_gpu, size);
            cudaMalloc((void **) &u_1d_gpu, size);
            cudaMalloc((void **) &f_1d_gpu, size);

            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);

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
        };
            break;

        case 3: {
            // define size
            int size = N * N * N * sizeof(double);

            // create 2 sub-matrices in host
            double ***u0_old = NULL;
            double ***u0 = NULL;
            double ***f0 = NULL;
            double ***u1_old = NULL;
            double ***u1 = NULL;
            double ***f1 = NULL;
            u0_old = d_malloc_3d(N / 2, N, N);
            u0 = d_malloc_3d(N / 2, N, N);
            f0 = d_malloc_3d(N / 2, N, N);
            u1_old = d_malloc_3d(N / 2, N, N);
            u1 = d_malloc_3d(N / 2, N, N);
            f1 = d_malloc_3d(N / 2, N, N);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        if (i < N / 2) {
                            u0_old[i][j][k] = u[i][j][k];
                            u0[i][j][k] = u[i][j][k];
                            f0[i][j][k] = u[i][j][k];
                        }
                        else {
                            u1_old[i - (N / 2)][j][k] = u[i][j][k];
                            u1[i - (N / 2)][j][k] = u[i][j][k];
                            f1[i - (N / 2)][j][k] = u[i][j][k];
                        }
                    }
                }
            }

            // transfer matrices to devices
            double ***u0_old_gpu = NULL;
            double ***u0_gpu = NULL;
            double ***f0_gpu = NULL;
            cudaSetDevice(DEVICE_0);
            u0_old_gpu = d_malloc_3d_gpu(N / 2, N, N);
            u0_gpu = d_malloc_3d_gpu(N / 2, N, N);
            f0_gpu = d_malloc_3d_gpu(N / 2, N, N);
            transfer_3d(u0_old_gpu, u0_old, N / 2, N, N, cudaMemcpyHostToDevice);
            transfer_3d(u0_gpu, u0, N / 2, N, N, cudaMemcpyHostToDevice);
            transfer_3d(f0_gpu, f0, N / 2, N, N, cudaMemcpyHostToDevice);
            double ***u1_old_gpu = NULL;
            double ***u1_gpu = NULL;
            double ***f1_gpu = NULL;
            cudaSetDevice(DEVICE_1);
            u1_old_gpu = d_malloc_3d_gpu(N / 2, N, N);
            u1_gpu = d_malloc_3d_gpu(N / 2, N, N);
            f1_gpu = d_malloc_3d_gpu(N / 2, N, N);
            transfer_3d(u1_old_gpu, u1_old, N / 2, N, N, cudaMemcpyHostToDevice);
            transfer_3d(u1_gpu, u1, N / 2, N, N, cudaMemcpyHostToDevice);
            transfer_3d(f1_gpu, f1, N / 2, N, N, cudaMemcpyHostToDevice);
            checkCudaErrors(cudaDeviceSynchronize());

            // convert matrices to 1d
            cudaSetDevice(DEVICE_0);
            double *u0_old_1d_gpu = NULL;
            double *u0_1d_gpu = NULL;
            double *f0_1d_gpu = NULL;
            cudaMalloc((void **) &u0_old_1d_gpu, size / 2);
            cudaMalloc((void **) &u0_1d_gpu, size / 2);
            cudaMalloc((void **) &f0_1d_gpu, size / 2);
            transfer_3d_to_1d(u0_old_1d_gpu, u0_old_gpu, N / 2, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u0_1d_gpu, u0_gpu, N / 2, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f0_1d_gpu, f0_gpu, N / 2, N, N, cudaMemcpyDeviceToDevice);
            cudaSetDevice(DEVICE_1);
            double *u1_1d_gpu = NULL;
            double *u1_old_1d_gpu = NULL;
            double *f1_1d_gpu = NULL;
            cudaMalloc((void **) &u1_old_1d_gpu, size / 2);
            cudaMalloc((void **) &u1_1d_gpu, size / 2);
            cudaMalloc((void **) &f1_1d_gpu, size / 2);
            transfer_3d_to_1d(u1_old_1d_gpu, u1_old_gpu, N / 2, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u1_1d_gpu, u1_gpu, N / 2, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f1_1d_gpu, f1_gpu, N / 2, N, N, cudaMemcpyDeviceToDevice);

            cudaSetDevice(DEVICE_0);
            cudaDeviceEnablePeerAccess(1, 0);
            cudaSetDevice(DEVICE_1);
            cudaDeviceEnablePeerAccess(0, 0);

            checkCudaErrors(cudaDeviceSynchronize());

            printf("N: %d, Grid size: %d, Block size: %d\n", N, N / BLOCK_SIZE, BLOCK_SIZE);
            dim_grid = dim3(N / (BLOCK_SIZE*2), N / BLOCK_SIZE, N / BLOCK_SIZE);
            dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
            start_time = omp_get_wtime();
            run_gpu_jacobi_3(u0_1d_gpu, u0_old_1d_gpu, f0_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block,
                             u1_1d_gpu, u1_old_1d_gpu, f1_1d_gpu);
            end_time = omp_get_wtime();
            printf("2 GPUs (%d): iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);
            checkCudaErrors(cudaDeviceSynchronize());

            // return u0 to host
            cudaSetDevice(DEVICE_0);
            double *h_u0_1d_gpu = NULL;
            cudaMallocHost((void **) &h_u0_1d_gpu, size / 2);
            cudaMemcpy(h_u0_1d_gpu, u0_1d_gpu, size / 2, cudaMemcpyDeviceToHost);
            checkCudaErrors(cudaDeviceSynchronize());

            // return u1 to host
            cudaSetDevice(DEVICE_1);
            double *h_u1_1d_gpu = NULL;
            cudaMallocHost((void **) &h_u1_1d_gpu, size / 2);
            cudaMemcpy(h_u1_1d_gpu, u1_1d_gpu, size / 2, cudaMemcpyDeviceToHost);
            checkCudaErrors(cudaDeviceSynchronize());

            // combine matrices in host
            double *h_u_1d_gpu = NULL;
            cudaMallocHost((void **) &h_u_1d_gpu, size);
            cudaMemcpy(h_u_1d_gpu, h_u0_1d_gpu, size/2, cudaMemcpyHostToHost);
            checkCudaErrors(cudaDeviceSynchronize());
            cudaMemcpy(h_u_1d_gpu + (N * N * N / 2), h_u1_1d_gpu, size/2, cudaMemcpyHostToHost);
            checkCudaErrors(cudaDeviceSynchronize());

            // transfer u in host back to 3d
            transfer_3d_from_1d(u, h_u_1d_gpu, N, N, N, cudaMemcpyHostToHost);
            checkCudaErrors(cudaDeviceSynchronize());

            /* debug
            int sum3 =0;
            for (int i=0; i<N; i++){
              for (int j=0; j<N; j++){
                for (int k=0; k<N; k++){
                  if (u[i][j][k]<2){// && u[i][j][k]!=20){
                    sum3++;
                  }
                }
              }
            }
            printf("h_u 3D : %d %d\n",sum3,N*N*N);
            */

            cudaSetDevice(DEVICE_0);
            cudaFree(u0_old_1d_gpu);
            cudaFree(u0_1d_gpu);
            cudaFree(f0_1d_gpu);
            cudaSetDevice(DEVICE_1);
            cudaFree(u1_old_1d_gpu);
            cudaFree(u1_1d_gpu);
            cudaFree(f1_1d_gpu);

            cudaFreeHost(h_u0_1d_gpu);
            cudaFreeHost(h_u1_1d_gpu);
            cudaFreeHost(h_u_1d_gpu);

            cudaDeviceDisablePeerAccess(DEVICE_0);
            cudaDeviceDisablePeerAccess(DEVICE_1);
        };
            break;

        case 4: {
            cudaSetDevice(DEVICE_0);

            int size = N * N * N * sizeof(double);

            cudaMalloc((void **) &u_old_1d_gpu, size);
            cudaMalloc((void **) &u_1d_gpu, size);
            cudaMalloc((void **) &f_1d_gpu, size);

            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            start_time = omp_get_wtime();
            run_gpu_jacobi_4(u_1d_gpu, u_old_1d_gpu, f_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block,
                             &tolerance);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f, tolerance: %f\n", gpu_run, iter, end_time - start_time,
                   tolerance);

            transfer_3d_from_1d(u_old_gpu, u_old_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_from_1d(u_gpu, u_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_from_1d(f_gpu, f_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            cudaFree(u_old_1d_gpu);
            cudaFree(u_1d_gpu);
            cudaFree(f_1d_gpu);
        };
            break;

        case 5: {
            cudaSetDevice(DEVICE_0);

            int size = N * N * N * sizeof(double);

            cudaMalloc((void **) &u_old_1d_gpu, size);
            cudaMalloc((void **) &u_1d_gpu, size);
            cudaMalloc((void **) &f_1d_gpu, size);

            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            start_time = omp_get_wtime();
            run_gpu_jacobi_5(u_1d_gpu, u_old_1d_gpu, f_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block,
                             &tolerance);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f, tolerance: %f\n", gpu_run, iter, end_time - start_time,
                   tolerance);

            transfer_3d_from_1d(u_old_gpu, u_old_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_from_1d(u_gpu, u_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_from_1d(f_gpu, f_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            cudaFree(u_old_1d_gpu);
            cudaFree(u_1d_gpu);
            cudaFree(f_1d_gpu);
        };
            break;
    }

    if (gpu_run!=3){
      transfer_3d(u, u_gpu, N, N, N, cudaMemcpyDeviceToHost);
    }

    char *output_prefix = "poisson3_res";
    char *output_ext = "";
    char output_filename[FILENAME_MAX];

    switch (output_type) {
        case 0:
            // no output at all
            break;
        case 3:
            output_ext = ".bin";
            sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
            fprintf(stderr, "Write binary dump to %s: ", output_filename);
            print_binary(output_filename, N, u);
            break;
        case 4:
            output_ext = ".vtk";
            sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
            fprintf(stderr, "Write VTK file to %s: ", output_filename);
            print_vtk(output_filename, N, u);
            break;
        default:
            fprintf(stderr, "Non-supported output type!\n");
            break;
    }

    free_gpu(u_old_gpu);
    free_gpu(u_gpu);
    free_gpu(f_gpu);
    free(u);

    return 0;
}
