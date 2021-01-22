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

#define BLOCK_SIZE 2

int main(int argc, char *argv[]) {
    int N = 0;
    int iter = 0;
    int iter_max = 100;
    double start_T = 0;
    int gpu_run = 0;
    int output_type = 0;

    N = atoi(argv[1]);    // grid size
    iter_max = atoi(argv[2]);  // max. no. of iterations
    start_T = atof(argv[3]);  // start T for all inner grid points
    gpu_run = atof(argv[4]); // 0 -> run CPU, 1/2/3-> run on GPU
    output_type = atof(argv[5]); // 0 -> run CPU, 1/2/3-> run on GPU

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

    int grid_size = (int)N/BLOCK_SIZE;
    if (N % BLOCK_SIZE > 0) grid_size++;
    printf("Grid size: %d", grid_size);

    dim3 dim_grid = dim3(grid_size, grid_size, grid_size);
    dim3 dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    double *u_old_1d_gpu = NULL;
    double *u_1d_gpu = NULL;
    double *f_1d_gpu = NULL;

    switch (gpu_run) { // GPU run
        case 0:{
            return 0;
            };
            break;

        case 1:{
            cudaSetDevice(DEVICE_0);
            start_time = omp_get_wtime();
            run_gpu_jacobi_1(u_gpu, u_old_gpu, f_gpu, N, delta, iter_max, &iter);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);

//            /* debug WORKS!*/
//            size = N * N * N  * sizeof(double);
//            cudaMalloc((void**)&u_old_1d_gpu, size);
//            cudaMalloc((void**)&u_1d_gpu, size);
//            cudaMalloc((void**)&f_1d_gpu, size);
//            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            double *huh=NULL,*hoh=NULL,*hfh=NULL;
//            cudaMallocHost((void**)&huh, size);
//            cudaMallocHost((void**)&hoh, size);
//            cudaMallocHost((void**)&hfh, size);
//            cudaMemcpy(huh, u_1d_gpu, size, cudaMemcpyDeviceToHost);
//            cudaMemcpy(hoh, u_old_1d_gpu, size, cudaMemcpyDeviceToHost);
//            cudaMemcpy(hfh, f_1d_gpu, size, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < N*N*N; ++i) {
//                if (huh[i] != 20 && huh[i] != 0) {
//                    printf("u_gpu[%d] = %f\n",i,huh[i]);
//                    //printf("u_old_gpu[%d] = %f\n",i,hoh[i]);
//                    //printf("f_gpu[%d] = %f\n",i,hfh[i]);
//                }
//            }


            };

            break;

        case 2:{
            cudaSetDevice(DEVICE_0);

            int size = N * N * N  * sizeof(double);

            cudaMalloc((void**)&u_old_1d_gpu, size);
            cudaMalloc((void**)&u_1d_gpu, size);
            cudaMalloc((void**)&f_1d_gpu, size);

            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);

            start_time = omp_get_wtime();
            run_gpu_jacobi_2(u_1d_gpu, u_old_1d_gpu, f_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block);
            end_time = omp_get_wtime();
            printf("GPU %d: iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);

            // debug
//            double *hu=NULL,*ho=NULL,*hf=NULL;
//            cudaMallocHost((void**)&hu, size);
//            cudaMallocHost((void**)&ho, size);
//            cudaMallocHost((void**)&hf, size);
//            cudaMemcpy(hu, u_1d_gpu, size, cudaMemcpyDeviceToHost);
//            cudaMemcpy(ho, u_old_1d_gpu, size, cudaMemcpyDeviceToHost);
//            cudaMemcpy(hf, f_1d_gpu, size, cudaMemcpyDeviceToHost);
//            for (int i = 0; i<N*N*N; i++){
//                if (hu[i] != 20 && hu[i] != 0) {
//                    printf("u_1d_gpu[%d] = %f\n",i,hu[i]);
//                    //printf("u_old_1d_gpu[%d] = %f\n",i,ho[i]);
//                    //printf("f_1d_gpu[%d] = %f\n",i,hf[i]);
//                }
//            }
//
//            transfer_3d_from_1d(u_old_gpu, u_old_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_from_1d(u_gpu, u_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_from_1d(f_gpu, f_1d_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//
//            cudaFree(u_old_1d_gpu);
//            cudaFree(u_1d_gpu);
//            cudaFree(f_1d_gpu);
            };
            break;

//        case 3:{
//            cudaSetDevice(DEVICE_0);
//            cudaDeviceEnablePeerAccess(1, 0);
//
//            size = N * N * N * sizeof(double);
//            dim_grid = dim3(N/BLOCK_SIZE, N/BLOCK_SIZE, N/BLOCK_SIZE);
//            dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
//
//            // both devices
//            cudaMalloc((void**)&u_old_1d_gpu, size);
//            cudaMalloc((void**)&u_1d_gpu, size);
//            cudaMalloc((void**)&f_1d_gpu, size);
//
//            // both devices
//            transfer_3d_to_1d(u_old_1d_gpu, u_old_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(u_1d_gpu, u_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//            transfer_3d_to_1d(f_1d_gpu, f_gpu, N, N, N, cudaMemcpyDeviceToDevice);
//
//            // device 0 mallocs
//            double *u0_old_1d_gpu = NULL;
//            double *u0_1d_gpu = NULL;
//            double *f0_1d_gpu = NULL;
//            cudaMalloc((void**)&u0_old_1d_gpu, size/2);
//            cudaMalloc((void**)&u0_1d_gpu, size/2);
//            cudaMalloc((void**)&f0_1d_gpu, size/2);
//
//            // device 0
//            cudaMemcpy(u0_old_1d_gpu, u_old_1d_gpu, N*N*N/2 * sizeof(double), cudaMemcpyDeviceToDevice);
//            cudaMemcpy(u0_1d_gpu, u_1d_gpu, N*N*N/2 * sizeof(double), cudaMemcpyDeviceToDevice);
//            cudaMemcpy(f0_1d_gpu, f_1d_gpu, N*N*N/2 * sizeof(double), cudaMemcpyDeviceToDevice);
//
//            cudaSetDevice(DEVICE_1);
//            cudaDeviceEnablePeerAccess(0, 0);
//
//            // device 1 mallocs
//            double *u1_old_1d_gpu = NULL;
//            double *u1_1d_gpu = NULL;
//            double *f1_1d_gpu = NULL;
//            cudaMalloc((void**)&u1_old_1d_gpu, size/2);
//            cudaMalloc((void**)&u1_1d_gpu, size/2);
//            cudaMalloc((void**)&f1_1d_gpu, size/2);
//
//            // device 1
//            cudaMemcpy(u1_old_1d_gpu, u_old_1d_gpu + (N*N*N/2), N*N*N/2 * sizeof(double), cudaMemcpyDeviceToDevice);
//            cudaMemcpy(u1_1d_gpu, u_1d_gpu + (N*N*N/2), N*N*N/2 * sizeof(double), cudaMemcpyDeviceToDevice);
//            cudaMemcpy(f1_1d_gpu, f_1d_gpu + (N*N*N/2), N*N*N/2 * sizeof(double), cudaMemcpyDeviceToDevice);
//
//            cudaSetDevice(DEVICE_0);
//            start_time = omp_get_wtime();
//            run_gpu_jacobi_3(u0_1d_gpu, u0_old_1d_gpu, f0_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block, u1_1d_gpu, u1_old_1d_gpu, f1_1d_gpu);
//            end_time = omp_get_wtime();
//            printf("GPU, DEVICE_0 (%d): iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);
//
//            // DEBUG
//            double *h_u00_1d_gpu = NULL;
//            cudaMallocHost((void**)&h_u00_1d_gpu, size/2);
//            cudaMemcpy(h_u00_1d_gpu, u0_1d_gpu, N*N*N/2 * sizeof(double), cudaMemcpyDeviceToHost);
//            printf("\n");
//            for (int i = 0; i<N*N*N/2; i++){
//              if (h_u00_1d_gpu[i] != 20 && h_u00_1d_gpu[i] != 0) {
//                printf("u0_1d_gpu[%d] = %f\n",i,h_u00_1d_gpu[i]);
//              }
//            }
//
//            // device 0 -> host
//            double *h_u_1d_gpu = NULL;
//            cudaMallocHost((void**)&h_u_1d_gpu, size);
//            cudaMemcpy(h_u_1d_gpu, u0_1d_gpu, N*N*N/2 * sizeof(double), cudaMemcpyDeviceToHost);
//
//            cudaSetDevice(DEVICE_1);
//            start_time = omp_get_wtime();
//            run_gpu_jacobi_3(u1_1d_gpu, u1_old_1d_gpu, f1_1d_gpu, N, delta, iter_max, &iter, dim_grid, dim_block, u0_1d_gpu, u0_old_1d_gpu, f0_1d_gpu);
//            end_time = omp_get_wtime();
//            printf("GPU, DEVICE_1 (%d): iterations done: %d time: %f\n", gpu_run, iter, end_time - start_time);
//
//            // DEBUG
//            double *h_u11_1d_gpu = NULL;
//            cudaMallocHost((void**)&h_u11_1d_gpu, size/2);
//            cudaMemcpy(h_u11_1d_gpu, u1_1d_gpu, N*N*N/2 * sizeof(double), cudaMemcpyDeviceToHost);
//            printf("\n");
//            for (int i = 0; i<N*N*N/2; i++){
//              if (h_u11_1d_gpu[i] != 20 && h_u11_1d_gpu[i] != 0) {
//                printf("u1_1d_gpu[%d] = %f\n",i,h_u11_1d_gpu[i]);
//              }
//            }
//
//            // device 1 -> host
//            cudaMemcpy(h_u_1d_gpu, u1_1d_gpu + (N*N*N/2), N*N*N/2 * sizeof(double), cudaMemcpyDeviceToHost);
//
//            cudaSetDevice(DEVICE_0);
//            // debug
//            for (int i = 0; i<N*N*N; i++){
//              if (h_u_1d_gpu[i] != 20 && h_u_1d_gpu[i] != 0) {
//                printf("u_1d_gpu[%d] = %f\n",i,h_u_1d_gpu[i]);
//              }
//            }
//
//            cudaFree(u_old_1d_gpu);
//            cudaFree(u_1d_gpu);
//            cudaFree(f_1d_gpu);
//            cudaFree(u0_old_1d_gpu);
//            cudaFree(u0_1d_gpu);
//            cudaFree(f0_1d_gpu);
//            cudaSetDevice(DEVICE_1);
//            cudaFree(u1_old_1d_gpu);
//            cudaFree(u1_1d_gpu);
//            cudaFree(f1_1d_gpu);
//
//            cudaDeviceDisablePeerAccess(DEVICE_0);
//            cudaDeviceDisablePeerAccess(DEVICE_1);
//            };
//            break;
    }

    transfer_3d(u, u_gpu, N, N, N, cudaMemcpyDeviceToHost);

    char *output_prefix = "poisson_res";
    char *output_ext = "";
    char output_filename[FILENAME_MAX];

    switch(output_type) {
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
