#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <stdio.h>

__global__
void gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, double ***temp_pointer, int delta_2, double div_val) {
    int i, j, k;
    for (i = 1; i < N - 1; ++i) {
        for (j = 1; j < N - 1; ++j) {
            for (k = 1; k < N - 1; ++k) {
                u[i][j][k] =  (u_old[i - 1][j][k] + u_old[i + 1][j][k]
                         + u_old[i][j - 1][k] + u_old[i][j + 1][k]
                         + u_old[i][j][k - 1] + u_old[i][j][k + 1]
                         + delta_2 * f[i][j][k]) * div_val;
            }
        }
    }

    temp_pointer = u;
    u = u_old;
    u_old = temp_pointer;
}

void run_gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter) {

    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double ***temp_pointer;

    while (*iter < iter_max) {
        gpu_jacobi_1<<<1, 1>>>(u, u_old, f, N, temp_pointer, delta_2, div_val);
        checkCudaErrors(cudaDeviceSynchronize());
        (*iter)++;
    }
}

__global__
void gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta_2, double div_val) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    //printf("x: %d, y: %d, z: %d\n", x,y,z);
    if (x > 0 && x < N - 1 && y > 0 && y < N - 1 && z > 0 && z < N - 1) {
            u[N * N * x + N * y + z] = (u_old[N * N * (x - 1) + N * y + z] + u_old[N * N * (x + 1) + N * y + z]
                                      + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                      + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                      + delta_2 * f[N * N * x + N * y + z]) * div_val;
    }
}

void run_gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block) {

    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer = NULL;

    while (*iter < iter_max) {
        gpu_jacobi_2<<<dim_grid, dim_block>>>(u, u_old, f, N, delta_2, div_val);
        checkCudaErrors(cudaDeviceSynchronize());
        (*iter)++;
        temp_pointer = u;
        u = u_old;
        u_old = temp_pointer;
    }
}

__global__
void gpu_jacobi_32(double *u, double *u_old, double *f, int N, int delta_2, double div_val,
                  double *u_old_) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x>0 && y>0 && z>0 && x<(N/2-1) && y<N-1 && z<N-1){ // inside the half cube
        u[N * N * x + N * y + z] = (u_old[N/2 * N * (x-1) + N * y + z] + u_old[N/2 * N * (x + 1) + N * y + z]
                                  + u_old[N/2 * N * x + N * (y - 1) + z] + u_old[N/2 * N * x + N * (y + 1) + z]
                                  + u_old[N/2 * N * x + N * y + (z - 1)] + u_old[N/2 * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N/2 * N * x + N * y + z]) * div_val;
    }
    else if (x == 0 && y>0 && z>0 && y<N-1 && z<N-1) { // border
        u[N * N * x + N * y + z] = (u_old_[N/2 * N * (N/2-1) + N * y + z] + u_old[N/2 * N * (x + 1) + N * y + z]
                                  + u_old[N/2 * N * x + N * (y - 1) + z] + u_old[N/2 * N * x + N * (y + 1) + z]
                                  + u_old[N/2 * N * x + N * y + (z - 1)] + u_old[N/2 * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N/2 * N * x + N * y + z]) * div_val;
    }
}

__global__
void gpu_jacobi_31(double *u, double *u_old, double *f, int N,int delta_2, double div_val,
                  double *u_old_) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x>0 && y>0 && z>0 && x<(N/2-1) && y<N-1 && z<N-1){ // inside the half cube
        u[N * N * x + N * y + z] = (u_old[N/2 * N * (x - 1) + N * y + z] + u_old_[N/2 * N * (x + 1) + N * y + z]
                                  + u_old[N/2 * N * x + N * (y - 1) + z] + u_old[N/2 * N * x + N * (y + 1) + z]
                                  + u_old[N/2 * N * x + N * y + (z - 1)] + u_old[N/2 * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N/2 * N * x + N * y + z]) * div_val;
    }
    else if (x == N/2-1 && y>0 && z>0 && y<N-1 && z<N-1) { // border
        u[N * N * x + N * y + z] = (u_old[N/2 * N * (x - 1) + N * y + z] + u_old_[N/2 * N * (0) + N * y + z]
                                  + u_old[N/2 * N * x + N * (y - 1) + z] + u_old[N/2 * N * x + N * (y + 1) + z]
                                  + u_old[N/2 * N * x + N * y + (z - 1)] + u_old[N/2 * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N/2 * N * x + N * y + z]) * div_val;
    }
}

void run_gpu_jacobi_3(double *u0, double *u0_old, double *f0, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block,
                      double *u1, double *u1_old, double *f1) {

    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer = NULL;

    while (*iter < iter_max) {
        cudaSetDevice(0);
        gpu_jacobi_31<<<dim_grid, dim_block>>>(u0, u0_old, f0, N, delta_2, div_val,
                                              u1_old);
        cudaSetDevice(1);
        gpu_jacobi_32<<<dim_grid, dim_block>>>(u1, u1_old, f1, N, delta_2, div_val,
                                              u0_old);
        cudaSetDevice(0);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaSetDevice(1);
        checkCudaErrors(cudaDeviceSynchronize());
        (*iter)++;
        temp_pointer = u0;
        u0 = u0_old;
        u0_old = temp_pointer;
        temp_pointer = u1;
        u1 = u1_old;
        u1_old = temp_pointer;
    }
}
