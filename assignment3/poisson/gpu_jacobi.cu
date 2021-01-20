#include <cuda_runtime_api.h>
#include <helper_cuda.h>

__global__
void gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, double *temp_pointer, int delta_2, double div_val) {
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
    checkCudaErrors(cudaDeviceSynchronize());
}

void run_gpu_jacobi_1(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter) {

    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer;

    while (*iter < iter_max) {
        gpu_jacobi_1_thread<<<1, 1>>>(u, u_old, f, N, temp_pointer, delta_2, div_val);
        iter++;
    }
}

__global__
void gpu_jacobi_2(double *u, double *u_old, double *f, int N, double *temp_pointer, int delta_2, double div_val) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x > 1 && x < N - 1 && y > 1 && y < N - 1 && z > 1 && z < N - 1) {
                u[N * N * x + N * y + z] = (u_old[N * N * (x - 1) + N * y + z] + u_old[N * N * (x + 1) + N * y + z]
                                          + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                          + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                          + delta_2 * f[N * N * x + N * y + z]) * div_val;
            }

            temp_pointer = u;
            u = u_old;
            u_old = temp_pointer;
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
}

void run_gpu_jacobi_2(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block) {

    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer;

    while (*iter < iter_max) {
        gpu_jacobi_2<<<dim_grid, dim_block>>>(u, u_old, f, N, temp_pointer, delta_2, div_val);
        iter++;
    }
}