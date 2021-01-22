#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#define BLOCK_SIZE 8

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

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

//    printf("x: %d, y: %d, z: %d\n", x,y,z);
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

    //cudaMalloc((void**)&temp_pointer, N*N*N);

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
void gpu_jacobi_3(double *u, double *u_old, double *f, int N, double *temp_pointer, int delta_2, double div_val,
                  double *u_, double *u_old_, double *f_) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x == N/2 ) { // check if we are at the border for the second half
        printf("2nd half border!!\n");
        u[N * N * x + N * y + z] = (u_old_[N * N * (x - 1) + N * y + z] + u_old[N * N * (x + 1) + N * y + z]
                                  + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                  + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N * N * x + N * y + z]) * div_val;
    }
    else if (x == ((N/2) -1)){
        printf("1st half border!!\n");
        u[N * N * x + N * y + z] = (u_old[N * N * (x - 1) + N * y + z] + u_old_[N * N * (x + 1) + N * y + z]
                                  + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                  + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N * N * x + N * y + z]) * div_val;
    }
    else if (x > 1 && x < N - 1 && y > 1 && y < N - 1 && z > 1 && z < N - 1) {
        u[N * N * x + N * y + z] = (u_old[N * N * (x - 1) + N * y + z] + u_old[N * N * (x + 1) + N * y + z]
                                  + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                  + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                  + delta_2 * f[N * N * x + N * y + z]) * div_val;
    }

    temp_pointer = u;
    u = u_old;
    u_old = temp_pointer;
}

void run_gpu_jacobi_3(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block,
                      double *u_, double *u_old_, double *f_) {

    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer;

    while (*iter < iter_max) {
        gpu_jacobi_3<<<dim_grid, dim_block>>>(u, u_old, f, N, temp_pointer, delta_2, div_val,
                                              u_, u_old_, f_);
        checkCudaErrors(cudaDeviceSynchronize());
        (*iter)++;
    }
}


__global__
void gpu_jacobi_4(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double *d) {

    double value = 0;
    double norm_diff, norm;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1 && z > 0 && z < N - 1) {
            u[N * N * x + N * y + z] = (u_old[N * N * (x - 1) + N * y + z] + u_old[N * N * (x + 1) + N * y + z]
                                      + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                      + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                      + delta_2 * f[N * N * x + N * y + z]) * div_val;

            norm_diff = u[N * N * x + N * y + z] - u_old[N * N * x + N * y + z];
            norm = norm_diff * norm_diff;

            atomicAdd(d, norm);
    }

}

void run_gpu_jacobi_4(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block, double *tolerance) {
    double d = 100000;
    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer = NULL;

    double *d_gpu;
    cudaMalloc((void**)&d_gpu, sizeof(double));

    printf("Iter: %i\n", *iter);
    printf("Tolerance: %f\n", *tolerance);
    while (d > *tolerance && *iter < iter_max) {
        d = 0;
        cudaMemcpy(d_gpu, &d, sizeof(double), cudaMemcpyHostToDevice);
        gpu_jacobi_4<<<dim_grid, dim_block>>>(u, u_old, f, N, delta_2, div_val, d_gpu);
        cudaMemcpy(&d, d_gpu, sizeof(double), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize());
        d = sqrt(d);
        // printf("d: %f\n", d);
        temp_pointer = u;
        u = u_old;
        u_old = temp_pointer;
        (*iter)++;
    }

    *tolerance = d;
    cudaFree(d_gpu);
}


__inline__ __device__
double warpReduceSum(double value) {
    for(int i = 16; i > 0; i/=2)
        value += __shfl_down_sync(-1, value, i);
    return value;
}

__inline__ __device__
double blockReduceSum(double value) {
    __shared__ double smem[32]; // Max 32 warp sums
    int blockThreadIdx = (threadIdx.x + (threadIdx.y * BLOCK_SIZE) + (threadIdx.z * BLOCK_SIZE *BLOCK_SIZE));

    if (blockThreadIdx < 32)
        smem[blockThreadIdx] = 0;

    __syncthreads();
    value = warpReduceSum(value);

    if (blockThreadIdx % 32 == 0)
        smem[blockThreadIdx / warpSize] = value;

    __syncthreads();

    if (blockThreadIdx < 32)
        value = smem[blockThreadIdx];

    return warpReduceSum(value);
}

// __global__
// void reduction_presum(double *a, int n, double *res) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     double value = 0;
//     for (int i = idx; i < n; i += blockDim.x * gridDim.x)
//         value += a[i];

//     value = blockReduceSum(value);
//     if (threadIdx.x == 0)
//         atomicAdd(res, value);
// }

__global__
void gpu_jacobi_5(double *u, double *u_old, double *f, int N, int delta_2, double div_val, double *d) {

    double norm_diff, norm;

    norm = 0;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1 && z > 0 && z < N - 1) {
            u[N * N * x + N * y + z] = (u_old[N * N * (x - 1) + N * y + z] + u_old[N * N * (x + 1) + N * y + z]
                                      + u_old[N * N * x + N * (y - 1) + z] + u_old[N * N * x + N * (y + 1) + z]
                                      + u_old[N * N * x + N * y + (z - 1)] + u_old[N * N * x + N * y + (z + 1)]
                                      + delta_2 * f[N * N * x + N * y + z]) * div_val;

            norm_diff = u[N * N * x + N * y + z] - u_old[N * N * x + N * y + z];
            
            norm = norm_diff * norm_diff;  
    }
    norm = blockReduceSum(norm);
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            atomicAdd(d, norm);
}

void run_gpu_jacobi_5(double *u, double *u_old, double *f, int N, int delta, int iter_max, int *iter, dim3 dim_grid, dim3 dim_block, double *tolerance) {
    double d = 100000;
    double delta_2 = delta * delta;
    double div_val = 1.0 / 6.0;
    double *temp_pointer = NULL;

    double *d_gpu;
    cudaMalloc((void**)&d_gpu, sizeof(double));

    printf("Iter: %i\n", *iter);
    printf("Tolerance: %f\n", *tolerance);
    while (d > *tolerance && *iter < iter_max) {
        d = 0;
        cudaMemcpy(d_gpu, &d, sizeof(double), cudaMemcpyHostToDevice);
        gpu_jacobi_4<<<dim_grid, dim_block>>>(u, u_old, f, N, delta_2, div_val, d_gpu);
        cudaMemcpy(&d, d_gpu, sizeof(double), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize());
        d = sqrt(d);
        // printf("d: %f\n", d);
        temp_pointer = u;
        u = u_old;
        u_old = temp_pointer;
        (*iter)++;
    }

    *tolerance = d;
    cudaFree(d_gpu);
}