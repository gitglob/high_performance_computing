#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cublas_v2.h"

extern "C" {

    #include <cblas.h>
    #include "matmult.h"
    #include <omp.h>

    #define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
    #define BLOCK_SIZE 32

    
     // Matrix multiplication for gpu version 5
    void matmult_gpu5(int m, int n, int k, double *h_A, double *h_B, double *h_C) {
        double *d_A, *d_B, *d_C;
        double *d_Asub, *d_Bsub, *d_Csub;
        int grid_size_m, grid_size_n;

        // assume block size = 32x32
        grid_size_m = (int)m/BLOCK_SIZE;
        grid_size_n = (int)n/BLOCK_SIZE;

        //First is the number of elements in row, second in columns
        dim3 dimGrid(grid_size_n,grid_size_m,1); 
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1); //maximizng the amount of threads in a block

        cudaMalloc((void **)&d_A, m * k * sizeof(double)); 
        cudaMalloc((void **)&d_B, k * n * sizeof(double)); 
        cudaMalloc((void **)&d_C, m * n * sizeof(double));

        cudaMalloc((void **)&d_Asub, BLOCK_SIZE * k * sizeof(double)); 
        cudaMalloc((void **)&d_Bsub, k * BLOCK_SIZE * sizeof(double)); 
        cudaMalloc((void **)&d_Csub, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);
  
        kernel_gpu5<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C, d_Asub, d_Bsub, d_Csub);
        cudaDeviceSynchronize();
    

        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_Asub);
        cudaFree(d_Bsub);
        cudaFree(d_Csub);    
    }
    
    // Kernel for matrix multiplication for gpu version 5
    __global__ void kernel_gpu5(int m, int n, int k, double *A, double *B, double *C, double *Asub, double *Bsub, double *Csub) {
        // Block row and column
        int blockRow = blockIdx.y;
        int blockCol = blockIdx.x;
        int kk, e;

        Csub = &C[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

        double Cvalue = 0;
        int row = threadIdx.y;
        int col = threadIdx.x;

        for (kk = 0; kk < (k/BLOCK_SIZE); ++kk) {
            Asub = &A[k * BLOCK_SIZE * blockRow + BLOCK_SIZE * kk];
            Bsub = &B[n * BLOCK_SIZE * kk + BLOCK_SIZE * blockCol];

            __shared__ double As[BLOCK_SIZE*BLOCK_SIZE];
            __shared__ double Bs[BLOCK_SIZE*BLOCK_SIZE];

            As[row*BLOCK_SIZE + col] = Asub[row * k + col];
            Bs[row*BLOCK_SIZE + col] = Bsub[row * n + col];

            __syncthreads();
            
            for(e = 0; e <BLOCK_SIZE; e++) {
                Cvalue += As[row*BLOCK_SIZE + e]*Bs[e * BLOCK_SIZE + col];
            }

            __syncthreads();
        
        }

        Csub[row * n + col] = Cvalue;
    }


    // Matrix multiplication for gpu version 4
    void matmult_gpu4(int m, int n, int k, double *h_A, double *h_B, double *h_C) {
        double *d_A, *d_B, *d_C;
        int grid_size_m, grid_size_n;
        double time;

        // assume block size = 32x32
        grid_size_m = (int)(m/5)/32;
        grid_size_n = (int)n/32;
        if (m % 32 > 0) grid_size_m++;
        if (n % 32 > 0) grid_size_n++;
        //First is the number of elements in row, second in columns
        dim3 dimGrid(grid_size_n,grid_size_m,1); 
        dim3 dimBlock(32,32,1); //maximizng the amount of threads in a block

        cudaMalloc((void **)&d_A, m * k * sizeof(double)); 
        cudaMalloc((void **)&d_B, k * n * sizeof(double)); 
        cudaMalloc((void **)&d_C, m * n * sizeof(double));

        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        // time = omp_get_wtime();  
        kernel_gpu4<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
        cudaDeviceSynchronize();
        // printf("Calculation time = %3.2f seconds\n", omp_get_wtime() - time); 
    

        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);   
    }
    // Neighbour to the bottom version
    // Kernel for matrix multiplication for gpu version 4
    __global__ void kernel_gpu4(int m, int n, int k, double *A, double *B, double *C) {
        int kk;
        // int col = blockIdx.x * blockDim.x + threadIdx.x;
        // int row = blockIdx.y * blockDim.y + threadIdx.y;
        int nn = blockIdx.x * blockDim.x + threadIdx.x;
        int mm = blockIdx.y * blockDim.y + threadIdx.y;
        // We want the method to run for every size and 
        // there might be too much threads initialized because of that
        if (mm*5<m && nn<n) { 
            for (kk = 0; kk < k; kk++){
                if (kk == 0) C[mm*5*n + nn] = 0;
                C[mm*5*n + nn] += A[mm*5*k +kk]*B[kk*n + nn];
                if((mm*5)+1<m) C[(mm*5 + 1)*n + nn] += A[(mm*5 + 1)*k +kk]*B[kk*n + nn];
                if((mm*5)+2<m) C[(mm*5 + 2)*n + nn] += A[(mm*5 + 2)*k +kk]*B[kk*n + nn];
                if((mm*5)+3<m) C[(mm*5 + 3)*n + nn] += A[(mm*5 + 3)*k +kk]*B[kk*n + nn];
                if((mm*5)+4<m) C[(mm*5 + 4)*n + nn] += A[(mm*5 + 4)*k +kk]*B[kk*n + nn];
                
            } 
        }
    }

    // Matrix multiplication for tests of element size of gpu version 4
    void matmult_blk(int m, int n, int k, double *h_A, double *h_B, double *h_C, int elems) {
        double *d_A, *d_B, *d_C;
        int grid_size_m, grid_size_n;
        double time;

        // assume block size = 32x32
        grid_size_m = (int)(m/elems)/32;
        grid_size_n = (int)n/32;
        if (m % 32 > 0) grid_size_m++;
        if (n % 32 > 0) grid_size_n++;
        //First is the number of elements in row, second in columns
        dim3 dimGrid(grid_size_n,grid_size_m,1); 
        dim3 dimBlock(32,32,1); //maximizng the amount of threads in a block

        cudaMalloc((void **)&d_A, m * k * sizeof(double)); 
        cudaMalloc((void **)&d_B, k * n * sizeof(double)); 
        cudaMalloc((void **)&d_C, m * n * sizeof(double));

        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        // time = omp_get_wtime();  
        kernel_gpu4_test<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C, elems);
        cudaDeviceSynchronize();
        // printf("Calculation time = %3.2f seconds\n", omp_get_wtime() - time); 
    

        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);   
    }
    // Neighbour to the bottom version
    // Kernel for matrix multiplication for gpu version 4
    __global__ void kernel_gpu4_test(int m, int n, int k, double *A, double *B, double *C, int elems) {
        int kk, elemCounter;
        // int col = blockIdx.x * blockDim.x + threadIdx.x;
        // int row = blockIdx.y * blockDim.y + threadIdx.y;
        int nn = blockIdx.x * blockDim.x + threadIdx.x;
        int mm = blockIdx.y * blockDim.y + threadIdx.y;
        // We want the method to run for every size and 
        // there might be too much threads initialized because of that
        if (mm*elems<m && nn<n) { 
            for (kk = 0; kk < k; kk++){
                if (kk == 0) C[mm*elems*n + nn] = 0;
                C[mm*elems*n + nn] += A[mm*elems*k +kk]*B[kk*n + nn];
                for (elemCounter = 1; elemCounter<elems; elemCounter++){
                    if((mm*elems)+elemCounter<m) C[(mm*elems + elemCounter)*n + nn] += A[(mm*elems + elemCounter)*k +kk]*B[kk*n + nn];
                }
                // if((mm*2)+1<m) C[(mm*2 + 1)*n + nn] += A[(mm*2 + 1)*k +kk]*B[kk*n + nn];
                
            } 
        }
    }

    // Matrix multiplication for gpu version 3
    void matmult_gpu3(int m, int n, int k, double *h_A, double *h_B, double *h_C) {
        double *d_A, *d_B, *d_C;
        int grid_size_m, grid_size_n;
        double time;

        // assume block size = 32x32
        grid_size_m = (int)(m/2)/32;
        grid_size_n = (int)n/32;
        if (m % 32 > 0) grid_size_m++;
        if (n % 32 > 0) grid_size_n++;
        //First is the number of elements in row, second in columns
        dim3 dimGrid(grid_size_n,grid_size_m,1); 
        dim3 dimBlock(32,32,1); //maximizng the amount of threads in a block

        cudaMalloc((void **)&d_A, m * k * sizeof(double)); 
        cudaMalloc((void **)&d_B, k * n * sizeof(double)); 
        cudaMalloc((void **)&d_C, m * n * sizeof(double));

        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        // time = omp_get_wtime();  
        kernel_gpu3<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
        cudaDeviceSynchronize();
        // printf("Calculation time = %3.2f seconds\n", omp_get_wtime() - time); 
    

        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);   
    }
    // Neighbour to the bottom version
    // Kernel for matrix multiplication for gpu version 3
    __global__ void kernel_gpu3(int m, int n, int k, double *A, double *B, double *C) {
        int kk;
        // int col = blockIdx.x * blockDim.x + threadIdx.x;
        // int row = blockIdx.y * blockDim.y + threadIdx.y;
        int nn = blockIdx.x * blockDim.x + threadIdx.x;
        int mm = blockIdx.y * blockDim.y + threadIdx.y;
        // We want the method to run for every size and 
        // there might be too much threads initialized because of that
        if (mm*2<m && nn<n) { 
            for (kk = 0; kk < k; kk++){
                if (kk == 0) C[mm*2*n + nn] = 0;
                C[mm*2*n + nn] += A[mm*2*k +kk]*B[kk*n + nn];
                if((mm*2)+1<m) C[(mm*2 + 1)*n + nn] += A[(mm*2 + 1)*k +kk]*B[kk*n + nn];
                
            } 
        }
    }


    // Matrix multiplication for gpu version 2
    void matmult_gpu2(int m, int n, int k, double *h_A, double *h_B, double *h_C) {
        double *d_A, *d_B, *d_C;
        int grid_size_m, grid_size_n;
        double time;

        // assume block size = 32x32
        grid_size_m = (int)m/32;
        grid_size_n = (int)n/32;
        if (m % 32 > 0) grid_size_m++;
        if (n % 32 > 0) grid_size_n++;
        //First is the number of elements in row, second in columns
        dim3 dimGrid(grid_size_n,grid_size_m,1); 
        dim3 dimBlock(32,32,1); //maximizng the amount of threads in a block

        cudaMalloc((void **)&d_A, m * k * sizeof(double)); 
        cudaMalloc((void **)&d_B, k * n * sizeof(double)); 
        cudaMalloc((void **)&d_C, m * n * sizeof(double));

        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        // time = omp_get_wtime();  
        kernel_gpu2<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
        cudaDeviceSynchronize();
        // printf("Calculation time = %3.2f seconds\n", omp_get_wtime() - time); 
    

        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);   
    }

    // Kernel for matrix multiplication for gpu version 2
    __global__ void kernel_gpu2(int m, int n, int k, double *A, double *B, double *C) {
        int kk;
        // int col = blockIdx.x * blockDim.x + threadIdx.x;
        // int row = blockIdx.y * blockDim.y + threadIdx.y;
        int nn = blockIdx.x * blockDim.x + threadIdx.x;
        int mm = blockIdx.y * blockDim.y + threadIdx.y;
        // We want the method to run for every size and 
        // there might be too much threads initialized because of that
        if (mm<m && nn<n) { 
            for (kk = 0; kk < k; kk++){
                if (kk == 0) C[mm*n + nn] = 0;
                C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
            } 
        }
    }



    // Matrix multiplication for gpu version 1
    void matmult_gpu1(int m, int n, int k, double *h_A, double *h_B, double *h_C) {
        double *d_A, *d_B, *d_C;

        cudaMalloc((void **)&d_A, m * k * sizeof(double)); 
        cudaMalloc((void **)&d_B, k * n * sizeof(double)); 
        cudaMalloc((void **)&d_C, m * n * sizeof(double));

        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        kernel_gpu1<<<1,1>>>(m, n, k, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost); 
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C); 
    }

    // Kernel for matrix multiplication for gpu version 1
    __global__ void kernel_gpu1(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (mm = 0; mm < m; mm++) {
            for (kk = 0; kk < k; kk++){
                for (nn = 0; nn < n; nn++)  {
                    if (kk == 0) C[mm*n + nn] = 0;
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // Matrix multiplication with the use of CBLAS_DGEMM
    void matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,A,k,B,n,0,C,n);
    }

    // Matrix multiplication in order m->n->k
    void matmult_mnk(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (mm = 0; mm < m; mm++) {
            for (nn = 0; nn < n; nn++) {
                C[mm*n + nn] = 0;
                for (kk = 0; kk < k; kk++) {
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // Basic version of matrix multiplication
    void matmult_nat(int m, int n, int k, double *A, double *B, double *C) {
        matmult_mnk(m, n, k, A, B, C);
    }




    // NOT USED

    // Matrix multiplication in order m->k->n
    void matmult_mkn(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (mm = 0; mm < m; mm++) {
            for (kk = 0; kk < k; kk++){
                for (nn = 0; nn < n; nn++)  {
                    if (kk == 0) C[mm*n + nn] = 0;
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // Matrix multiplication in order n->m->k
    void matmult_nmk(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (nn = 0; nn < n; nn++) {
            for (mm = 0; mm < m; mm++) {
                for (kk = 0; kk < k; kk++)  {
                    if (kk == 0) C[mm*n + nn] = 0;
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // Matrix multiplication in order n->k->m
    void matmult_nkm(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (nn = 0; nn < n; nn++) {
            for (kk = 0; kk < k; kk++) {
                for (mm = 0; mm < m; mm++) {
                    if (kk == 0) C[mm*n + nn] = 0;
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // Matrix multiplication in order k->m->n
    void matmult_kmn(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (kk = 0; kk < k; kk++){
            for (mm = 0; mm < m; mm++){
                for (nn = 0; nn < n; nn++)  {
                    if (kk == 0) C[mm*n + nn] = 0;
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // Matrix multiplication in order k->n->m
    void matmult_knm(int m, int n, int k, double *A, double *B, double *C) {
        int mm, nn, kk;
        for (kk = 0; kk < k; kk++){
            for (nn = 0; nn < n; nn++) {
                for (mm = 0; mm < m; mm++) {
                    if (kk == 0) C[mm*n + nn] = 0;
                    C[mm*n + nn] += A[mm*k +kk]*B[kk*n + nn];
                }
            }
        }
    }

    // // Matrix multiplication in order m->k->n
    // void matmult_blk(int m, int n, int k, double *A, double *B, double *C, int bs) {
    //     int mm, nn, kk1, kk2;
    //     for (kk1 = 0; kk1 < k; kk1 +=bs){
    //         for (mm = 0; mm < m; mm++) {
    //             for(kk2=0;kk2<MIN(k-kk1, bs);kk2++){
    //                 for (nn = 0; nn < n; nn++)  {
    //                     if (kk1+kk2 == 0) C[mm*n + nn] = 0;
    //                     C[mm*n + nn]  += A[mm*k + kk1+kk2]*B[(kk1+kk2)*n + nn];
    //                 }
    //             }
    //         }
    //     }
    // }


}
