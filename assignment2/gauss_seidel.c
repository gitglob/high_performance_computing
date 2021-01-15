/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include "gauss_seidel.h"

// Sequential version of Gauss-Seidel method
int gauss_seidel_seq(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double d = 10000000000;
    double norm_diff, u_old;

    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;

    while (d > *tolerance && iter < iter_max) {
        d = 0.0;
        for (i = 1; i < N - 1; ++i) {
            for (j = 1; j < N - 1; ++j) {
                for (k = 1; k < N - 1; ++k) {
                    double u_new = u[i - 1][j][k] + u[i + 1][j][k]
                                 + u[i][j - 1][k] + u[i][j + 1][k]
                                 + u[i][j][k - 1] + u[i][j][k + 1]
                                 + delta_2 * f[i][j][k];

                    u_old = u[i][j][k];
                    u[i][j][k] = u_new * div_val;
                    norm_diff = u[i][j][k] - u_old;
        
                    d += norm_diff * norm_diff;
                }
            }
        }
        d = sqrt(d);
        iter++;
    }
    *tolerance = d;
    return iter;
}

// Basic parallelization of Gauss-Seldel method
int gauss_seidel_paral(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double d = 1000;


    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;

    while (iter < iter_max) {
        d = 0.0;
        #pragma omp parallel default(none) \
                shared(N,f,delta_2,div_val,u) \
                private(i,j,k) 
        {
            #pragma omp for ordered(2) schedule(static,1)
            for (i = 1; i < N - 1; ++i) {
                for (j = 1; j < N - 1; ++j) {
                    #pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1) 
                    for (k = 1; k < N - 1; ++k) {
                        double u_new = u[i - 1][j][k] + u[i + 1][j][k]
                                    + u[i][j - 1][k] + u[i][j + 1][k]
                                    + u[i][j][k - 1] + u[i][j][k + 1]
                                    + delta_2 * f[i][j][k];

                        u[i][j][k] = u_new * div_val;
                    }
                    #pragma omp ordered depend(source)
                }
            }
        }
        iter++;
        
    }
    *tolerance = d;
    return iter;
}

// Parallelized version of Jacobi method with while inside parallel region
int gauss_seidel_paral_while(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double d = 1000;


    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;

    #pragma omp parallel default(none) \
        shared(N,f,delta_2,div_val,u,iter,iter_max) \
        private(i,j,k) 
    {
        while (iter < iter_max) {
            #pragma omp barrier

            #pragma omp for ordered(2) schedule(static,1)
            for (i = 1; i < N - 1; ++i) {
                for (j = 1; j < N - 1; ++j) {
                    #pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1) 
                    for (k = 1; k < N - 1; ++k) {
                        double u_new = u[i - 1][j][k] + u[i + 1][j][k]
                                    + u[i][j - 1][k] + u[i][j + 1][k]
                                    + u[i][j][k - 1] + u[i][j][k + 1]
                                    + delta_2 * f[i][j][k];

                        u[i][j][k] = u_new * div_val;
                    }
                    #pragma omp ordered depend(source)
                }
            }
            #pragma omp single 
            {
                iter++;
            }   
        }
    }
    *tolerance = d;
    return iter;
}
