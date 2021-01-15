/* jacobi.c - Poisson problem in 3d
 *
 */
#include <math.h>

// Sequential version of Jacobi method
int jacobi_seq(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double norm_diff, norm;
    double d;
    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;
    double ***temp_pointer;

    d = 10000;
    while (d > *tolerance && iter < iter_max) {
        d = 0.0;
        for (i = 1; i < N - 1; ++i) {
            for (j = 1; j < N - 1; ++j) {
                for (k = 1; k < N - 1; ++k) {
                    u[i][j][k] =  (u_old[i - 1][j][k] + u_old[i + 1][j][k]
                                + u_old[i][j - 1][k] + u_old[i][j + 1][k]
                                + u_old[i][j][k - 1] + u_old[i][j][k + 1]
                                + delta_2 * f[i][j][k])*div_val;

                    norm_diff = u[i][j][k] - u_old[i][j][k];
                    norm = norm_diff * norm_diff;
                    d += norm;
                }
            }
        }
        // Pointer redo
        d = sqrt(d);
        temp_pointer = u;
        u = u_old;
        u_old = temp_pointer;
        iter++;
    }
    *tolerance = d;
    return iter;
}

// Basic parallelization of Jacobi method
int jacobi_paral(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double norm_diff, norm;
    double d;
    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;
    double ***temp_pointer;

    d = 10000;
    while (d > *tolerance && iter < iter_max) {
        d = 0.0;
        #pragma omp parallel for default(none) \
        shared(N,f,delta_2,div_val,u,u_old,temp_pointer,iter) \
        private(i,j,k,norm_diff,norm) \
        reduction(+:d) schedule(static)
        for (i = 1; i < N - 1; ++i) {
            for (j = 1; j < N - 1; ++j) {
                for (k = 1; k < N - 1; ++k) {
                    u[i][j][k] =  (u_old[i - 1][j][k] + u_old[i + 1][j][k]
                                + u_old[i][j - 1][k] + u_old[i][j + 1][k]
                                + u_old[i][j][k - 1] + u_old[i][j][k + 1]
                                + delta_2 * f[i][j][k])*div_val;

                    norm_diff = u[i][j][k] - u_old[i][j][k];
                    norm = norm_diff * norm_diff;
                    d += norm;
                }
            }
        }
        // Pointer redo
        d = sqrt(d);
        temp_pointer = u;
        u = u_old;
        u_old = temp_pointer;
        iter++;
    }
    *tolerance = d;
    return iter;
}


// Parallelized version of Jacobi method with while inside parallel region
int jacobi_paral_while(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double norm_diff;
    double d, d_priv;
    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;
    double ***temp_pointer;

	#pragma omp parallel default(none) \
            shared(iter_max,tolerance,N,f,delta_2,div_val,u,u_old,temp_pointer,iter,d) \
            private(i,j,k,norm_diff,d_priv) 
    {
        d = 10000.0;

        while (d > *tolerance && iter < iter_max) {
            #pragma omp barrier
            #pragma omp single 
            {
                d = 0.0;
            }
            d_priv = 0.0;
            #pragma omp for
            for (i = 1; i < N - 1; ++i) {
                for (j = 1; j < N - 1; ++j) {
                    for (k = 1; k < N - 1; ++k) {
                        u[i][j][k] =  (u_old[i - 1][j][k] + u_old[i + 1][j][k]
                                 + u_old[i][j - 1][k] + u_old[i][j + 1][k]
                                 + u_old[i][j][k - 1] + u_old[i][j][k + 1]
                                 + delta_2 * f[i][j][k])*div_val;

                        norm_diff = u[i][j][k] - u_old[i][j][k];
                        d_priv += norm_diff * norm_diff;
                    }
                }
            }
            #pragma omp critical
            d +=d_priv;
            // Pointer redo
            #pragma omp barrier
            #pragma omp single
            {
            d = sqrt(d);
            temp_pointer = u;
            u = u_old;
            u_old = temp_pointer;
            iter++;
            }
        }
    }
    *tolerance = d;
    return iter;
}

// Failed attempt of parallelization of Jacobi method with while region inside and d as reduction variable
int jacobi_failed(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance) {
    int i, j, k;
    int iter = 0;
    double norm_diff;
    double d;
    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;
    double ***temp_pointer;

      #pragma omp parallel default(none) \
            shared(iter_max,tolerance,N,f,delta_2,div_val,u,u_old,temp_pointer,iter) \
            private(i,j,k,norm_diff) \
            reduction (+:d)
    {
        d = 10000;
        while (d > *tolerance && iter < iter_max) {
            #pragma omp barrier
            #pragma omp single
            {
                d = 0.0;
            }
            #pragma omp for
            for (i = 1; i < N - 1; ++i) {
                for (j = 1; j < N - 1; ++j) {
                    for (k = 1; k < N - 1; ++k) {
                        u[i][j][k] =  (u_old[i - 1][j][k] + u_old[i + 1][j][k]
                                 + u_old[i][j - 1][k] + u_old[i][j + 1][k]
                                 + u_old[i][j][k - 1] + u_old[i][j][k + 1]
                                 + delta_2 * f[i][j][k])*div_val;

                        norm_diff = u[i][j][k] - u_old[i][j][k];
                        d += norm_diff * norm_diff;
                    }
                }
            }
            // Pointer redo
            #pragma omp single
            {
              d = sqrt(d);
              temp_pointer = u;
              u = u_old;
              u_old = temp_pointer;
              iter++;
            }
        }
    }
    *tolerance = d;
    return iter;
}
