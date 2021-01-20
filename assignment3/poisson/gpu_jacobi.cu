#include <cuda_runtime_api.h>
#include <helper_cuda.h>

__global__
void gpu_jacobi(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max) {
    int i, j, k;
    int iter = 0;
    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;
    double ***temp_pointer;

        while (iter < iter_max) {
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
            iter++;
        }
    }
    return iter;
}