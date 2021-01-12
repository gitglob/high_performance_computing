/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include "init.h"
#include "alloc3d.h"

void gauss_seidel_seq(double ***u, double ***f, int N, int delta, int iter_max, double tolerance) {
    int i, j, k;
    int iter = 0;
    double d = 10000000000;
    double norm_diff, u_old;

    while (d > tolerance && iter < iter_max) {
        d = 0.0;
        for (i = 1; i < N - 1; ++i) {
            for (j = 1; j < N - 1; ++j) {
                for (k = 1; k < N - 1; ++k) {

                    double u_new = u[i - 1][j][k] + u[i + 1][j][k]
                                 + u[i][j - 1][k] + u[i][j + 1][k]
                                 + u[i][j][k - 1] + u[i][j][k + 1]
                                 + delta * delta * f[i][j][k];

                    u_old = u[i][j][k];
                    u[i][j][k] = u_new / 6.0;
                    norm_diff = u[i][j][k] - u_old;

                    d += sqrt(norm_diff * norm_diff);
                }
            }
        }
        iter++;
    }
}

void gauss_seidel(int N, double ***u, int iter_max, double tolerance) {
    // fill in your code here

    double delta = 1.0 / N;
    double ***f = d_malloc_3d(N, N, N);

    f_init(f, N);
    u_init(u, N);

    gauss_seidel_seq(u, f, N, delta, iter_max, tolerance);
}
