/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>

#include "init.h"
#include "alloc3d.h"
#include "gauss_seidel.h"

int gauss_seidel_seq(double ***u, double ***f, int N, int delta, int iter_max, double tolerance) {
    int i, j, k;
    int iter = 0;
    double d = 10000000000;
    double norm_diff, u_old;

    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;

    while (d > tolerance && iter < iter_max) {
        d = 0.0;
        for (i = 1; i < N - 1; ++i) {
            for (j = 1; j < N - 1; ++j) {
                for (k = 1; k < N - 1; ++k) {
                    // Make a delta square - done
                    double u_new = u[i - 1][j][k] + u[i + 1][j][k]
                                 + u[i][j - 1][k] + u[i][j + 1][k]
                                 + u[i][j][k - 1] + u[i][j][k + 1]
                                 + delta_2 * f[i][j][k];

                    u_old = u[i][j][k];
                    // Make it a mult - done
                    u[i][j][k] = u_new * div_val;
                    norm_diff = u[i][j][k] - u_old;
        
                    d += norm_diff * norm_diff;
                }
            }
        }
        d = sqrt(d);
        iter++;
    }
    return iter;
}
