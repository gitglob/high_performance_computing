void cpu_jacobi(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, int *iter) {
    int temp_iter = *iter;
    int i, j, k;
    double delta_2 = delta*delta;
    double div_val = 1.0/6.0;
    double ***temp_pointer;

    #pragma omp parallel default(none) \
            shared(iter_max, N, f, delta_2, div_val, u, u_old, temp_pointer, temp_iter) \
            private(i,j,k)
    {

        while (temp_iter < iter_max) {
            #pragma omp for
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

            // Pointer redo
            #pragma omp barrier
            #pragma omp single
            {
                temp_pointer = u;
                u = u_old;
                u_old = temp_pointer;
                temp_iter++;
            }
        }
    }
    *iter = temp_iter;
}
