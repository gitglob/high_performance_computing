/* init.c - matrix initialization functions for poisson problem
 *
 */

// #################################################
// Jacobi parallel initialization - schedule(static)
// #################################################

// U matrix initialization, parallel
void u_init_jac(double ***u, int N, double start_T) {
    int i, j, k;

    #pragma omp parallel for default(none) \
            shared(u,N,start_T) \
            private(i,j,k) schedule(static)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {

                // i -> z, j -> y, k -> x
                if (j == 0)
                    u[i][j][k] = 0.0;

                else if (j == N - 1 || i == 0 || i == N - 1 || k == 0 || k == N - 1)
                    u[i][j][k] = 20.0;

                else
                    u[i][j][k] = start_T;
            }
        }
    }
}
// F matrix initialization, parallel
void f_init_jac(double ***f, int N) {
    int i, j, k;
    #pragma omp parallel for default(none) \
            shared(f,N) \
            private(i,j,k) schedule(static)
    for (i = 1; i < N-1; ++i) {
        for (j = 1; j < N-1; ++j) {
            for (k = 1; k < N-1; ++k) {

                if (i >= (N-2) / 6.0 && i < 0.5 * (N - 2)    // -2/3 <= z <= 0
                    && j >= 0 && j <= 0.25 * (N - 2)         // -1 <= y <= -1/2
                    && k >= 0 && k <= 5.0 * (N - 2) / 16.0 )  // -1 <= x <= -3/8
                    f[i][j][k] = 200.0;
                else f[i][j][k] = 0.0;
            }
        }
    }
}

// #########################################################
// Gauss-Seidel parallel initialization - schedule(static,1)
// #########################################################

// U matrix initialization, parallel
void u_init_gauss(double ***u, int N, double start_T) {
    int i, j, k;

    #pragma omp parallel for default(none) \
            shared(u,N,start_T) \
            private(i,j,k) schedule(static,1)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {

                // i -> z, j -> y, k -> x
                if (j == 0)
                    u[i][j][k] = 0.0;

                else if (j == N - 1 || i == 0 || i == N - 1 || k == 0 || k == N - 1)
                    u[i][j][k] = 20.0;

                else
                    u[i][j][k] = start_T;
            }
        }
    }
}
// F matrix initialization, parallel
void f_init_gauss(double ***f, int N) {
    int i, j, k;
    #pragma omp parallel for default(none) \
            shared(f,N) \
            private(i,j,k) schedule(static,1)
    for (i = 1; i < N-1; ++i) {
        for (j = 1; j < N-1; ++j) {
            for (k = 1; k < N-1; ++k) {

                if (i >= (N-2) / 6.0 && i < 0.5 * (N - 2)    // -2/3 <= z <= 0
                    && j >= 0 && j <= 0.25 * (N - 2)         // -1 <= y <= -1/2
                    && k >= 0 && k <= 5.0 * (N - 2) / 16.0 )  // -1 <= x <= -3/8
                    f[i][j][k] = 200.0;
                else f[i][j][k] = 0.0;
            }
        }
    }
}

// ########################
// OLD, SEQUENTIAL VERSIONS
// ########################

// U matrix initialization, sequential
void u_init_seq(double ***u, int N, double start_T) {
    int i, j, k;

    #pragma omp parallel for default(none) \
            shared(u,N,start_T) \
            private(i,j,k) schedule(static)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {

                // i -> z, j -> y, k -> x
                if (j == 0)
                    u[i][j][k] = 0.0;

                else if (j == N - 1 || i == 0 || i == N - 1 || k == 0 || k == N - 1)
                    u[i][j][k] = 20.0;

                else
                    u[i][j][k] = start_T;
            }
        }
    }
}
// F matrix initialization, sequential
void f_init_seq(double ***f, int N) {
    int i, j, k;
    #pragma omp parallel for default(none) \
            shared(f,N) \
            private(i,j,k) schedule(static)
    for (i = 1; i < N-1; ++i) {
        for (j = 1; j < N-1; ++j) {
            for (k = 1; k < N-1; ++k) {

                if (i >= (N-2) / 6.0 && i < 0.5 * (N - 2)    // -2/3 <= z <= 0
                    && j >= 0 && j <= 0.25 * (N - 2)         // -1 <= y <= -1/2
                    && k >= 0 && k <= 5.0 * (N - 2) / 16.0 )  // -1 <= x <= -3/8
                    f[i][j][k] = 200.0;
                else f[i][j][k] = 0.0;
            }
        }
    }
}