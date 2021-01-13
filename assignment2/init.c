void u_init(double ***u, int N, double start_T) {
    int i, j, k;
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

void f_init(double ***f, int N) {
    int i, j, k;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {

                if (i >= N / 6.0 - 1 && i < 0.5 * N - 1    // -2/3 <= z <= 0
                    && j >= 0 && j <= 0.25 * N - 1         // -1 <= y <= -1/2
                    && k >= 0 && k <= 5.0 * N / 16.0 - 1)  // -1 <= x <= -3/8
                    f[i][j][k] = 200.0;
                else f[i][j][k] = 0.0;
            }
        }
    }
}