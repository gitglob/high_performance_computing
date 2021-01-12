void u_init(double ***u, int N) {
    int i, j, k;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {

                // i -> z, j -> y, k -> x
                if (j == N || k == N || k == 0 || i == 0 || i == N)
                    u[i][j][k] = 20.0;
                else u[i][j][k] = 0.0;
            }
        }
    }
}

void f_init(double ***f, int N) {
    int i, j, k;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {

                if (i >= N / 6.0 && i < 0.5 * N    // -2/3 <= z <= 0
                 && j >= 0 && j <= 0.25 * N        // -1 <= y <= -1/2
                 && k >= 0 && k <= 5.0 * N / 16.0) // -1 <= x <= -3/8
                    f[i][j][k] = 200.0;
                else f[i][j][k] = 0.0;
            }
        }
    }
}