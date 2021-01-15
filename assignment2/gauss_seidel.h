/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

// Sequential version of Gauss-Seidel method
int gauss_seidel_seq(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance);

// Basic parallelization of Gauss-Seldel method
int gauss_seidel_paral(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance);

// Parallelized version of Jacobi method with while inside parallel region
int gauss_seidel_paral_while(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance);


#endif
