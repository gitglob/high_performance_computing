/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

void gauss_seidel_seq(double ***u, double ***f, int N, int delta, int iter_max, double tolerance);
void gauss_seidel(int N, double ***u, int iter_max, double tolerance, double start_T);


#endif
