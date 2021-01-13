/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

int gauss_seidel_seq(double ***u, double ***f, int N, int delta, int iter_max, double *tolerance);



#endif
