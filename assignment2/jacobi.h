/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

// Sequential version of Jacobi method
int jacobi_seq(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance);

// Basic parallelization of Jacobi method
int jacobi_paral(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance);

// Parallelized version of Jacobi method with while inside parallel region
int jacobi_paral_while(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance);

// Failed attempt of parallelization of Jacobi method with while region inside and d as reduction variable
int jacobi_failed(double ***u, double ***u_old, double ***f, int N, int delta, int iter_max, double *tolerance);

#endif
