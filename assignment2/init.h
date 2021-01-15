/* init.c - matrix initialization functions for poisson problem
 *
 */

#ifndef _INIT_H
#define _INIT_H

// Jacobi parallel initialization - schedule(static)
void u_init_jac(double ***u, int N,double start_T);
void f_init_jac(double ***f, int N);

// Gauss-Seidel parallel initialization - schedule(static,1)
void u_init_gauss(double ***u, int N,double start_T);
void f_init_gauss(double ***f, int N);

// OLD, SEQUENTIAL VERSIONS
void u_init_seq(double ***u, int N,double start_T);
void f_init_seq(double ***f, int N);

#endif