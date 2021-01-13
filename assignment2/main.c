/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <omp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#include "init.h"

#define N_DEFAULT 100

// void gauss_seidel(int n, double ***pDouble, int max, double tolerance, double start_T);

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    // Our variables start
    double ***f = NULL, ***u_old = NULL;
    int iterations_done;
    double start_time, end_time;
    // Our variables end

    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    // allocate memory
    if ( (u = d_malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    // Our code start
    double delta = 2.0 / (N-2);
    if ( (f = d_malloc_3d(N, N, N)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (u_old = d_malloc_3d(N, N, N)) == NULL ) {
        perror("array u_old: allocation failed");
        exit(-1);
    }

    u_init(u, N, start_T);
    f_init(f, N);

    #ifdef _GAUSS_SEIDEL
    start_time = omp_get_wtime();
    iterations_done = gauss_seidel_seq(u, f, N, delta, iter_max, &tolerance);
    end_time = omp_get_wtime();
    #endif

    #ifdef _JACOBI
    u_init(u_old, N, start_T);
    start_time = omp_get_wtime();
    iterations_done = jacobi(u, u_old, f, N, delta, iter_max, &tolerance);
    end_time = omp_get_wtime();
    #endif

   
    // Uncomment for descriptive output
    printf("iterations done: %d tolerance: %f time: %f\n", iterations_done, tolerance, end_time-start_time);
    // Uncomment for output that can be used in plotting
    // printf("%d %f %f\n", iterations_done, tolerance, end_time-start_time);
    // OUR CODE END

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);

    return(0);
}
