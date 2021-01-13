#!/bin/bash
# 02614 - High-Performance Computing, January 2021

#BSUB -J jac
#BSUB -o jac_output/jac_%J.out
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

# define the driver name to use
#
EXECUTABLE=poisson_j

# define the number of threads
#
#THREADS=1
THREADS="1 2 4 8 16"

# define the grid size
#
GRID_SIZE=150

# define max iterations
#
MAX_ITER=18000

# define the tolerance
#
TOLERANCE=0.1

# define start for inner grid points
#
T_START=0

# start the collect command with the above settings
for T in $THREADS
do
    time OMP_NUM_THREADS=$T ./$EXECUTABLE $GRID_SIZE $MAX_ITER $TOLERANCE $T_START
done
