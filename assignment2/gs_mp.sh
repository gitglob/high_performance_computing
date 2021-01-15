#!/bin/bash

#BSUB -J poisson_gs_mp
#BSUB -o results_poisson_gs_%J/results_poisson_gs_%J.out
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "rusage[mem=2048]"
#BSUB -W 20

TARGET=poisson_gs

JID=${LSB_JOBID}
EXPOUT="${TARGET}.${JID}.er"
EXECUTABLE=${TARGET}
OUTDIR="results_${TARGET}_"${JID}""

lscpu
mkdir ${OUTDIR}

threads=(1 2 4 8 16)

N=150
iter_max=18000  #max. no. of iterations
tolerance=0.1    #tolerance
start_T=0      #start T for all inner grid points
output_type=0  #ouput type

export OMP_PROC_BIND=close
export OMP_PLACES=cores

for i in "${threads[@]}"
do
    time OMP_NUM_THREADS=$i ./$EXECUTABLE $N $iter_max $tolerance $start_T $output_type >> ${OUTDIR}/"${EXECUTABLE}".txt
done

export OMP_PROC_BIND=false