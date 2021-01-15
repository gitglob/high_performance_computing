#!/bin/bash

#BSUB -J poisson_j
#BSUB -o results_poisson_j_%J/poisson_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 10

TARGET=poisson_j

JID=${LSB_JOBID}
EXPOUT="${TARGET}.${JID}.er"
EXECUTABLE=${TARGET}
OUTDIR="results_${TARGET}_"${JID}""

lscpu
mkdir ${OUTDIR}

sizes=(10 30 50 70 90 110 130 150)

iter_max=20000  #max. no. of iterations
tolerance=0.1    #tolerance
start_T=0      #start T for all inner grid points
output_type=0  #ouput type

for i in "${sizes[@]}"
do
    ./$EXECUTABLE $i $iter_max $tolerance $start_T $output_type >> ${OUTDIR}/"${EXECUTABLE}".txt
done
