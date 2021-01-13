#!/bin/bash

#BSUB -J poisson
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

sizes=(10 20 50 75)

iter_max=1000  #max. no. of iterations
tolerance=1    #tolerance
start_T=0      #start T for all inner grid points
output_type=4  #ouput type

for i in "${sizes[@]}"
do
    ./$EXECUTABLE $i $iter_max $tolerance $start_T $output_type > ${OUTDIR}/"$i".txt
done
