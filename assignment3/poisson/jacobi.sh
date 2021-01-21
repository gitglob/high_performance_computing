#BSUB -J jacobi
#BSUB -o results/jacobi_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -gpu "num=1"

# hpcintrogpush
module load cuda/11.1
module load gcc/9.2.0

EXECUTABLE=jacobi.nvcc

#This defines amount of iterations of matrix calculation
export MFLOPS_MAX_IT=1

lscpu

N=100
ITERATIONS=100
START_T=0
VERSIONS=(0)

for i in "${VERSIONS[@]}"
do
    ./$EXECUTABLE $N $ITERATIONS $START_T $i >> results/jacobi_"${LSB_JOBID}".txt
done
