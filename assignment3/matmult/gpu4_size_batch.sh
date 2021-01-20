# THIS IS A SPECIAL SHELL SCRIPT FOR FINDING THE OPTIMAL AMOUNT OF ELEMENTS IN GPU4
#
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -gpu "num=1"

# hpcintrogpush
module load cuda/11.1
module load gcc/9.2.0

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=matmult_f.nvcc

S="5040"
# Blk because we use this name to have additional variable available
PERM="blk"

#This defines amount of iterations of matrix calculation 
export MFLOPS_MAX_IT=1

ELEMSIZES="2 3 4 5 6 7"

# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0

# lscpu

# start the collect command with the above settings
for ELEMSIZE in $ELEMSIZES
do
    ./$EXECUTABLE $PERM $S $S $S $ELEMSIZE
done
