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

# define the mkn values in the MKN variable
#
# SIZES="5000"
S="5040"

# define the permutation type in PERM
#
PERMS="gpu2 gpu3 gpu4"

#This defines amount of iterations of matrix calculation 
export MFLOPS_MAX_IT=1


# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0

# lscpu

# start the collect command with the above settings
# for S in $SIZES
for PERM in $PERMS
do
    ./$EXECUTABLE $PERM $S $S $S 
    # numactl --cpunodebind=0 ./$EXECUTABLE $PERM $S $S $S
done
