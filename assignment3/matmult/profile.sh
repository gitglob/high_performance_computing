#BSUB -J proftest
#BSUB -q hpcintrogpu
#BSUB -n 4
#BSUB -R "span[hosts=1]" 
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -W 10
#BSUB -R "rusage[mem=2048]" 

export TMPDIR=$__LSF_JOB_TMPDIR__
module load cuda/11.1

export MFLOPS_MAX_IT=1 

nv-nsight-cu-cli -o profile_$LSB_JOBID \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStats \
    --section LaunchStats \
    --section Occupancy \
    ./matmult_f.nvcc gpu3 2048 2048 2048