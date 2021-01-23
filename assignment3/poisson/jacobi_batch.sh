#BSUB -J jacobi
#BSUB -q gpua100i
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=2048]"

module load cuda/11.1
module load gcc/9.2.0

export TMPDIR=$__LSF_JOB_TMPDIR__
export MFLOPS_MAX_IT=1
export CUDA_VISIBLE_DEVICES=0,1

nv-nsight-cu-cli -o profile_BLOCK_SIZE_32 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStats \
    --section LaunchStats \
    --section Occupancy \
    ./jacobi.nvcc 512 1 0 3 0 0.1
