#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m3 5000 > m3.out
# srun nsys profile --stats=true ./m3 5000 > profile.out

# FOLDER=test_kernel_fusion

# { time srun ./$FOLDER/m3 100 > $FOLDER/m3_100.out; } 2> $FOLDER/time_m3_100.out
# { time srun ./$FOLDER/m3 1000 > $FOLDER/m3_1000.out; } 2> $FOLDER/time_m3_1000.out
# { time srun ./$FOLDER/m3 10000 > $FOLDER/m3_10000.out; } 2> $FOLDER/timem3_10000.out



# FOLDER=test_fp16

# { time srun ./$FOLDER/m3 100 > $FOLDER/m3_100.out; } 2> $FOLDER/time_m3_100.out
# { time srun ./$FOLDER/m3 1000 > $FOLDER/m3_1000.out; } 2> $FOLDER/time_m3_1000.out
# { time srun ./$FOLDER/m3 10000 > $FOLDER/m3_10000.out; } 2> $FOLDER/timem3_10000.out

# srun nsys profile --stats=true ./$FOLDER/m3 > $FOLDER/profile.out

# Generate analysis_file.ncu-rep with a detailed profiling result
FOLDER=test_kernel_fusion
srun ncu -f --set full -o ./$FOLDER/analysis_file ./$FOLDER/m3 > ./$FOLDER/m3.out
FOLDER=test_loop_unroll
srun ncu -f --set full -o ./$FOLDER/analysis_file ./$FOLDER/m3 > ./$FOLDER/m3.out
FOLDER=test_restrict
srun ncu -f --set full -o ./$FOLDER/analysis_file ./$FOLDER/m3 > ./$FOLDER/m3.out
FOLDER=test_streams
srun ncu -f --set full -o ./$FOLDER/analysis_file ./$FOLDER/m3 > ./$FOLDER/m3.out
FOLDER=test_tensor_core
srun ncu -f --set full -o ./$FOLDER/analysis_file ./$FOLDER/m3 > ./$FOLDER/m3.out