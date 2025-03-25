#!/bin/bash

#SBATCH --time=2:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64000M   # 64G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=delayed-opt-rerun

source /home/cayjobla/delayed-optimization-pytorch/.venv/bin/activate
wandb disabled
python rerun_best_hyperparams.py