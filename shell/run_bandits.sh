#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=thin
#SBATCH --time=2:00:00

source /home/${USER}/.bashrc
source activate mujoco_env

# slurm run
srun python3 $HOME/rl-squared/runs/bandits/run.py --k=5 --n=100
