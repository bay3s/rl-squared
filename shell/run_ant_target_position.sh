#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=10:00:00

source /home/${USER}/.bashrc
source activate rl

srun python3 $HOME/rl-squared/runs/ant/run.py --env-name=ant_target_position --prod
