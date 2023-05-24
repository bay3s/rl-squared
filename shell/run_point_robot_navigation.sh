#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00

source /home/${USER}/.bashrc
source activate rl

srun python3 $HOME/rl-squared/runs/point_robot/run.py --env-name=point_robot_navigation --prod
