#!/bin/bash

#SBATCH --time 2:30:00
#SBATCH --ntask
#SBATCH --gpus=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sirbay3s@gmail.com

# activate env
source /home/${USER}/.bashrc
source activate mujoco_env

# slurm run
srun $HOME/rl-squared/runs/bandits/run.py --n=10 --k=10
