#!/bin/bash

#SBATCH -t 2:30:00
#SBATCH -n 10
#SBATCH --gpus=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sirbay3s@gmail.com

# load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

conda activate mujoco_env
python3 $HOME/rl-squared/rl_squared/runs/bandits/run.py --n=10 --k=10
