#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00

source /home/${USER}/.bashrc
source activate rl

srun python3 $HOME/rl-squared/runs/bandits/run.py --n=500 --k=5 --prod
srun python3 $HOME/rl-squared/runs/bandits/run.py --n=500 --k=10 --prod
srun python3 $HOME/rl-squared/runs/bandits/run.py --n=500 --k=50 --prod
