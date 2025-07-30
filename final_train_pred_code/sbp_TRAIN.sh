#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00



source ~/.bashrc
module use /projects/community/modulefiles
module load git
conda activate sigml
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python train.py
