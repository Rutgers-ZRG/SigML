#!/bin/bash
#SBATCH --job-name=sigml-labels
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH --output=logs/gen_labels_%A_%a.out
#SBATCH --error=logs/gen_labels_%A_%a.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate triqs_ctseg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_solver_phasec}"
mkdir -p logs labels

mpirun -np "${SLURM_NTASKS}" python scripts/solver/gen_labels.py \
  --mesh mesh_beta70.h5 \
  --out-dir labels \
  --n-samples "${SIGML_SAMPLES_PER_TASK:-1}" \
  --seed "${SIGML_SEED:-20260603}" \
  --beta 70.0 \
  --n-cycles "${SIGML_N_CYCLES:-100000}" \
  --n-warmup-cycles "${SIGML_N_WARMUP:-5000}"
