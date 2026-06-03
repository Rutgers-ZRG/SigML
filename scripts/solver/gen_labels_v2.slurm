#!/bin/bash
#SBATCH --job-name=sigml-v2-labels
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:12:00
#SBATCH --mem=8G
#SBATCH --array=0-49%12
#SBATCH --output=logs/gen_labels_v2_%A_%a.out
#SBATCH --error=logs/gen_labels_v2_%A_%a.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate triqs_ctseg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_solver_v2}"
mkdir -p logs labels_v2

mpirun -np "${SLURM_NTASKS}" python scripts/solver/gen_labels_v2.py \
  --mesh mesh_beta70.h5 \
  --weights orb1_weights.npz \
  --out-dir labels_v2 \
  --n-samples "${SIGML_SAMPLES_PER_TASK:-4}" \
  --seed "${SIGML_SEED:-20260603}" \
  --n-cycles "${SIGML_N_CYCLES:-30000}" \
  --n-warmup-cycles "${SIGML_N_WARMUP:-3000}" \
  --noise-every "${SIGML_NOISE_EVERY:-10}"
