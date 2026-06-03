#!/bin/bash
#SBATCH --job-name=sigml-v2-diag
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --mem=8G
#SBATCH --output=logs/diagnose_v2_%j.out
#SBATCH --error=logs/diagnose_v2_%j.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate triqs_ctseg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_solver_v2}"
mkdir -p logs results

mpirun -np "${SLURM_NTASKS}" python scripts/solver/diagnose_v2.py \
  --mesh mesh_beta70.h5 \
  --weights orb1_weights.npz \
  --out results/diagnose_v2.json \
  --n-cycles "${SIGML_N_CYCLES:-100000}" \
  --n-warmup-cycles "${SIGML_N_WARMUP:-5000}" \
  --tol "${SIGML_TOL:-5e-3}" \
  --max-ctseg-iter "${SIGML_MAX_CTSEG_ITER:-8}"
