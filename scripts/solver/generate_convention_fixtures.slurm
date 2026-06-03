#!/bin/bash
#SBATCH --job-name=sigml-fixtures
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=logs/convention_fixtures_%j.out
#SBATCH --error=logs/convention_fixtures_%j.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate triqs_ctseg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_solver_closeout}"
mkdir -p logs results/fixtures

mpirun -np "${SLURM_NTASKS}" python scripts/solver/generate_convention_fixtures.py \
  --mesh mesh_beta70.h5 \
  --mldmft-root "${MLDMFT_ROOT:-/projects/f_lz432_1/li/mlDMFT}" \
  --out-dir results/fixtures \
  --n-cycles "${SIGML_FIXTURE_N_CYCLES:-100000}" \
  --n-warmup-cycles "${SIGML_FIXTURE_N_WARMUP:-5000}"
