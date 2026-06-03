#!/bin/bash
#SBATCH --job-name=sigml-phasec
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate triqs_ctseg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_solver_phasec}"
mkdir -p logs results

mpirun -np "${SLURM_NTASKS}" python scripts/solver/benchmark_phase_c.py \
  --mesh mesh_beta70.h5 \
  --weights orb1_weights.npz \
  --out-dir results \
  --u-values ${SIGML_U_VALUES:-2.0 4.0} \
  --mott-u-values ${SIGML_MOTT_U_VALUES:-2.0 3.0 4.0 5.0 6.0} \
  --beta 70.0 \
  --n-cycles "${SIGML_N_CYCLES:-100000}" \
  --n-warmup-cycles "${SIGML_N_WARMUP:-5000}" \
  --max-ctseg-iter "${SIGML_MAX_CTSEG_ITER:-12}" \
  --tol "${SIGML_TOL:-5e-3}"
