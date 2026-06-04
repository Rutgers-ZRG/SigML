#!/bin/bash
#SBATCH --job-name=sigml-b40-t2g
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:08:00
#SBATCH --mem=12G
#SBATCH --array=0-149%24
#SBATCH --output=logs/bootstrap_t2g_b40_%A_%a.out
#SBATCH --error=logs/bootstrap_t2g_b40_%A_%a.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate solid_dmft

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_stage2/SigML}"
mkdir -p logs /scratch/lz432/sigml_stage2/labels/bootstrap_t2g_b40_shards

extra_args=()
if [[ "${SIGML_CONVERGED_ONLY:-0}" == "1" ]]; then
  extra_args+=(--converged-only)
fi

mpirun -np "${SLURM_NTASKS}" python scripts/solver/bootstrap_t2g_b40.py \
  --reference-h5 "${SIGML_SRVO3_REFERENCE_H5:-/scratch/lz432/sigml_stage2/srvo3_reference/run_public_config_64c_refined_40M/out/svo.h5}" \
  --out-dir "${SIGML_BOOTSTRAP_SHARD_DIR:-/scratch/lz432/sigml_stage2/labels/bootstrap_t2g_b40_shards}" \
  --seed "${SIGML_SEED:-20260603}" \
  --n-cycles "${SIGML_N_CYCLES:-300000}" \
  --n-warmup-cycles "${SIGML_N_WARMUP:-8000}" \
  --length-cycle "${SIGML_LENGTH_CYCLE:-120}" \
  --neighborhood "${SIGML_NEIGHBORHOOD:-0.025}" \
  --projection "${SIGML_PROJECTION:-direct}" \
  --solver-interface "${SIGML_SOLVER_INTERFACE:-g0}" \
  --verbosity "${SIGML_VERBOSITY:-0}" \
  "${extra_args[@]}"
