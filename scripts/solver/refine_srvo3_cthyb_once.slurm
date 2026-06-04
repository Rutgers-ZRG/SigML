#!/bin/bash
#SBATCH --job-name=sigml-srvo3-nn1
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --output=logs/refine_srvo3_nn1_%j.out
#SBATCH --error=logs/refine_srvo3_nn1_%j.err

set -euo pipefail

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate solid_dmft

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SIGML_RUN_DIR:-/scratch/lz432/sigml_stage2/SigML}"
mkdir -p logs "${SIGML_REFINE_OUT_DIR:-/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_warm_tail_mix002_cthyb}"

mpirun -np "${SLURM_NTASKS}" python scripts/solver/refine_srvo3_cthyb_once.py \
  --harness-npz "${SIGML_HARNESS_NPZ:?Set SIGML_HARNESS_NPZ to the NN harness observables npz}" \
  --out-dir "${SIGML_REFINE_OUT_DIR:-/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_warm_tail_mix002_cthyb}" \
  --n-cycles "${SIGML_N_CYCLES:-300000}" \
  --n-warmup-cycles "${SIGML_N_WARMUP:-8000}" \
  --length-cycle "${SIGML_LENGTH_CYCLE:-120}" \
  --seed "${SIGML_SEED:-20260603}"
