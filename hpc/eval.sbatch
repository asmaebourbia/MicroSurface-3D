#!/bin/bash
#SBATCH --job-name=MicroSurfaceEval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
python -m microsurface.evaluate \
  --checkpoint runs/microsurface/last.pt --data data/surfaces.npz \
  --out runs/microsurface/evaluation --count 100 --device cuda
python -m microsurface.adaptive \
  --source runs/microsurface/evaluation/samples.npz \
  --out runs/microsurface/mesh_refinement --budget 500
python -m microsurface.photometric --out runs/microsurface/photometric
python -m microsurface.prune \
  --checkpoint runs/microsurface/last.pt --data data/surfaces.npz \
  --out runs/microsurface/pruning --amount 0.2 --device cuda
