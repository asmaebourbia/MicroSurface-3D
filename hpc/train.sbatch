#!/bin/bash
#SBATCH --job-name=MicroSurfaceGAN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# Select your cluster's exact partition with: sbatch --partition=YOUR_PARTITION ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
python - <<'PY'
import torch
print('PyTorch:',torch.__version__, 'CUDA:',torch.cuda.is_available(),flush=True)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: wrong PyTorch build or GPU not allocated')
print('GPU:',torch.cuda.get_device_name(0), flush=True)
PY
python -m microsurface.train \
  --data data/surfaces.npz --out runs/microsurface \
  --epochs "${EPOCHS:-50}" --batch "${BATCH:-32}" \
  --lambda-physics "${LAMBDA_PHYSICS:-5}" --workers 2 --device cuda
