#!/usr/bin/env bash
# Usage: PARTITION=gpu-2080ti CONDA_ENV=microsurface bash hpc/submit_pipeline.sh
# QOS defaults to long; override with QOS=normal if needed.
set -euo pipefail
mkdir -p logs runs data
if [[ -z "${PARTITION:-}" ]]; then
  echo 'Specify the exact SLURM partition: sinfo -h -o "%P %G"' >&2
  echo 'Then run: PARTITION=your_partition bash hpc/submit_pipeline.sh' >&2
  exit 2
fi
# Default to the user-confirmed L3i QOS long (7 days maximum).
# This is independent of each job's requested wall time (12h train / 2h eval).
# Override with QOS=normal if needed.
qos="${QOS:-long}"
qos_args=(--qos="$qos")
echo "Using QOS: $qos"
# SLURM may ignore the login-node activated conda environment, so propagate its name.
train_id=$(sbatch --parsable --partition="$PARTITION" "${qos_args[@]}" \
  --export="ALL,CONDA_ENV=${CONDA_ENV:-},EPOCHS=${EPOCHS:-50},BATCH=${BATCH:-32},LAMBDA_PHYSICS=${LAMBDA_PHYSICS:-5}" \
  hpc/train.sbatch)
train_id=${train_id%%;*}
eval_id=$(sbatch --parsable --partition="$PARTITION" "${qos_args[@]}" \
  --dependency="afterok:${train_id}" --export="ALL,CONDA_ENV=${CONDA_ENV:-}" \
  hpc/eval.sbatch)
eval_id=${eval_id%%;*}
echo "Submitted training job: $train_id"
echo "Submitted evaluation job: $eval_id (starts ONLY after successful training)"
echo "Monitor: squeue -j $train_id,$eval_id"
echo "Train log: logs/train_${train_id}.out"
