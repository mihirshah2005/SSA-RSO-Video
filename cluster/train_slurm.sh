#!/bin/bash
# Train the temporal heatmap detector on the NUS SoC Compute Cluster (Slurm).
#
#   ssh <soc_unix_id>@xlogin.comp.nus.edu.sg
#   cd ~/Video_CV && sbatch cluster/train_slurm.sh                                 # default config, any GPU
#   sbatch --gpus=a100-80 cluster/train_slurm.sh                                   # pick a GPU type
#   sbatch cluster/train_slurm.sh configs/train_heatmap.yaml --pilot               # 200-step pilot first
#   CHAIN=1 sbatch cluster/train_slurm.sh configs/train_heatmap.yaml --set frames=1 --set out_dir=runs/k1
#
# SoC GPU jobs run up to 3 hours (default limit 15 min unless --time is given). The trainer
# checkpoints every epoch (atomically) to <out_dir>/last.pt, resumes automatically, and writes
# <out_dir>/DONE when it finishes or stops early. With CHAIN=1 each job queues its successor
# *before* training (afterany dependency), so a walltime kill still continues; a job that finds
# DONE exits immediately and stops the chain.
# Reference: https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus
#
#SBATCH --job-name=rso-heatmap
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=rso-heatmap-%j.out

set -euo pipefail
CONFIG=${1:-configs/train_heatmap.yaml}
shift || true
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# one-time environment setup (kept in ~/venvs/rso)
if [ ! -d "$HOME/venvs/rso" ]; then
  python3 -m venv "$HOME/venvs/rso"
  source "$HOME/venvs/rso/bin/activate"
  pip install --upgrade pip
  pip install -e ".[train]"
else
  source "$HOME/venvs/rso/bin/activate"
fi

OUT=$(python -m starship_rso.train.train_heatmap --config "$CONFIG" "$@" --print-out-dir)
if [ -f "$OUT/DONE" ]; then
  echo "training in $OUT already finished: $(cat "$OUT/DONE")"
  exit 0
fi

IS_PILOT=0
for a in "$@"; do [ "$a" = "--pilot" ] && IS_PILOT=1; done
if [ "${CHAIN:-0}" = "1" ] && [ "$IS_PILOT" = "0" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  NEXT=$(CHAIN=1 sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" "$0" "$CONFIG" "$@")
  echo "queued continuation job $NEXT (runs after this one ends; exits at once if $OUT/DONE exists)"
fi

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m starship_rso.train.train_heatmap --config "$CONFIG" "$@"
