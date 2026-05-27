#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/kongqingguo/bentele"
ENV_PY="/home/kongqingguo/miniconda3/envs/bentele-pi05/bin/python"
TORCHRUN="/home/kongqingguo/miniconda3/envs/bentele-pi05/bin/torchrun"
DATASET_ROOT="/home/kongqingguo/datasets/RoboTwin-LeRobot-v3.0-place_phone_stand/place_phone_stand"
SOURCE_RUN="/home/kongqingguo/bentele/model/pi05_place_phone_stand_vla_fullft_fsdp2_trainonly_10k_save500_20260525_123307"
SOURCE_CKPT="${SOURCE_RUN}/checkpoints/step_0010000"
DATA_ROOT="/data/120T/kqg/bentele/model"
COPIED_CKPT="${DATA_ROOT}/sft_full_checkpoints/pi05_place_phone_stand_vla_fullft_fsdp2_trainonly_10k_save500_20260525_123307/checkpoints/step_0010000"
RUN_NAME="pi05_place_phone_stand_vla_fullft_fsdp2_continue10k_from_step10000_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${DATA_ROOT}/${RUN_NAME}"
LOG_DIR="/data/120T/kqg/bentele/logs"
LOG_FILE="${LOG_DIR}/trigger_continue_full_sft_after_10k.log"

mkdir -p "${LOG_DIR}" "$(dirname "${COPIED_CKPT}")" "${OUT_DIR}"
exec >>"${LOG_FILE}" 2>&1

echo "[$(date '+%F %T')] watcher started"
echo "SOURCE_CKPT=${SOURCE_CKPT}"
echo "COPIED_CKPT=${COPIED_CKPT}"
echo "OUT_DIR=${OUT_DIR}"

while [ ! -f "${SOURCE_CKPT}/model_state_dict/full_weights.pt" ]; do
  echo "[$(date '+%F %T')] waiting for ${SOURCE_CKPT}/model_state_dict/full_weights.pt"
  sleep 120
done

prev_size=0
while true; do
  size="$(stat -c '%s' "${SOURCE_CKPT}/model_state_dict/full_weights.pt")"
  if [ "${size}" = "${prev_size}" ] && [ "${size}" -gt 1000000000 ]; then
    break
  fi
  prev_size="${size}"
  echo "[$(date '+%F %T')] waiting for checkpoint size to stabilize: ${size}"
  sleep 120
done

while pgrep -f "scripts/train_robotwin_vla.py .*--output-dir ${SOURCE_RUN}" >/dev/null; do
  echo "[$(date '+%F %T')] source run still active, waiting"
  sleep 120
done

echo "[$(date '+%F %T')] copying step_0010000 checkpoint to /data/120T"
rsync -a --delete "${SOURCE_CKPT}/" "${COPIED_CKPT}/"

cat >"${OUT_DIR}/run_command.txt" <<EOF
RUN_NAME=${RUN_NAME}
SOURCE_CKPT=${SOURCE_CKPT}
COPIED_CKPT=${COPIED_CKPT}
GPUS=1,2
TRAIN_STEPS=10000
NOTE=Second 10k SFT segment initialized from the first run's step_0010000 checkpoint.
EOF

echo "[$(date '+%F %T')] launching continuation"
cd "${REPO_DIR}"
CUDA_VISIBLE_DEVICES=1,2 \
PYTHONPATH=src:/home/kongqingguo/RoboTwin \
USE_TF=0 \
TRANSFORMERS_NO_TF=1 \
USE_TORCH=1 \
USE_FLAX=0 \
"${TORCHRUN}" --standalone --nproc_per_node=2 scripts/train_robotwin_vla.py \
  --dataset-root "${DATASET_ROOT}" \
  --train-repo-ids aloha-agilex_clean_50 aloha-agilex_randomized_500 \
  --model-path "${COPIED_CKPT}" \
  --output-dir "${OUT_DIR}" \
  --distributed-backend fsdp \
  --dist-timeout-minutes 60 \
  --batch-size 32 \
  --grad-accum-steps 1 \
  --train-steps 10000 \
  --log-every 10 \
  --save-every 500 \
  --save-step-checkpoints \
  --eval-every 0 \
  --action-probe-every 0 \
  --device cuda \
  --num-workers 0 \
  --wandb-enabled \
  --wandb-mode online \
  --wandb-project bentele \
  --wandb-group pi05_place_phone_stand_vla_fullft_fsdp2_trainonly_10k_save500_20260525_123307 \
  --wandb-run-name "${RUN_NAME}-train" \
  --wandb-log-dir "/data/120T/kqg/bentele/wandb" \
  > "${OUT_DIR}/train.log" 2>&1

echo "[$(date '+%F %T')] continuation finished"
