#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kongqingguo/bentele}"
CONDA_PYTHON="${CONDA_PYTHON:-/home/kongqingguo/miniconda3/envs/bentele-pi05/bin/python}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/kongqingguo/RoboTwin}"

export PYTHONPATH="${REPO_ROOT}/src:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_TORCH=1
export USE_FLAX=0

cd "${REPO_ROOT}"

exec "${CONDA_PYTHON}" -m bentele.cli.train_embodied \
  --config-name robotwin_place_phone_stand_ppo_openpi_pi05_2gpu_lora_rl_from_sft13k \
  "$@"
