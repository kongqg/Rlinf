# RLinf

这个仓库按上游 RLinf 的目录风格组织：核心框架代码放在 `src/rlinf/`，RoboTwin/pi0.5 相关代码放在 `src/rlinf/projects/robotwin/`。

## 目录结构

```text
src/
  rlinf/
    algorithms/                 # RL 算法和 loss
    data/                       # 通用数据结构
    envs/                       # 环境封装，包括 RoboTwin env wrapper
    hybrid_engines/             # FSDP / 分布式训练相关工具
    models/                     # embodied model wrapper，包括 OpenPI/pi0.5
    projects/
      robotwin/
        adapters/               # RoboTwin task adapter
        cli/                    # 可 import 的命令行入口
        configs/                # RoboTwin 实验 Hydra 配置
        data/                   # RoboTwin dataset adapter
        integration/            # project config 到 RLinf runner 的桥接层
        robotwin/               # dataset、eval、norm stats、probe
        runtime/                # path、env var、logging、seed、device 工具
        training/
          sft/                  # 兼容旧 import 的 SFT wrapper
          local_rl/             # 兼容旧 import 的 local RL wrapper
        utils/                  # 兼容旧 eval API 的项目工具
    runners/                    # RLinf runner
    scheduler/                  # cluster、placement、worker scheduling
    training/                   # 通用 training 入口、协议和 common helper
    utils/                      # RLinf 通用工具
    workers/                    # actor/env/rollout/reward worker

  openpi/                       # vendored OpenPI runtime 子集
  openpi_client/                # vendored OpenPI client helper

scripts/                        # 保持兼容的脚本入口
agents/                         # 本地多 agent role prompt
docs/                           # pipeline 说明文档
examples/                       # 轻量 example
skills/                         # 本地 Codex skill
model/                          # 本地实验产物，不是源码主路径
```

## RoboTwin 项目模块

- `cli/`：薄入口，只负责解析 CLI，然后调用训练或评测逻辑。
- `adapters/`：RoboTwin task adapter，负责把 dataset、model、env、eval 接到通用 training 入口。
- `data/`：本地 RoboTwin V3 数据集 adapter。
- `configs/embodiment/`：RoboTwin `place_phone_stand` 的 Hydra 配置。
- `integration/rlinf_embodied.py`：校验 Hydra config，并启动 RLinf embodied runner。
- `robotwin/eval.py` 和 `utils/robotwin_eval.py`：policy eval 逻辑和旧 API 兼容层。
- `robotwin/norm_stats.py`：state/action normalization stats 计算。
- `training/sft/`：兼容旧 import 的 SFT wrapper。
- `training/local_rl/`：兼容旧 import 的 local RL wrapper。
- `runtime/`：项目共享的路径、环境变量、随机种子、device、logging 工具。

## 安装

```bash
pip install -r requirements.txt
python scripts/patch_transformers_for_openpi.py
```

只有机器人节点需要相机或 spacemouse 相关依赖时，才安装：

```bash
pip install -r requirements-realworld.txt
```

只有跑 ManiSkill 相关 smoke path 时，才安装：

```bash
pip install -r requirements-maniskill.txt
```

## 基本使用示例

先按本地实际路径设置这些变量：

```bash
cd /Path/to/Project
export PYTHONPATH=$PWD/src

export ROBOTWIN_ASSETS=/path/to/RoboTwin
export DATASET_ROOT=/path/to/robotwin_lerobot_datasets
export BASE_PI05=/path/to/pi05_base_torch
export OUTPUT_ROOT=/data/120T/kqg/bentele/model
```

### 1. 计算 norm stats（pi0.5要用）

```bash
python scripts/compute_robotwin_norm_stats.py \
  --dataset-dirs "$DATASET_ROOT/place_phone_stand" \
  --output-dir "$BASE_PI05" \
  --config-name pi05_aloha_robotwin
```

### 2. 跑 full VLA SFT

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_robotwin_vla.py \
  --dataset-root "$DATASET_ROOT" \
  --train-repo-ids place_phone_stand \
  --model-path "$BASE_PI05" \
  --output-dir "$OUTPUT_ROOT/pi05_place_phone_stand_vla_fullft" \
  --assets-path "$ROBOTWIN_ASSETS" \
  --batch-size 64 \
  --train-steps 10000 \
  --save-every 500 \
  --save-step-checkpoints \
  --eval-every 0 \
  --wandb-enabled
```

两卡 FSDP：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_robotwin_vla.py \
  --dataset-root "$DATASET_ROOT" \
  --train-repo-ids place_phone_stand \
  --model-path "$BASE_PI05" \
  --output-dir "$OUTPUT_ROOT/pi05_place_phone_stand_vla_fullft_fsdp2" \
  --assets-path "$ROBOTWIN_ASSETS" \
  --batch-size 64 \
  --train-steps 10000 \
  --save-every 500 \
  --save-step-checkpoints \
  --distributed-backend fsdp \
  --eval-every 0 \
  --wandb-enabled
```

### 3. 跑 LoRA SFT

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_robotwin_vla_lora.py \
  --dataset-root "$DATASET_ROOT" \
  --train-repo-ids place_phone_stand \
  --model-path "$BASE_PI05" \
  --output-dir "$OUTPUT_ROOT/pi05_place_phone_stand_vla_lora" \
  --assets-path "$ROBOTWIN_ASSETS" \
  --batch-size 64 \
  --train-steps 10000 \
  --save-every 500 \
  --save-step-checkpoints \
  --lora-rank 32 \
  --eval-every 0 \
  --wandb-enabled
```

### 4. 单独评估 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_robotwin_policy.py \
  --assets-path "$ROBOTWIN_ASSETS" \
  --model-path "$OUTPUT_ROOT/pi05_place_phone_stand_vla_fullft/checkpoints/step_0010000" \
  --num-envs 30 \
  --eval-step-limit 400 \
  --eval-rollout-epochs 1 \
  --record-progress \
  --device cuda \
  --output-json "$OUTPUT_ROOT/evals/fullft_step10000_eval30.json"
```

固定 eval seeds：

```bash
python scripts/eval_robotwin_policy.py \
  --assets-path "$ROBOTWIN_ASSETS" \
  --model-path /path/to/checkpoint \
  --num-envs 30 \
  --eval-step-limit 400 \
  --eval-seeds-path /path/to/eval_seeds.json \
  --record-progress \
  --output-json /path/to/eval_result.json
```

### 5. 跑本地 RL fine-tune

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_robotwin_local_rl.py \
  --assets-path "$ROBOTWIN_ASSETS" \
  --model-path "$OUTPUT_ROOT/pi05_place_phone_stand_vla_fullft/checkpoints/step_0010000" \
  --output-dir "$OUTPUT_ROOT/pi05_place_phone_stand_local_rl" \
  --total-steps 20000 \
  --total-num-envs 8 \
  --rollout-epoch 4 \
  --max-steps-per-rollout-epoch 200 \
  --global-batch-size 32 \
  --micro-batch-size 4 \
  --update-epoch 2 \
  --entropy-bonus 0.02 \
  --save-every 500 \
  --eval-every 100 \
  --eval-num-envs 20 \
  --wandb-enabled
```

## 快速查看参数

VLA SFT：

```bash
python scripts/train_robotwin_vla.py --help
```

LoRA SFT：

```bash
python scripts/train_robotwin_vla_lora.py --help
```

本地 RL：

```bash
python scripts/train_robotwin_local_rl.py --help
```

单独 eval：

```bash
python scripts/eval_robotwin_policy.py --help
```

计算 norm stats：

```bash
python scripts/compute_robotwin_norm_stats.py --help
```

预设 shell 脚本：

- `scripts/run_robotwin_lora_rl_from_sft13k.sh`：从 SFT checkpoint 启动 LoRA RL 的 preset。
- `scripts/trigger_continue_full_sft_after_10k.sh`：从 10k step checkpoint 继续 full SFT 的 preset。

## 当前主要实验路径

当前仓库主要围绕这一条任务路径：

- 任务：`RoboTwin place_phone_stand`
- 模型：`pi0.5 / OpenPI`
- SFT 训练：`src/rlinf/projects/robotwin/training/sft/`
- 本地 RL 训练：`src/rlinf/projects/robotwin/training/local_rl/`
- RLinf embodied 配置：`src/rlinf/projects/robotwin/configs/embodiment/`
- RoboTwin env wrapper：`src/rlinf/envs/robotwin/`
- OpenPI action model：`src/rlinf/models/embodiment/openpi/`

## 第三方代码

这个仓库 vendored 了一部分上游 RLinf 和 OpenPI 代码。来源和 license 说明见：

- `THIRD_PARTY.md`
- `THIRD_PARTY_LICENSES/`
