# RLinf RoboTwin

这个仓库现在按上游 RLinf 的目录风格组织：核心框架代码放在 `src/rlinf/`，RoboTwin/pi0.5 相关的项目 glue code 放在 `src/rlinf/projects/robotwin/`。

`bentele` 不再作为 Python package 路径使用。后续项目代码 import 应该走：

```python
import rlinf.projects.robotwin
```

注意：部分历史实验配置里仍然保留 `wandb_project=bentele`、本地输出目录名里带 `bentele`。这些是为了兼容之前实验和对比结果，不代表 Python 包名还是 `bentele`。

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
        cli/                    # 可 import 的命令行入口
        configs/                # RoboTwin 实验 Hydra 配置
        integration/            # project config 到 RLinf runner 的桥接层
        robotwin/               # dataset、eval、norm stats、probe
        runtime/                # path、env var、logging、seed、device 工具
        training/
          sft/                  # VLA SFT 训练代码
          local_rl/             # 本地 RL fine-tune 代码
        utils/                  # 兼容旧 eval API 的项目工具
    runners/                    # RLinf runner
    scheduler/                  # cluster、placement、worker scheduling
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

## 主要入口

旧的 `scripts/` 路径继续保留，已有 shell 调用不需要改：

```bash
python scripts/train_robotwin_vla.py --help
torchrun --nproc_per_node=2 scripts/train_robotwin_vla.py --help
python scripts/train_robotwin_vla_lora.py --help
python scripts/train_robotwin_local_rl.py --help
python scripts/eval_robotwin_policy.py --help
python scripts/compute_robotwin_norm_stats.py --help
```

对应的新 module 入口是：

```bash
PYTHONPATH=src python -m rlinf.projects.robotwin.cli.train_robotwin_vla --help
PYTHONPATH=src python -m rlinf.projects.robotwin.cli.train_robotwin_local_rl --help
PYTHONPATH=src python -m rlinf.projects.robotwin.cli.eval_robotwin_policy --help
PYTHONPATH=src python -m rlinf.projects.robotwin.cli.compute_robotwin_norm_stats --help
PYTHONPATH=src python -m rlinf.projects.robotwin.cli.train_embodied --help
```

## RoboTwin 项目模块

- `cli/`：薄入口，只负责解析 CLI，然后调用训练或评测逻辑。
- `configs/embodiment/`：RoboTwin `place_phone_stand` 的 Hydra 配置。
- `integration/rlinf_embodied.py`：校验 Hydra config，并启动 RLinf embodied runner。
- `robotwin/dataset.py`：本地 RoboTwin V3 数据集 adapter。
- `robotwin/eval.py` 和 `utils/robotwin_eval.py`：policy eval 逻辑和旧 API 兼容层。
- `robotwin/norm_stats.py`：state/action normalization stats 计算。
- `robotwin/probes.py`：固定样本的 rollout-style action probe。
- `training/sft/`：VLA supervised fine-tuning，包括 args、data、model、FSDP、optim、checkpoint、eval、trainer。
- `training/local_rl/`：本地 RL fine-tuning，包括 rollout、batch、optim、metrics、checkpoint、eval、runner。
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

## 常用工作流

SFT 训练：

```bash
python scripts/train_robotwin_vla.py --help
```

LoRA SFT 训练：

```bash
python scripts/train_robotwin_vla_lora.py --help
```

本地 RL fine-tune：

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

排查 pipeline：

```bash
python scripts/trace_robotwin_pipeline.py --help
```

匹配 RoboTwin env reset seed 和 dataset episode：

```bash
python scripts/match_robotwin_episode_seeds.py --help
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
