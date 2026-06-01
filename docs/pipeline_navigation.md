# RoboTwin / pi0.5 Pipeline Navigation

这份文档对应当前 `main` 分支的真实代码结构。当前仓库的主线不是 `bentele` 包，也不是旧的 ManiSkill smoke path，而是：

```text
scripts/*
  -> src/rlinf/projects/robotwin/cli/*
  -> src/rlinf/projects/robotwin/adapters/*
  -> src/rlinf/training/*
  -> src/rlinf/models/envs/algorithms/*
```

当前最重要的任务链路是：

- 任务：`RoboTwin place_phone_stand`
- 模型：`pi0.5 / OpenPI`
- 训练一：VLA SFT / LoRA SFT
- 训练二：从 SFT checkpoint 继续做本地 RL fine-tune
- 评估：单独 eval checkpoint 或训练中周期 eval

README 里的流程图对应：

```text
docs/rlinf_finetune_pipeline.svg
```

---

## 1. 先看哪些文件

如果你只想快速理解当前 pipeline，按这个顺序看：

| 目的 | 文件 |
|---|---|
| 看整体使用方式 | `README.md` |
| 看流程图 | `docs/rlinf_finetune_pipeline.svg` |
| 跑 full VLA SFT | `scripts/train_robotwin_vla.py` |
| 跑 LoRA SFT | `scripts/train_robotwin_vla_lora.py` |
| 跑本地 RL fine-tune | `scripts/train_robotwin_local_rl.py` |
| 单独评估 checkpoint | `scripts/eval_robotwin_policy.py` |
| 计算 norm stats | `scripts/compute_robotwin_norm_stats.py` |
| RoboTwin 项目适配层 | `src/rlinf/projects/robotwin/adapters/` |
| 通用 SFT 训练循环 | `src/rlinf/training/sft/trainer.py` |
| 通用 RL batch/update 工具 | `src/rlinf/training/rl/` |
| PPO / GAE / loss 逻辑 | `src/rlinf/algorithms/` |
| RoboTwin env wrapper | `src/rlinf/envs/robotwin/robotwin_env.py` |
| OpenPI/pi0.5 action model | `src/rlinf/models/embodiment/openpi/openpi_action_model.py` |

---

## 2. 当前目录怎么分层

当前仓库大体分成三层：

```text
src/rlinf/projects/robotwin/
  项目层：RoboTwin 的 CLI、adapter、config、dataset、eval、runtime glue。

src/rlinf/
  RLinf 引擎层：algorithms、training、envs、models、workers、scheduler、utils。

src/openpi/ + src/openpi_client/
  vendored OpenPI 层：pi0.5 模型、transform、checkpoint、client helper。
```

也就是说：

- `projects/robotwin` 负责“这个项目怎么接进来”。
- `training` 和 `algorithms` 负责“训练循环和算法怎么算”。
- `models/embodiment/openpi` 负责“OpenPI/pi0.5 怎么包装成 RLinf policy”。
- `envs/robotwin` 负责“RoboTwin 环境怎么 reset/step/chunk_step”。

---

## 3. VLA SFT / LoRA SFT 怎么流

### 3.1 入口

```text
scripts/train_robotwin_vla.py
scripts/train_robotwin_vla_lora.py
```

这两个脚本本身很薄，只负责把 `src/` 放进 `PYTHONPATH`，然后调用：

```text
src/rlinf/projects/robotwin/cli/train_robotwin_vla.py
src/rlinf/projects/robotwin/cli/train_robotwin_vla_lora.py
```

### 3.2 CLI 层

SFT CLI 会做：

```text
tyro.cli(Args)
run_sft_training(args, RobotwinSFTTaskSpec())
```

LoRA CLI 继承同一个 `Args`，但默认：

```text
is_lora = True
lora_rank = 32
wandb_tags = ("lora",)
```

对应文件：

```text
src/rlinf/projects/robotwin/cli/train_robotwin_vla.py
src/rlinf/projects/robotwin/cli/train_robotwin_vla_lora.py
src/rlinf/projects/robotwin/training/sft/args.py
```

### 3.3 项目 adapter 层

核心文件：

```text
src/rlinf/projects/robotwin/adapters/sft_task.py
src/rlinf/projects/robotwin/adapters/_sft_impl.py
```

`RobotwinSFTTaskSpec` 把 RoboTwin 项目接到通用 SFT 训练器里。它主要提供这些 hook：

```text
init_runtime
build_dataloader
build_model
build_optimizer
move_batch_to_device
forward_loss
save_checkpoint
run_eval
```

真正的细节放在 `_sft_impl.py`：

- `_build_loader(...)`：读取 RoboTwin V3 本地数据集，接 OpenPI dataconfig 和 transform。
- `_build_model(...)`：加载 pi0.5/OpenPI checkpoint，必要时应用 LoRA。
- `_apply_openpi_lora(...)`：把 LoRA 注入 PaliGemma VLM 子树。
- `_save_checkpoint(...)`：保存训练产物。
- `_run_periodic_eval(...)`：训练中周期评估。

### 3.4 通用 SFT trainer

核心文件：

```text
src/rlinf/training/sft/trainer.py
```

主循环大概是：

```text
init runtime
-> init distributed
-> build dataloader
-> build model
-> build optimizer
-> for step in train_steps:
     batch = next(data_loader)
     loss = task.forward_loss(...)
     backward
     optimizer step / grad accumulation
     log / save / eval
```

SFT 的核心目标可以理解成行为克隆：

```text
L_SFT = mean(loss(model(observation), target_actions))
```

---

## 4. 本地 RL fine-tune 怎么流

### 4.1 入口

```text
scripts/train_robotwin_local_rl.py
```

薄入口继续调用：

```text
src/rlinf/projects/robotwin/cli/train_robotwin_local_rl.py
```

CLI 会解析参数，然后走：

```text
run_rl_training(args, RobotwinRLTaskSpec())
```

对应文件：

```text
src/rlinf/projects/robotwin/cli/train_robotwin_local_rl.py
src/rlinf/projects/robotwin/adapters/rl_task.py
src/rlinf/training/rl/runner.py
```

### 4.2 项目 RL 实现

核心文件：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py
```

本地 RL 不是走完整多 worker 分布式链路，而是一个轻量本地闭环：

```text
compose_cfg(args)
-> get_model(cfg.actor.model)
-> RoboTwinEnv(...)
-> for step in total_steps:
     rollout_batch, env_metrics = collect_rollout(...)
     rollout_batch = prepare_rollout_batch(...)
     train_metrics = run_update(...)
     log / save / eval
```

### 4.3 rollout 收集

核心函数：

```text
collect_rollout(...)
env_interact_step(...)
```

它做的事情是：

1. reset RoboTwin 环境。
2. 用当前 policy 预测 action chunk。
3. 记录 `prev_logprobs`、`prev_values`、`forward_inputs`。
4. 把 action chunk 送进环境。
5. 收集 reward、done、termination、truncation。
6. 结尾额外算一次 bootstrap value。
7. 转成训练 batch。

相关文件：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py
src/rlinf/data/embodied_io_struct.py
src/rlinf/envs/action_utils.py
src/rlinf/envs/robotwin/robotwin_env.py
```

### 4.4 advantage / return

核心文件：

```text
src/rlinf/training/rl/batch.py
src/rlinf/algorithms/registry.py
src/rlinf/algorithms/advantages.py
src/rlinf/algorithms/utils.py
```

关键入口：

```text
prepare_rollout_batch(...)
calculate_adv_and_returns(...)
```

它会做：

- 重新整理 rollout epoch 维度。
- 根据 `dones` 构造 `loss_mask`。
- 调 registry 分发到 GAE / GRPO / raw 等 advantage 实现。
- 对 embodied batch 做 shape adapter。

常见 GAE 逻辑：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = sum_l (gamma * lambda)^l * delta_{t+l}
```

### 4.5 policy update

核心文件：

```text
src/rlinf/training/rl/update.py
src/rlinf/algorithms/losses.py
src/rlinf/training/rl/optim.py
```

关键入口：

```text
run_update(...)
policy_loss(...)
```

它会做：

- shuffle rollout batch。
- 切 global batch / micro batch。
- 用 `forward_inputs` 重新前向，计算当前策略 logprob/value。
- 用旧 logprob 和当前 logprob 算 PPO ratio。
- 计算 clipped policy loss / value loss / entropy bonus。
- backward、gradient accumulation、clip grad、optimizer step。

PPO clipped objective 的核心形态：

```text
r_t = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))
L_clip = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
```

---

## 5. eval 怎么流

### 5.1 单独 eval

入口：

```text
scripts/eval_robotwin_policy.py
src/rlinf/projects/robotwin/cli/eval_robotwin_policy.py
```

主要逻辑：

```text
src/rlinf/projects/robotwin/robotwin/eval.py
src/rlinf/projects/robotwin/utils/robotwin_eval.py
```

它会加载 checkpoint，创建 RoboTwin env，执行 policy rollout，最后保存 JSON 结果。

### 5.2 SFT/RL 训练中 eval

SFT 中周期 eval：

```text
src/rlinf/projects/robotwin/adapters/sft_task.py::run_eval
src/rlinf/projects/robotwin/adapters/_sft_impl.py::_run_periodic_eval
```

本地 RL 中周期 eval：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py::evaluate_current_policy
```

---

## 6. norm stats 怎么流

入口：

```text
scripts/compute_robotwin_norm_stats.py
src/rlinf/projects/robotwin/cli/compute_robotwin_norm_stats.py
```

作用：

- 从 RoboTwin dataset 统计 state/action normalization stats。
- 写入 pi0.5/OpenPI checkpoint 目录。
- 后续 SFT / eval / RL 都会通过 OpenPI dataconfig 和 checkpoint loader 使用这些 stats。

相关文件：

```text
src/rlinf/projects/robotwin/robotwin/norm_stats.py
src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py
src/openpi/shared/normalize.py
```

---

## 7. 最容易读错的地方

### 7.1 `src/rlinf/projects/robotwin/training/sft/` 不是训练主循环

这里主要是兼容旧 import 的 wrapper / args。真正通用 SFT 主循环在：

```text
src/rlinf/training/sft/trainer.py
```

项目细节在：

```text
src/rlinf/projects/robotwin/adapters/sft_task.py
src/rlinf/projects/robotwin/adapters/_sft_impl.py
```

### 7.2 `training/local_rl/` 也不是核心算法实现

本地 RL 的项目入口在：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py
```

通用 batch/update/optimizer 在：

```text
src/rlinf/training/rl/
```

算法 loss / advantage 在：

```text
src/rlinf/algorithms/
```

### 7.3 `scripts/` 是用户入口，不是全部逻辑

`scripts/*.py` 通常只是薄入口。看完脚本后，要继续跳到：

```text
src/rlinf/projects/robotwin/cli/
src/rlinf/projects/robotwin/adapters/
```

---

## 8. 建议读代码路线

第一次读代码，按这个路线走：

```text
README.md
-> docs/rlinf_finetune_pipeline.svg
-> scripts/train_robotwin_vla.py
-> src/rlinf/projects/robotwin/cli/train_robotwin_vla.py
-> src/rlinf/projects/robotwin/adapters/sft_task.py
-> src/rlinf/projects/robotwin/adapters/_sft_impl.py
-> src/rlinf/training/sft/trainer.py
```

然后看 RL：

```text
scripts/train_robotwin_local_rl.py
-> src/rlinf/projects/robotwin/cli/train_robotwin_local_rl.py
-> src/rlinf/projects/robotwin/adapters/rl_task.py
-> src/rlinf/projects/robotwin/adapters/_rl_impl.py
-> src/rlinf/training/rl/batch.py
-> src/rlinf/training/rl/update.py
-> src/rlinf/algorithms/losses.py
-> src/rlinf/algorithms/advantages.py
```

最后补环境和模型：

```text
src/rlinf/envs/robotwin/robotwin_env.py
src/rlinf/envs/action_utils.py
src/rlinf/models/embodiment/openpi/openpi_action_model.py
src/openpi/models_pytorch/pi0_pytorch.py
```
