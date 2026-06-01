# RoboTwin / pi0.5 Pipeline Details

这份文档是 `docs/pipeline_navigation.md` 的详细版，对应当前 `main` 分支。

当前主线：

```text
RoboTwin place_phone_stand
+ pi0.5 / OpenPI
+ VLA SFT / LoRA SFT
+ local RL fine-tune
+ checkpoint eval
```

如果只想快速定位文件，先看：

```text
docs/pipeline_navigation.md
```

---

## 1. 当前代码分层

### 1.1 项目层：`src/rlinf/projects/robotwin/`

这一层放的是 RoboTwin 项目自己的 glue code：

```text
src/rlinf/projects/robotwin/
  adapters/               # 把 RoboTwin 接到通用 SFT/RL trainer
  cli/                    # 可 import 的命令行入口
  configs/                # Hydra 配置
  data/                   # 本地 RoboTwin dataset adapter
  robotwin/               # eval / norm_stats / probe 等项目逻辑
  runtime/                # path / env var / seed / device / logging 工具
  training/               # 兼容旧 import 的 args / wrapper
  utils/                  # 兼容旧 eval API 的工具
```

这里最重要的是：

```text
adapters/sft_task.py
adapters/_sft_impl.py
adapters/rl_task.py
adapters/_rl_impl.py
```

### 1.2 通用训练层：`src/rlinf/training/`

这一层是和具体项目相对解耦的训练工具：

```text
src/rlinf/training/sft/trainer.py
src/rlinf/training/rl/batch.py
src/rlinf/training/rl/update.py
src/rlinf/training/rl/optim.py
src/rlinf/training/rl/metrics.py
src/rlinf/training/common/
```

它不应该知道 RoboTwin 的细节，而是通过 project adapter 拿到 model、batch、loss、eval hook。

### 1.3 算法层：`src/rlinf/algorithms/`

这里放 PPO/GRPO/GAE/loss 等算法实现：

```text
src/rlinf/algorithms/advantages.py
src/rlinf/algorithms/losses.py
src/rlinf/algorithms/registry.py
src/rlinf/algorithms/utils.py
src/rlinf/algorithms/loss_scales.py
```

### 1.4 环境和模型层

环境：

```text
src/rlinf/envs/robotwin/robotwin_env.py
src/rlinf/envs/action_utils.py
```

模型：

```text
src/rlinf/models/embodiment/openpi/openpi_action_model.py
src/rlinf/models/embodiment/openpi/dataconfig/
src/openpi/models_pytorch/pi0_pytorch.py
```

---

## 2. VLA SFT 详细流程

### 2.1 命令入口

用户一般从这里启动：

```bash
python scripts/train_robotwin_vla.py ...
```

这个脚本是薄入口，作用是：

1. 把仓库的 `src/` 加进 `sys.path`。
2. 调用 `src/rlinf/projects/robotwin/cli/train_robotwin_vla.py::main()`。

### 2.2 CLI 解析参数

文件：

```text
src/rlinf/projects/robotwin/cli/train_robotwin_vla.py
```

核心逻辑：

```python
def main() -> None:
    args = tyro.cli(Args)
    run_sft_training(args, RobotwinSFTTaskSpec())
```

这里产生两个东西：

- `args`：从 CLI 得到的训练参数。
- `RobotwinSFTTaskSpec()`：RoboTwin 项目适配器。

### 2.3 SFT adapter 做什么

文件：

```text
src/rlinf/projects/robotwin/adapters/sft_task.py
```

`RobotwinSFTTaskSpec` 是通用 SFT trainer 和 RoboTwin 项目之间的接口层。它提供：

```text
init_runtime
build_dataloader
build_model
on_after_dataloader_built
on_after_model_built
build_optimizer
move_batch_to_device
forward_loss
on_step_end
save_checkpoint
should_eval
export_eval_model_dir
run_eval
```

也就是说，通用 trainer 不需要知道 RoboTwin 数据怎么读、pi0.5 怎么加载、eval 怎么跑；这些都由 adapter 提供。

### 2.4 dataset 怎么读

真正实现：

```text
src/rlinf/projects/robotwin/adapters/_sft_impl.py::_build_loader
src/rlinf/projects/robotwin/data/robotwin_v3_dataset.py
```

流程：

```text
dataset_root / repo_id
-> 读取 LeRobot/RoboTwin parquet
-> 读取 video frame
-> 构造 observation.state / action / prompt / camera images
-> openpi_data.transform_dataset(raw_dataset, data_config)
-> TorchDataLoader
-> DataLoaderImpl
```

这里关键是 OpenPI dataconfig：

```text
src/rlinf/models/embodiment/openpi/dataconfig/__init__.py
src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py
```

它决定：

- 图像字段怎么映射。
- state/action 怎么 padding 或变换。
- norm stats 怎么使用。
- OpenPI model transform 怎么接入。

### 2.5 model 怎么建

实现位置：

```text
src/rlinf/projects/robotwin/adapters/_sft_impl.py::_build_model
```

非 FSDP 单卡路径通常走：

```text
rlinf.models.embodiment.openpi.get_model(cfg)
```

FSDP 路径会走：

```text
_build_openpi_model_for_fsdp(...)
_build_fsdp_model(...)
```

模型构建时会处理：

1. 读取 `model_path`。
2. 构造 OpenPI/pi0.5 action model config。
3. 加载 checkpoint 权重。
4. 加载 norm stats。
5. 挂 input/output transforms。
6. 如果 `is_lora=True`，应用 LoRA。

### 2.6 LoRA 怎么接进来

LoRA 入口：

```text
scripts/train_robotwin_vla_lora.py
src/rlinf/projects/robotwin/cli/train_robotwin_vla_lora.py
```

LoRA CLI 继承 SFT 参数，但默认：

```text
is_lora = True
lora_rank = 32
```

真正注入 LoRA 的函数：

```text
src/rlinf/projects/robotwin/adapters/_sft_impl.py::_apply_openpi_lora
```

它会在 PaliGemma VLM 子树上应用 PEFT LoRA。简化理解：

```text
base pi0.5 checkpoint
-> 找到 model.paligemma_with_expert.paligemma
-> 注入 LoRA adapter
-> 只训练 LoRA/可训练参数
```

### 2.7 SFT loss 怎么算

adapter 层：

```text
src/rlinf/projects/robotwin/adapters/sft_task.py::forward_loss
```

通用 trainer：

```text
src/rlinf/training/sft/trainer.py
```

核心形态：

```text
observation, actions = batch
losses = model(data={"observation": observation, "actions": actions}, forward_type=ForwardType.SFT)
loss = mean(losses)
```

可以把它理解成行为克隆：

```text
L_SFT = mean_i loss(f_theta(o_i), a_i)
```

### 2.8 SFT trainer 主循环

文件：

```text
src/rlinf/training/sft/trainer.py
```

主循环：

```text
init_runtime
init_distributed
seed_everything
device_for_training
build_dataloader
build_model
build_optimizer
for step in train_steps:
    batch = next(loader)
    batch = move_batch_to_device(...)
    loss = forward_loss(...)
    loss / grad_accum_steps backward
    if should_step:
        set lr
        clip grad
        optimizer.step
    log
    save checkpoint
    optional eval
shutdown
```

---

## 3. 本地 RL fine-tune 详细流程

### 3.1 命令入口

用户一般从这里启动：

```bash
python scripts/train_robotwin_local_rl.py ...
```

薄入口调用：

```text
src/rlinf/projects/robotwin/cli/train_robotwin_local_rl.py::main()
```

CLI 进一步调用：

```text
run_rl_training(args, RobotwinRLTaskSpec())
```

### 3.2 RL adapter

文件：

```text
src/rlinf/projects/robotwin/adapters/rl_task.py
```

这个 adapter 很薄：

```text
RobotwinRLTaskSpec.run_rl_training(args)
-> src/rlinf/projects/robotwin/adapters/_rl_impl.py::main(args)
```

所以本地 RL 的主要逻辑在：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py
```

### 3.3 配置怎么组合

函数：

```text
compose_cfg(args)
```

作用：

- 读取 Hydra config。
- 用 CLI 参数覆盖 env/model/algorithm 设置。
- 写入：
  - `env.train.total_num_envs`
  - `actor.model.model_path`
  - `algorithm.rollout_epoch`
  - `algorithm.update_epoch`
  - `algorithm.entropy_bonus`
  - `actor.global_batch_size`
  - `actor.micro_batch_size`

最终得到 `cfg`，用于创建模型、环境和训练参数。

### 3.4 模型和环境怎么启动

模型：

```text
model = get_model(cfg.actor.model).to(device)
```

相关文件：

```text
src/rlinf/models/__init__.py
src/rlinf/models/embodiment/openpi/__init__.py
src/rlinf/models/embodiment/openpi/openpi_action_model.py
```

环境：

```text
RoboTwinEnv(cfg=cfg.env.train, ...)
```

相关文件：

```text
src/rlinf/envs/robotwin/robotwin_env.py
```

环境 wrapper 负责：

- 创建 RoboTwin VectorEnv。
- reset / step / chunk_step。
- 提取图像、wrist image、state、task description。
- 返回 reward、done、termination、truncation。

### 3.5 rollout 怎么收集

函数：

```text
collect_rollout(...)
```

流程：

```text
for rollout_epoch:
    env.reset()
    for chunk_step:
        model.predict_action_batch(...)
        保存 prev_logprobs / prev_values / forward_inputs
        env_interact_step(...)
        保存 reward / done / termination / truncation
    对最后 obs 再算一次 bootstrap value
convert_trajectories_to_batch
```

这里的几个字段非常重要：

| 字段 | 作用 |
|---|---|
| `actions` | 当前策略采样出的 action chunk |
| `prev_logprobs` | rollout 时旧策略的 logprob，PPO ratio 要用 |
| `prev_values` | rollout 时 critic value，GAE 要用 |
| `forward_inputs` | update 阶段重算当前策略 logprob/value 的输入 |
| `rewards` | 环境返回 reward |
| `dones` | episode 是否结束 |
| `terminations` | 任务成功/失败等终止 |
| `truncations` | 超时截断 |

数据容器在：

```text
src/rlinf/data/embodied_io_struct.py
```

### 3.6 action 怎么送进环境

函数：

```text
env_interact_step(...)
```

关键调用：

```text
prepare_actions(...)
env.chunk_step(chunk_actions)
```

相关文件：

```text
src/rlinf/envs/action_utils.py
src/rlinf/envs/robotwin/robotwin_env.py
```

`prepare_actions` 负责把模型输出的 raw action chunk 变成环境可执行的 action 格式。

### 3.7 rollout batch 怎么准备

函数：

```text
prepare_rollout_batch(cfg, rollout_batch)
```

文件：

```text
src/rlinf/training/rl/batch.py
```

它主要做：

1. `process_nested_dict_for_adv`：按 rollout epoch 重新整理时间 / batch 维。
2. `compute_loss_mask`：根据 done 生成有效训练位置。
3. `calculate_adv_and_returns`：通过 registry 调具体 advantage 算法。
4. 把 `advantages` / `returns` 写回 rollout batch。

### 3.8 GAE / advantage 怎么算

文件：

```text
src/rlinf/algorithms/registry.py
src/rlinf/algorithms/advantages.py
src/rlinf/algorithms/utils.py
```

分发入口：

```text
calculate_adv_and_returns(...)
```

常见 GAE 形式：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
return_t = A_t + V(s_t)
```

注意：embodied batch 需要在 `utils.py` 里做 shape 转换，典型转换是：

```text
[num_chunk, batch, chunk_size]
-> [num_chunk * chunk_size, batch]
```

### 3.9 PPO update 怎么跑

函数：

```text
run_update(...)
```

文件：

```text
src/rlinf/training/rl/update.py
```

流程：

```text
shuffle rollout batch
split global batch
split micro batch
for micro batch:
    model(forward_inputs, compute_logprobs=True, compute_values=True)
    policy_loss(...)
    entropy bonus
    backward with gradient accumulation
clip grad
optimizer.step
```

loss 分发在：

```text
src/rlinf/algorithms/registry.py::policy_loss
```

具体 PPO loss 在：

```text
src/rlinf/algorithms/losses.py
```

核心 ratio：

```text
r_t = exp(logprob_new - logprob_old)
```

clip objective：

```text
L_clip = E[min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)]
```

实际代码里通常用负号把最大化目标写成最小化 loss。

### 3.10 本地 RL 保存和评估

保存：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py::save_checkpoint
```

产物：

```text
output_dir/model_state_dict/full_weights.pt
output_dir/training_state.pt
output_dir/sft_metadata.json
output_dir/local_rl_summary.json
output_dir/history.jsonl
```

评估：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py::evaluate_current_policy
```

它会新建 eval env，用当前 policy 跑 rollout，并保存：

```text
output_dir/evals/step_xxxxxxx.json
```

---

## 4. 单独 eval 详细流程

入口：

```bash
python scripts/eval_robotwin_policy.py ...
```

薄入口调用：

```text
src/rlinf/projects/robotwin/cli/eval_robotwin_policy.py
```

核心 eval 逻辑：

```text
src/rlinf/projects/robotwin/robotwin/eval.py
src/rlinf/projects/robotwin/utils/robotwin_eval.py
```

主要流程：

```text
load model checkpoint
create RoboTwinEnv
env.reset
while not done and not exceed step_limit:
    policy.predict_action_batch
    env.chunk_step
collect metrics
save json
```

常看指标：

- `success_at_end`
- `success_once`
- `return`
- `reward`
- `episode_len`
- `num_trajectories`

---

## 5. norm stats 详细流程

入口：

```bash
python scripts/compute_robotwin_norm_stats.py ...
```

薄入口调用：

```text
src/rlinf/projects/robotwin/cli/compute_robotwin_norm_stats.py
```

核心逻辑：

```text
src/rlinf/projects/robotwin/robotwin/norm_stats.py
```

它读取 RoboTwin dataset，统计模型需要的 normalization statistics，并写到 base checkpoint 目录。

这些 stats 后续会被：

```text
src/rlinf/projects/robotwin/adapters/_sft_impl.py
src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py
src/openpi/shared/normalize.py
```

共同使用。

---

## 6. 文件职责总表

| 文件 | 作用 |
|---|---|
| `scripts/train_robotwin_vla.py` | full VLA SFT 用户入口 |
| `scripts/train_robotwin_vla_lora.py` | LoRA SFT 用户入口 |
| `scripts/train_robotwin_local_rl.py` | 本地 RL fine-tune 用户入口 |
| `scripts/eval_robotwin_policy.py` | checkpoint 单独评估入口 |
| `scripts/compute_robotwin_norm_stats.py` | norm stats 计算入口 |
| `src/rlinf/projects/robotwin/cli/*` | CLI 参数解析和任务入口 |
| `src/rlinf/projects/robotwin/adapters/sft_task.py` | SFT project adapter |
| `src/rlinf/projects/robotwin/adapters/_sft_impl.py` | RoboTwin SFT 具体实现 |
| `src/rlinf/projects/robotwin/adapters/rl_task.py` | RL project adapter |
| `src/rlinf/projects/robotwin/adapters/_rl_impl.py` | 本地 RL 具体实现 |
| `src/rlinf/training/sft/trainer.py` | 通用 SFT 主循环 |
| `src/rlinf/training/rl/batch.py` | rollout batch 预处理和 advantage 入口 |
| `src/rlinf/training/rl/update.py` | 本地 RL update loop |
| `src/rlinf/training/rl/optim.py` | actor / value head optimizer 分组 |
| `src/rlinf/training/rl/metrics.py` | RL 日志和 history.jsonl |
| `src/rlinf/algorithms/advantages.py` | GAE / GRPO / raw advantage |
| `src/rlinf/algorithms/losses.py` | PPO actor/value loss |
| `src/rlinf/algorithms/registry.py` | loss / advantage registry |
| `src/rlinf/envs/robotwin/robotwin_env.py` | RoboTwin env wrapper |
| `src/rlinf/envs/action_utils.py` | 模型 action 到环境 action 的适配 |
| `src/rlinf/data/embodied_io_struct.py` | EnvOutput / RolloutResult / Trajectory 等数据容器 |
| `src/rlinf/models/embodiment/openpi/openpi_action_model.py` | OpenPI/pi0.5 policy wrapper |
| `src/openpi/models_pytorch/pi0_pytorch.py` | pi0.5 PyTorch 模型主体 |

---

## 7. 建议 debug 路线

### 7.1 SFT loss 异常

优先看：

```text
src/rlinf/projects/robotwin/adapters/_sft_impl.py::_build_loader
src/rlinf/projects/robotwin/adapters/sft_task.py::forward_loss
src/rlinf/training/sft/trainer.py
```

重点检查：

- dataset 样本 shape。
- action horizon 是否和模型一致。
- norm stats 是否存在。
- observation / action 是否正确 move 到 device。

### 7.2 RL reward / done 异常

优先看：

```text
src/rlinf/projects/robotwin/adapters/_rl_impl.py::env_interact_step
src/rlinf/envs/robotwin/robotwin_env.py
```

重点检查：

- `chunk_rewards`
- `chunk_terminations`
- `chunk_truncations`
- `infos["episode"]`
- `final_observation`

### 7.3 PPO loss 异常

优先看：

```text
src/rlinf/training/rl/batch.py
src/rlinf/training/rl/update.py
src/rlinf/algorithms/losses.py
```

重点检查：

- `prev_logprobs` 是否来自 rollout 时旧策略。
- `logprobs` 是否来自 update 时当前策略。
- `advantages` shape 是否和 `loss_mask` 对齐。
- `loss_mask_sum` 是否正确。
- `micro_batch_size` 是否整除 `global_batch_size`。

### 7.4 eval 结果异常

优先看：

```text
scripts/eval_robotwin_policy.py
src/rlinf/projects/robotwin/robotwin/eval.py
src/rlinf/projects/robotwin/utils/robotwin_eval.py
```

重点检查：

- eval config name。
- eval seeds。
- `eval_step_limit`。
- action execute horizon。
- checkpoint 是否包含 norm stats / config / assets。
