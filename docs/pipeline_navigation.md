# RoboTwin Phone Pipeline 导航图

这份文档只讲当前主线：

- 任务：`RoboTwin place_phone_stand`
- 模型：`pi0.5 / openpi`
- 算法：`PPO`

目标不是把每一步讲得很细，而是回答两个问题：

1. 整个 pipeline 大概怎么流。
2. 如果想看某块细节，应该去哪个文件、哪个函数。

---

## 1. 一句话总览

主线是：

`Hydra 配置 -> bentele 入口 -> rlinf bridge -> runner -> env/rollout/actor worker -> env 产 obs -> rollout 用 pi0.5 出 action -> env 执行动作产 trajectory -> actor 算 PPO update -> 新权重同步回 rollout`

如果你只想先记住最重要的几个文件：

- 训练入口：
  `src/bentele/cli/train_embodied.py`
- 配置入口：
  `src/bentele/configs/embodiment/robotwin_place_phone_stand_ppo_openpi_pi05.yaml`
- bridge：
  `src/bentele/integration/rlinf_embodied.py`
- 主循环：
  `src/rlinf/runners/embodied_runner.py`
- 环境：
  `src/rlinf/envs/robotwin/robotwin_env.py`
- rollout 推理：
  `src/rlinf/workers/rollout/hf/huggingface_worker.py`
- pi0.5 policy：
  `src/rlinf/models/embodiment/openpi/openpi_action_model.py`
- actor 训练：
  `src/rlinf/workers/actor/fsdp_actor_worker.py`

---

## 2. 从哪里开始

### 2.1 配置从哪进来

先看：

- `src/bentele/configs/embodiment/robotwin_place_phone_stand_ppo_openpi_pi05.yaml`
- `src/bentele/configs/embodiment/env/robotwin_place_phone_stand.yaml`
- `src/bentele/configs/embodiment/model/pi0_5.yaml`

这里主要决定：

- 任务名
- env 参数
- pi0.5 模型参数
- PPO 超参数
- rollout / actor / eval 设置

如果你想知道“这次跑的到底是什么任务、多少 env、什么 checkpoint、什么 PPO 参数”，先看这几份。

### 2.2 训练从哪启动

看：

- `src/bentele/cli/train_embodied.py`
  - `main(...)`

它做的事很简单：

- Hydra 读配置
- 把配置交给 bridge

如果你想知道“程序是从哪里起的”，就从这里开始。

---

## 3. 配置怎么变成 worker 和 runner

看：

- `src/bentele/integration/rlinf_embodied.py`
  - `run_sync_embodied_training(...)`
  - `_create_sync_actor_group(...)`

这一层的职责是：

- 校验配置
- 创建 cluster / placement
- 创建 actor group
- 创建 rollout group
- 创建 env group
- 创建 runner

如果你想知道“哪些 worker 被建起来了、同步训练到底起了哪些角色”，看这个文件。

再往下看：

- `src/rlinf/runners/embodied_runner.py`
  - `__init__(...)`
  - `init_workers(...)`
  - `run(...)`
  - `update_rollout_weights(...)`
  - `evaluate(...)`

这里是最核心的主循环。

如果你想知道：

- 每一轮什么时候 rollout
- 什么时候 actor 更新
- 什么时候 eval
- 什么时候同步权重

看 `EmbodiedRunner.run(...)`。

---

## 4. 环境这边怎么流

### 4.1 真正的 RoboTwin 环境在哪里

看：

- `src/rlinf/envs/robotwin/robotwin_env.py`
  - `__init__(...)`
  - `_init_env(...)`
  - `reset(...)`
  - `step(...)`
  - `chunk_step(...)`
  - `_extract_obs_image(...)`

这里负责：

- 起 RoboTwin `VectorEnv`
- `reset`
- `step`
- 把 RoboTwin 原始观测抽成：
  - `main_images`
  - `wrist_images`
  - `states`
  - `task_descriptions`

如果你想知道：

- obs 长什么样
- action 怎么喂回环境
- reward / termination / truncation 怎么来

看这个文件。

### 4.2 env worker 怎么和 rollout/actor 通信

看：

- `src/rlinf/workers/env/env_worker.py`
  - `interact(...)`
  - `_run_interact_once(...)`
  - `send_env_batch(...)`
  - `recv_rollout_results(...)`
  - `env_interact_step(...)`
  - `evaluate(...)`
  - `env_evaluate_step(...)`

这是“环境 worker 层”，不是单纯环境本体。

这里负责：

- 把 obs 发给 rollout
- 收 rollout 回来的 action
- 调 env.step
- 攒成 trajectory
- eval 时单独跑评测闭环

如果你想知道：

- env 和 rollout 怎么接
- trajectory 是怎么攒起来的
- eval 时环境怎么跑

看这个文件。

---

## 5. rollout 这边怎么流

看：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py`
  - `init_worker(...)`
  - `setup_sample_params(...)`
  - `generate(...)`
  - `evaluate(...)`
  - `recv_env_output(...)`
  - `predict(...)`
  - `send_chunk_actions(...)`
  - `send_rollout_result(...)`

这里的 rollout worker 干的事是：

- 收 env obs
- 调 policy 前向
- 产出 action / logprob / value
- 发回 env

如果你想知道：

- rollout 到底什么时候前向
- 训练模式和 eval 模式的采样参数怎么区别
- `temperature_train / temperature_eval` 用在什么地方

看这个文件。

---

## 6. pi0.5 / openpi 在哪里看

### 6.1 模型是怎么被加载的

看：

- `src/rlinf/models/embodiment/openpi/__init__.py`
  - `get_model(...)`

它负责：

- 读 `model_path`
- 加载 `model.safetensors` 或 checkpoint
- 加载 `norm_stats`
- 给模型挂 transform wrapper

如果你想知道：

- 模型 checkpoint 怎么进来
- norm stats 怎么加载

先看这里。

### 6.2 RoboTwin/ALOHA 数据适配在哪里

看：

- `src/rlinf/models/embodiment/openpi/dataconfig/__init__.py`
  - `get_openpi_config(...)`
- `src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py`
  - `LeRobotAlohaDataConfig.create(...)`

这一层负责把：

- RoboTwin / ALOHA 数据格式

适配成：

- pi0.5/openpi 想吃的格式

如果你想知道：

- 为什么 state 从 14 维变成模型内部 32 维
- 为什么要做 ALOHA 输入/输出适配
- quantile norm / z-score 配置从哪进来

看这两个文件。

### 6.3 policy 真正前向和出动作在哪里

看：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `obs_processor(...)`
  - `input_transform(...)`
  - `precision_processor(...)`
  - `predict_action_batch(...)`
  - `sample_actions(...)`

这是最关键的 policy 包装层。

这里负责：

- env obs -> policy obs
- transform / normalize
- Observation 构造
- pi0.5 前向
- action chunk 输出
- `prev_logprobs`
- `prev_values`
- `forward_inputs`

如果你想知道：

- obs 是怎么变成模型输入的
- action 是怎么出来的
- PPO 后面为什么还能重算 logprob/value

看这个文件。

### 6.4 更底层的 pi0.5 PyTorch 主体

看：

- `src/openpi/models_pytorch/pi0_pytorch.py`

如果你想看：

- pi0.5 网络主体
- flow/diffusion 样式动作采样
- value 头前后的内部细节

再往下进这个文件。

---

## 7. actor 训练这边怎么流

看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py`
  - `init_worker(...)`
  - `recv_rollout_trajectories(...)`
  - `_process_received_rollout_batch(...)`
  - `compute_advantages_and_returns(...)`
  - `run_training(...)`
  - `sync_model_to_rollout(...)`

这是训练端。

它负责：

- 收 env 发来的 trajectory
- 整成 rollout batch
- 算 advantage / return
- 用 `forward_inputs` 重新前向
- 算 PPO loss
- backward / optimizer step
- 把新权重发给 rollout

如果你想知道：

- PPO update 真正在哪里发生
- `prev_values` / `forward_inputs` 后面怎么用
- 为什么训练完要同步权重给 rollout

看这个文件。

---

## 8. 训练数据容器在哪里看

看：

- `src/rlinf/data/embodied_io_struct.py`
  - `EnvOutput`
  - `RolloutResult`
  - `Trajectory`
  - `EmbodiedRolloutResult`

这里主要回答：

- env 发出去的结构是什么
- rollout 回来的结构是什么
- 最后 actor 收到的 trajectory 长什么样

如果你想知道：

- `prev_values`
- `forward_inputs`
- `terminations`
- `truncations`
- `dones`

这些字段具体在哪一层出现，就看这个文件。

---

## 9. advantage / return / PPO loss 在哪里看

看：

- `src/rlinf/algorithms/advantages.py`
- `src/rlinf/algorithms/registry.py`
- `src/rlinf/algorithms/utils.py`

如果你想知道：

- GAE 怎么算
- return 怎么算
- advantage reshape/对齐是怎么做的

看这些。

PPO loss 细节再看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py`

---

## 10. 评测应该看哪里

### 10.1 独立 policy 评测入口

看：

- `scripts/eval_robotwin_policy.py`

这是现在最适合你的 eval 入口。它不走训练 runner，直接：

- 起 env
- 起 policy
- 跑 `reset -> predict_action_batch -> env.step`
- 聚合 eval metrics

如果你想做：

- pretrain model eval
- VLA-only checkpoint eval
- RL fine-tuned checkpoint eval

先看这个脚本。

### 10.2 数据流 trace 入口

看：

- `scripts/trace_robotwin_pipeline.py`

这个更适合看一次最小闭环的数据走向。

---

## 11. 想看什么，就去哪里

### 想看“整个训练从哪里开始”

看：

- `src/bentele/cli/train_embodied.py`
- `src/bentele/integration/rlinf_embodied.py`
- `src/rlinf/runners/embodied_runner.py`

### 想看“环境 obs 是怎么出来的”

看：

- `src/rlinf/envs/robotwin/robotwin_env.py`

### 想看“obs 怎么变成 pi0.5 输入”

看：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `obs_processor`
  - `input_transform`
  - `precision_processor`

### 想看“action 怎么出来的”

看：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `predict_action_batch`
  - `sample_actions`

### 想看“为什么有 prev_values / forward_inputs”

看：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py`
- `src/rlinf/data/embodied_io_struct.py`
- `src/rlinf/workers/actor/fsdp_actor_worker.py`

### 想看“PPO 是怎么更新的”

看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py`
  - `compute_advantages_and_returns`
  - `run_training`

### 想看“权重为什么同步给 rollout”

看：

- `src/rlinf/runners/embodied_runner.py`
  - `update_rollout_weights`
- `src/rlinf/workers/actor/fsdp_actor_worker.py`
  - `sync_model_to_rollout`

### 想看“纯评测怎么跑”

看：

- `scripts/eval_robotwin_policy.py`

---

## 12. 当前最推荐的阅读顺序

如果你第一次读这条链，建议顺序就是：

1. `src/bentele/configs/embodiment/robotwin_place_phone_stand_ppo_openpi_pi05.yaml`
2. `src/bentele/cli/train_embodied.py`
3. `src/bentele/integration/rlinf_embodied.py`
4. `src/rlinf/runners/embodied_runner.py`
5. `src/rlinf/workers/env/env_worker.py`
6. `src/rlinf/workers/rollout/hf/huggingface_worker.py`
7. `src/rlinf/models/embodiment/openpi/openpi_action_model.py`
8. `src/rlinf/workers/actor/fsdp_actor_worker.py`

这 8 个文件基本就能把主线串起来。
