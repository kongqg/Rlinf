> 说明
>
> 这份文件是之前的详细工作笔记，里面有些更早期的记录和历史内容。
> 如果你现在只想理解当前 `RoboTwin place_phone_stand + pi0.5 + PPO` 主线，
> 请先看 `docs/pipeline_navigation.md`。

my_rllnf



- src/bentele/
  你的项目层。以后你自己的入口、配置、任务定义、三阶段逻辑、数据组织，优先都往这里放。

- src/rlinf/
  底层 RL 引擎层。PPO/SAC、runner、worker、scheduler、env 对接都在这里。

- src/openpi/
  pi0.5 模型层。模型定义、PyTorch 推理实现、checkpoint、数据 transform 都在这里。

- src/openpi_client/
  runtime 和 websocket client 层。

- vendor_examples/rlinf/
  现在降级成“参考入口”，不再是主入口。

- examples/
  还是本地轻量 scaffold 示例，也不再是主入口。
  
  bentele 现在的包说明

- bentele.cli
  主入口包。真正从这里启动训练。

- bentele.integration
  桥接包。负责把 bentele 的入口接到 rlinf 的 runner/worker 上。

- bentele.configs
  你自己的 Hydra 配置目录。以后 ManiSkill、realworld、三阶段任务配置都应该收这里。

- bentele.algorithms
  旧的本地轻量算法 scaffold，目前不是主训练链。

- bentele.data
  旧的本地数据结构和 replay scaffold。

- bentele.envs
  旧的本地 env scaffold，现在主要还是 mock 用。

- bentele.models
  旧的本地 policy/prompt scaffold，不是现在真实 pi0.5 + RLinf 主链。

- bentele.runners
  旧的本地 runner scaffold。

- bentele.scheduler
  旧的本地 cluster/placement scaffold。

- bentele.workers
  
  
  
  
  
  
  旧的本地 worker scaffold。
  
  bentele 里关键文件现在怎么分工

- src/bentele/__init__.py:1
  说明 bentele 的定位：主包是 cli/configs/integration，旧 scaffold 还在但不再是主路径。

- src/bentele/cli/train_embodied.py:1
  同步 embodied 训练入口，默认走 bentele 自己的 ManiSkill smoke config。

- src/bentele/cli/train_embodied_async.py:1
  异步 embodied 训练入口。

- src/bentele/integration/rlinf_embodied.py:1
  这次最关键的桥接文件。负责：
  
  1. 调 rlinf.config.validate_cfg
  2. 建 cluster 和 placement
  3. 起 actor / rollout / env / reward group
  4. 选同步还是异步 runner

- src/bentele/configs/embodiment/maniskill_ppo_openpi_pi05_smoke.yaml:1
  bentele 自己的同步 smoke config。

- src/bentele/configs/embodiment/maniskill_async_ppo_openpi_pi05_smoke.yaml:1
  bentele 自己的异步 smoke config。

- src/bentele/configs/embodiment/env/maniskill_put_on_plate_in_scene_25_main.yaml:1
  bentele 自己持有的 ManiSkill env config。

- src/bentele/configs/embodiment/model/pi0_5.yaml:1
  pi0.5/openpi 模型配置模板。

- src/bentele/configs/embodiment/training_backend/fsdp.yaml:1
  FSDP 后端配置模板。
  
  旧 scaffold 那部分文件，现在怎么看

- bentele.data/schema.py
  轻量 Observation / Transition / SFTSample 定义。

- bentele.data/replay_buffer.py
  轻量 replay buffer。

- bentele.envs/mock_realworld.py
  假真机 env。

- bentele.models/pi05_policy.py
  假的 pi0.5 adapter，占位接口。

- bentele.models/prompting.py
  prompt 组装。

- bentele.runners/embodied_runner.py
  本地简化 runner。

- bentele.workers/*
  本地简化 actor/env/reward/rollout worker。
  
  这部分现在更适合当“接口草图”和“教学样例”，不适合当你真正的训练主线。
  
  
  
  
  
  rlinf 现在怎么用

- rlinf.config
  配置校验和模型/env/backend 注册。

- rlinf.algorithms
  PPO/SAC 的 advantage 和 loss。

- rlinf.data
  embodied rollout、trajectory、replay buffer 数据结构。

- rlinf.envs
  真正的 env registry 和动作适配；你现在重点会看 maniskill/ 和后面的 realworld/。

- rlinf.models
  真正把 openpi 包成 RL policy 的地方，关键文件还是
  src/rlinf/models/embodiment/openpi/openpi_action_model.py:1

- rlinf.runners
  同步/异步训练主循环。

- rlinf.workers
  actor、rollout、env、reward 四类 worker。

- rlinf.scheduler
  cluster、channel、placement、worker group 这一套基础设施。
  
  
  
  
  
  openpi 现在怎么用

- openpi.models
  原始 JAX/Flax 主实现。

- openpi.models_pytorch
  你现在真正会跑到的 PyTorch 实现，关键文件是
  src/openpi/models_pytorch/pi0_pytorch.py:1

- openpi.policies
  policy 包装层。

- openpi.training
  checkpoint、data loader、训练工具。

- openpi.shared
  normalize、image、download、类型工具。
  
  运行入口现在怎么记

- 主入口：
  scripts/run_bentele_maniskill.sh:1
  和
  scripts/run_bentele_maniskill_async.sh:1

- 参考入口：
  vendor_examples/rlinf/embodiment/*

- 旧 scaffold 入口：
  examples/*


## RLinf Pipeline 详细拆解

当前只讲你现在真正会跑的这条主线：

`bentele -> rlinf -> ManiSkill -> openpi/pi0.5 -> PPO`

先讲第一块：`程序启动 -> 配置加载 -> worker 初始化`。
这一块只讲系统怎么站起来，不讲 rollout 和训练循环。

### 1. 入口文件先拿到配置

看：

- `src/bentele/cli/train_embodied.py:11`

关键代码：

- 第 `11-15` 行：`@hydra.main(...)`
- 第 `16` 行：`def main(cfg: DictConfig) -> None:`
- 第 `17` 行：`run_sync_embodied_training(cfg)`

这里的数据流：

- 输入：
  启动脚本时指定的 Hydra 配置。默认是 `bentele` 自己这份 `maniskill_ppo_openpi_pi05_smoke`。
- 输出：
  一个 Hydra 解析好的 `cfg`，类型是 `DictConfig`。

到第 `17` 行为止，程序手里只有一份大配置 `cfg`，还没有环境、没有模型、没有 worker。

### 2. 进入 `bentele -> rlinf` 的桥接层

看：

- `src/bentele/integration/rlinf_embodied.py:99`

同步入口：

- 第 `99` 行：`def run_sync_embodied_training(cfg):`

#### 2.1 先校验配置

- 第 `100` 行：`validate_cfg(cfg)`

输入输出：

- 输入：`cfg`
- 输出：无返回值，原地检查

这里主要检查：

- `actor.strategy` 是否合法
- `env.env_type` 是否合法
- `actor.model.model_type` 是否合法

这一层还是配置检查，还没有创建训练对象。

### 3. 创建分布式放置关系

继续看：

- 第 `101` 行：`cluster = Cluster()`
- 第 `102-106` 行：`placement = HybridComponentPlacement(...)`

#### 3.1 `Cluster()`

输入输出：

- 输入：无
- 输出：`cluster`

它的作用不是环境，也不是模型，而是后面负责起 worker group。

#### 3.2 `HybridComponentPlacement(...)`


## RoboTwin Place Phone Stand 实际数据流

这一节不是泛讲，而是按一次真实 trace 来整理：

- 任务：`RoboTwin place_phone_stand`
- 配置：`robotwin_place_phone_stand_ppo_openpi_pi05`
- 模型：`pi0.5/openpi`
- 追踪脚本：
  [trace_robotwin_pipeline.py](/home/kqg/bentele/scripts/trace_robotwin_pipeline.py:1)
- 本地日志：
  [robotwin_place_phone_stand_pipeline.log](/home/kqg/Downloads/robotwin_place_phone_stand_pipeline.log:1)

这一段只关心一件事：

`配置 -> env.reset() -> obs_processor -> input_transform -> predict_action_batch -> RolloutResult -> env.step()`

### 1. 配置输入

主配置：

- [robotwin_place_phone_stand_ppo_openpi_pi05.yaml](/home/kqg/bentele/src/bentele/configs/embodiment/robotwin_place_phone_stand_ppo_openpi_pi05.yaml:1)

环境配置：

- [robotwin_place_phone_stand.yaml](/home/kqg/bentele/src/bentele/configs/embodiment/env/robotwin_place_phone_stand.yaml:1)

这一步解析出来的关键字段是：

- `task_name = place_phone_stand`
- `planner_backend = mplib`
- `assets_path = ~/RoboTwin`
- `actor.model.model_path = ~/bentele/model/pi05_base_torch`
- `actor.model.openpi.config_name = pi05_aloha_robotwin`
- `sampling.do_sample = true`
- `temperature_train = 1.0`
- `temperature_eval = 0.6`

对应 trace 里的代码位置：

- [trace_robotwin_pipeline.py:153](/home/kqg/bentele/scripts/trace_robotwin_pipeline.py:153)

### 2. 环境初始化

环境类：

- [robotwin_env.py:32](/home/kqg/bentele/src/rlinf/envs/robotwin/robotwin_env.py:32)

真正起 RoboTwin 的位置：

- [robotwin_env.py:78](/home/kqg/bentele/src/rlinf/envs/robotwin/robotwin_env.py:78)

这一层输入是：

- `cfg.env.train`
- `num_envs=1`
- `seed_offset=0`

这一层输出是一个 `RoboTwinEnv` 实例，trace 里确认到：

- `task_name = place_phone_stand`
- `device = cuda`
- `use_custom_reward = true`
- `use_rel_reward = true`

### 3. `env.reset()` 输出的原始观测

reset 入口：

- [robotwin_env.py:238](/home/kqg/bentele/src/rlinf/envs/robotwin/robotwin_env.py:238)

观测提取：

- [robotwin_env.py:161](/home/kqg/bentele/src/rlinf/envs/robotwin/robotwin_env.py:161)

这一层输入：

- 当前 `RoboTwinEnv`
- 内部 task state

这一层输出：

- `main_images`: `torch.uint8`, shape `[1, 240, 320, 3]`
- `wrist_images`: `torch.uint8`, shape `[1, 2, 240, 320, 3]`
- `states`: `torch.float64`, shape `[1, 14]`
- `task_descriptions`: `list[str]`

这里的 `states` 还是 RoboTwin/ALOHA 的原始 14 维状态，不是 pi0.5 最终吃的 32 维 state。

### 4. 包成 `EnvOutput`

相关代码：

- [embodied_io_struct.py:93](/home/kqg/bentele/src/rlinf/data/embodied_io_struct.py:93)
- [embodied_io_struct.py:241](/home/kqg/bentele/src/rlinf/data/embodied_io_struct.py:241)

这一层输入：

- 上一步 `reset_obs`

这一层输出：

- `EnvOutput.to_dict()`

结构上主要变成：

- `obs.main_images`
- `obs.wrist_images`
- `obs.states`
- `obs.task_descriptions`

这一步的意义只是把环境侧观测打包成 rollout worker 会收的 transport 格式，还没有模型变换。

### 5. 建 OpenPI / pi0.5 policy 和 dataconfig

模型入口：

- [openpi/__init__.py:22](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/__init__.py:22)

dataconfig 入口：

- [dataconfig/__init__.py:439](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/dataconfig/__init__.py:439)

RoboTwin ALOHA 适配：

- [robotwin_aloha_dataconfig.py:76](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py:76)

这一层输入：

- `cfg.actor.model`
- `cfg.actor.model.openpi_data`
- `model_path = ~/bentele/model/pi05_base_torch`

这一层输出：

- `OpenPi0ForRLActionPrediction`
- `LeRobotAlohaDataConfig`
- 已加载的 `physical-intelligence/robotwin/norm_stats.json`

trace 里确认到：

- `repo_id = physical-intelligence/robotwin`
- `action_chunk = 50`
- `action_env_dim = 14`
- `num_steps = 5`

### 6. `obs_processor`：环境观测改成 policy 观测 key

代码：

- [openpi_action_model.py:476](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/openpi_action_model.py:476)

这一层输入还是环境那套 key：

- `main_images`
- `wrist_images`
- `states`
- `task_descriptions`

这一层输出变成：

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

也就是说，到这一步，字段名已经从 env 风格变成 policy 风格了，但数值语义还没有彻底转换。

### 7. `input_transform`：真正做 repack/data/model transforms

代码：

- [openpi_action_model.py:249](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/openpi_action_model.py:249)
- [robotwin_aloha_dataconfig.py:76](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py:76)
- [model.py:109](/home/kqg/bentele/src/openpi/models/model.py:109)

这一层输入：

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

这一层输出是已经适配到 pi0.5 训练语义的数据：

- `image.base_0_rgb`
- `image.left_wrist_0_rgb`
- `image.right_wrist_0_rgb`
- `image_mask.*`
- `state`: shape `[1, 32]`
- `tokenized_prompt`: shape `[1, 200]`
- `tokenized_prompt_mask`: shape `[1, 200]`

这里最关键的是：

- 原来的 14 维 `observation/state`
- 被 RoboTwin ALOHA dataconfig 适配成了 pi0.5 最终使用的 32 维 `state`

所以这一步已经不是简单重命名，而是：

- 相机 key 重排
- ALOHA state 适配
- prompt tokenize
- norm stats 参与的 state 归一化路径

### 8. `precision_processor + Observation.from_dict`

代码：

- [openpi_action_model.py:499](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/openpi_action_model.py:499)
- [model.py:109](/home/kqg/bentele/src/openpi/models/model.py:109)

这一层输入：

- 上一步 transform 后的 dict

这一层输出是模型真正前向时的 `Observation`：

- 三路图像都变成 `[1, 3, 224, 224]`
- dtype 变成 `float32`
- device 变成 `cuda:0`
- 图像数值范围变成 `[-1, 1]`
- `state` 仍是 `[1, 32]`
- `tokenized_prompt` 和 mask 也搬到 `cuda`

这一步之后，已经是标准的模型输入对象，不再是 env/raw dict。

### 9. `predict_action_batch`

代码：

- [huggingface_worker.py:248](/home/kqg/bentele/src/rlinf/workers/rollout/hf/huggingface_worker.py:248)
- [openpi_action_model.py:518](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/openpi_action_model.py:518)
- [openpi_action_model.py:613](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/openpi_action_model.py:613)
- [openpi_action_model.py:294](/home/kqg/bentele/src/rlinf/models/embodiment/openpi/openpi_action_model.py:294)

这一层输入：

- 原始 env obs

内部会自己再走：

- `obs_processor`
- `input_transform`
- `precision_processor`
- `sample_actions`
- `output_transform`

这一层输出：

- `actions`: shape `[1, 50, 14]`
- `prev_logprobs`: shape `[1, 50, 14]`
- `prev_values`: shape `[1, 1]`
- `forward_inputs`

这里的 `forward_inputs` 很关键，它保存了 actor 训练时要重新前向的上下文，比如：

- `chains`
- `denoise_inds`
- `tokenized_prompt`
- `action`
- `model_action`
- 原始 `observation/image`
- 原始 `observation/state`
- 原始 `observation/wrist_image`

所以这一步不只是“出动作”，还顺手把 PPO 重算 logprob/value 所需的上下文缓存出来了。

### 10. 包成 `RolloutResult`

代码：

- [embodied_io_struct.py:261](/home/kqg/bentele/src/rlinf/data/embodied_io_struct.py:261)
- [huggingface_worker.py:392](/home/kqg/bentele/src/rlinf/workers/rollout/hf/huggingface_worker.py:392)

这一层输入：

- `actions`
- `prev_logprobs`
- `prev_values`
- `forward_inputs`

这一层输出：

- `RolloutResult`

这就是 rollout worker 会发回 env/actor 的标准结构。

### 11. `env.step(actions)`

代码：

- [robotwin_env.py:258](/home/kqg/bentele/src/rlinf/envs/robotwin/robotwin_env.py:258)

这一层输入：

- `actions`: shape `[1, 50, 14]`

这一层输出：

- `next_obs`
- `rewards`
- `terminations`
- `truncations`
- `step_infos`

trace 里这次拿到的是：

- `rewards`: `[1]`
- `terminations`: `[1]`
- `truncations`: `[1]`
- `step_infos.success`
- `step_infos.episode.success_once / return / episode_len / reward`

也就是说，动作 chunk 已经真实喂回 RoboTwin 环境了，链路不是停在模型输出。

## 一句话主线

你现在这个 `place_phone_stand` 任务，真实的数据走向是：

1. Hydra 配置解析出 env/model/sampling 参数
2. `RoboTwinEnv.reset()` 产出原始图像、14 维 state、任务文本
3. `EnvOutput` 把它打包成 rollout 传输格式
4. `obs_processor` 改成 policy 风格 key
5. `input_transform` 把 RoboTwin/ALOHA 数据适配成 pi0.5 要的三路图像、32 维 state、tokenized prompt
6. `precision_processor + Observation.from_dict` 变成真正模型输入
7. `predict_action_batch` 产出 `[B, 50, 14]` 的动作 chunk，并缓存 `forward_inputs`
8. `RolloutResult` 打包 rollout 输出
9. `env.step(actions)` 把动作喂回环境，得到下一步 obs/reward/done

## 你现在最该盯的 3 个边界

如果你后面只想真正理解这条链，最值得反复看的不是全链所有函数，而是这 3 个边界：

- `env.reset()` 之后  
  这里看 env 原始观测长什么样

- `input_transform()` 之后  
  这里看 RoboTwin/ALOHA 数据怎么变成 pi0.5 输入

- `predict_action_batch()` 之后  
  这里看 policy 真正产出的动作 chunk、value、forward_inputs

这三处一旦看懂，这条 `place_phone_stand` pipeline 基本就串起来了。

它从 `cfg.cluster` 里读这些字段：

- `actor_pool_id`
- `rollout_pool_id`
- `env_pool_id`
- `reward_pool_id`

输入输出：

- 输入：`cfg.cluster.*`
- 输出：`placement`

这个对象只负责一件事：定义 actor、rollout、env、reward 分别放到哪个资源池。

### 4. 创建 actor group

看：

- `src/bentele/integration/rlinf_embodied.py:40`

函数：

- 第 `40` 行：`def _create_sync_actor_group(cluster, placement, cfg):`

关键代码：

- 第 `41-48` 行：从 `cfg.actor.fsdp_config` 取 FSDP 配置
- 第 `49-57` 行：`cluster.spawn(...)`

这里的数据流：

- 输入：
  - `cluster`
  - `placement`
  - `cfg.actor`
  - `cfg.actor.fsdp_config`
- 输出：
  - `actor_worker_group`

这里起出来的是一个 actor group，里面真正执行训练逻辑的 worker 类是：

- `FSDPActorWorker`

同时这里把这些关键信息传进去了：

- `role="actor"`
- `model_type=cfg.actor.model.model_type`
- `strategy=cfg.actor.strategy`
- `world_size=cfg.actor.fsdp_config.actor.world_size`

这些字段决定：

- actor 用什么模型类型
- actor 用什么训练策略
- actor 有多少 rank

### 5. 创建 rollout / env / reward group

回到：

- `src/bentele/integration/rlinf_embodied.py:99`

继续看：

- 第 `108-113` 行：创建 `rollout_worker_group`
- 第 `115-120` 行：创建 `env_worker_group`
- 第 `122-127` 行：如果启用 reward model，就创建 `reward_worker_group`

#### 5.1 rollout group

输入输出：

- 输入：
  - `cluster`
  - `cfg.rollout`
  - `placement.rollout_pool_id`
- 输出：
  - `rollout_worker_group`

它对应的真实逻辑类是：

- `HuggingFaceWorker`

它负责：

- 接收 env 发来的观测
- 调 policy 推理
- 把 action / logprob / value 发回 env

#### 5.2 env group

输入输出：

- 输入：
  - `cluster`
  - `cfg.env`
  - `placement.env_pool_id`
- 输出：
  - `env_worker_group`

它对应的真实逻辑类是：

- `EnvWorker`

它负责：

- 建环境
- `reset`
- `step`
- 收 reward / done
- 把轨迹打包发给 actor

#### 5.3 reward group

当前 smoke config 里：

- `reward.use_reward_model: false`

所以这条主线默认不建 reward worker。

### 6. 用这些 group 组装成一个 runner

继续看：

- 第 `129-135` 行：`runner = EmbodiedRunner(...)`

看 runner 本体：

- `src/rlinf/runners/embodied_runner.py:52`

#### 6.1 `EmbodiedRunner.__init__`

关键代码：

- 第 `75` 行：`self.env_channel = Channel.create("Env")`
- 第 `76` 行：`self.rollout_channel = Channel.create("Rollout")`
- 第 `77` 行：`self.actor_channel = Channel.create("Actor")`
- 第 `78-81` 行：如果有 reward，再建 `reward_channel`

输入输出：

- 输入：
  - `actor_worker_group`
  - `rollout_worker_group`
  - `env_worker_group`
  - `reward_worker_group`
  - `cfg`
- 输出：
  - 一个 `runner`
  - runner 内部持有 3 到 4 条 channel

这几条 channel 的职责：

- `Env` channel：
  env 把观测发给 rollout
- `Rollout` channel：
  rollout 把 action / value / logprob 发回 env
- `Actor` channel：
  env 把 trajectory 发给 actor
- `Reward` channel：
  只有启用 reward model 时才用

从这一步开始，后面所有主要数据流都会走这些 channel。

### 7. `runner.init_workers()` 真正初始化各角色

回到桥接层：

- 第 `136` 行：`runner.init_workers()`

看：

- `src/rlinf/runners/embodied_runner.py:134`

关键代码：

- 第 `135` 行：`self.rollout_wg.init_worker()`
- 第 `136` 行：`self.env_wg.init_worker()`
- 第 `137-138` 行：reward 可选
- 第 `139` 行：`self.actor_wg.init_worker()`

顺序是：

1. rollout
2. env
3. reward
4. actor

这里的含义是：先让 rollout 和 env 站起来，actor 最后初始化。

### 8. rollout 初始化时干了什么

看：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py:90`

函数：

- `init_worker()`

#### 8.1 先拷一份 actor 的模型配置

大致逻辑是：

- 先 `copy.deepcopy(self.config.actor.model)`
- 再用 `cfg.rollout.model` 覆盖 rollout 专属字段

输入输出：

- 输入：`cfg.actor.model`
- 输出：`rollout_model_config`

这表示 rollout 和 actor 用同一类模型结构，但 rollout 可以有自己独立的推理精度和权重路径。

#### 8.2 真正加载模型

后面会调用：

- `self.hf_model = get_model(rollout_model_config)`

输入输出：

- 输入：`rollout_model_config`
- 输出：`self.hf_model`

当前你的配置里，这里最终会加载 `openpi/pi0.5` 对应的 RL 包装模型。

也就是说，到这一步 rollout worker 已经持有一个可推理的 policy。

#### 8.3 建立 rollout 和 env/actor 的 rank 映射

初始化里还会建立：

- `dst_ranks`
- `src_ranks`

它们不是业务数据，而是通信拓扑：

- rollout 从哪些 env rank 收输入
- rollout 把输出发回哪些 env rank
- rollout 从哪些 actor rank 收权重

#### 8.4 设置采样参数

初始化里还会调用：

- `setup_sample_params()`

它根据配置决定：

- train / eval 模式
- 是否采样
- action chunk 长度
- 推理时的采样参数

所以 rollout 初始化完成后的状态是：

- 已加载好的 `hf_model`
- env <-> rollout 的通信映射
- actor -> rollout 的权重同步映射
- 一套采样参数

### 9. env 初始化时干了什么

看：

- `src/rlinf/workers/env/env_worker.py:124`

函数：

- `init_worker()`

#### 9.1 先确定通信映射

它会先做：

- `_setup_dst_rank_map()`
- `_setup_src_rank_map()`

作用是：

- env 要把 obs 发给哪些 rollout rank
- env 要从哪些 rollout rank 收 action
- env 要把 trajectory 发给哪些 actor rank

#### 9.2 根据配置选环境类

后面会调用：

- `get_env_cls(...)`

核心输入来自：

- `cfg.env.env_type`

你当前配置里是：

- `env_type: maniskill`

所以这里选出来的是：

- `ManiskillEnv`

输入输出：

- 输入：`cfg.env.env_type = "maniskill"`
- 输出：`env_cls = ManiskillEnv`

#### 9.3 真正创建环境对象

后面会继续执行：

- `_setup_env_and_wrappers(...)`
- `_init_env()`

它会根据：

- `cfg.env.train.total_num_envs`
- stage 数
- worker 切分关系

真正创建：

- `self.env_list`

当前 smoke config 里：

- `total_num_envs = 4`

所以 env worker 初始化完成后，手里已经持有对应的 ManiSkill 环境实例。

### 10. actor 初始化时干了什么

这一块先抓核心，不往 optimizer/FSDP 细节里展开。

在 `runner.init_workers()` 最后：

- `self.actor_wg.init_worker()`

它的目标本质上是两件事：

1. 建训练模型
2. 建优化器/FSDP/训练状态

输入输出：

- 输入：
  - `cfg.actor.model`
  - `cfg.actor.optim`
  - `cfg.actor.fsdp_config`
  - `cfg.algorithm`
- 输出：
  - 一个可训练的 actor 模型
  - 优化器
  - 训练状态

这里和 rollout 的区别是：

- rollout 持有推理版模型
- actor 持有训练版模型

### 11. 这一块结束时，系统里已经有什么

到 `runner.init_workers()` 结束，系统状态是：

已存在的对象：

- `cfg`
- `cluster`
- `placement`
- `runner`
- `actor_worker_group`
- `rollout_worker_group`
- `env_worker_group`

已存在的通信通道：

- `Env channel`
- `Rollout channel`
- `Actor channel`

已初始化好的角色：

- rollout：已经加载 `pi0.5/openpi` 模型
- env：已经创建 ManiSkill 环境
- actor：已经创建训练模型和优化器

但此时还没有发生的事：

- 还没有 `reset` 环境
- 还没有生成 action
- 还没有收 trajectory
- 还没有做 PPO update

也就是说，这一块只是把系统架起来。

### 12. 这一块的心智模型

这一块可以压缩成：

- `train_embodied.py` 负责拿配置
- `rlinf_embodied.py` 负责按配置拼系统
- `EmbodiedRunner` 负责拿到所有角色和 channel
- `init_workers()` 负责让每个角色把自己准备好

这一块结束后的结构就是：

`cfg -> bridge -> runner -> channels -> env / rollout / actor`

这里只完成“启动和初始化”，还没有开始真正的数据交互。


## 第二块：初始化之后，真正的数据交互怎么开始

这一块讲的是：

`EnvWorker 第一次拿观测 -> 发给 RolloutWorker -> OpenPI/pi0.5 推理 -> RolloutResult 发回 EnvWorker -> EnvWorker 真正 step 环境`

这里先只讲一轮最小闭环，不讲 actor 怎么更新参数。

### 1. runner 先同时启动 env 和 rollout

看：

- `src/rlinf/runners/embodied_runner.py:268`

关键代码：

- 第 `281-286` 行：`self.env.interact(...)`
- 第 `287-290` 行：`self.rollout.generate(...)`
- 第 `296-298` 行：`self.actor.recv_rollout_trajectories(...)`

这里的含义是：

- `env.interact(...)` 开始负责环境侧主循环
- `rollout.generate(...)` 开始负责策略推理循环
- `actor.recv_rollout_trajectories(...)` 先阻塞等 rollout 结束后的轨迹

所以真正的数据流起点，是 `EnvWorker.interact(...)` 和 `RolloutWorker.generate(...)` 这两个循环同时跑起来。

### 2. `EnvWorker` 先做 bootstrap，把第一帧观测送出去

看：

- `src/rlinf/workers/env/env_worker.py:911`

函数：

- 第 `911` 行：`async def _run_interact_once(...)`

它一开始先建 rollout 缓存：

- 第 `920-925` 行：`self.rollout_results = [EmbodiedRolloutResult(...)]`

这说明 env worker 后面会一边和环境交互，一边把每一步结果攒到 `EmbodiedRolloutResult` 里。

#### 2.1 先做 bootstrap

- 第 `928` 行：进入每个 `epoch`
- 第 `929` 行：`env_outputs = self.bootstrap_step()`

看 bootstrap 本体：

- `src/rlinf/workers/env/env_worker.py:829`

关键代码：

- 第 `841` 行：`extracted_obs, infos = self.env_list[stage_id].reset()`
- 第 `846-858` 行：把 reset 的结果包成 `EnvOutput`

这里的数据流：

- 输入：
  - 一个 ManiSkill 环境实例
- 输出：
  - `EnvOutput`

`EnvOutput` 的结构定义在：

- `src/rlinf/data/embodied_io_struct.py:48`

字段主要有：

- `obs`
- `final_obs`
- `dones`
- `terminations`
- `truncations`
- `rewards`
- `intervene_actions`
- `intervene_flags`

在 bootstrap 这一步里，最关键的是：

- `obs`：reset 后的第一帧观测
- `dones/terminations/truncations`：全零
- `rewards`：此时还没有

#### 2.2 `EnvOutput.obs` 里面到底有什么

这个要看 ManiSkill env 自己怎么包装观测。

看：

- `src/rlinf/envs/maniskill/maniskill_env.py`

在 `_wrap_obs(...)` 里，当前主观测会被整理成：

- `main_images`
- `states`
- `task_descriptions`

也就是说，`bootstrap_step()` 最终拿到的 `obs`，对当前 `pi0.5 + ManiSkill` 路径来说，核心就是：

- 图像
- 机器人状态
- 语言描述

### 3. `EnvWorker` 把第一批观测发给 `RolloutWorker`

继续看：

- `src/rlinf/workers/env/env_worker.py:930`

关键代码：

- 第 `931-932` 行：`env_output.to_dict()`
- 第 `933-939` 行：`self.send_env_batch(...)`

真正发的内容是：

- `"obs": env_batch["obs"]`
- `"final_obs": env_batch["final_obs"]`

看发送函数：

- `src/rlinf/workers/env/env_worker.py:712`

关键代码：

- 第 `729-731` 行：按 `dst_rank_map["rollout_train"]` 切 batch
- 第 `733-736` 行：`rollout_channel.put(...)`

这里的数据流：

- 输入：
  - `env_batch = {"obs": ..., "final_obs": ...}`
- 输出：
  - 若干个发往 rollout rank 的分片 batch

也就是说，env 不会把整个 batch 一股脑广播给所有 rollout worker，而是：

1. 先按映射关系切 batch
2. 每个 rollout rank 只拿自己那一份

### 4. `RolloutWorker` 收到观测并合并 batch

看：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py:392`

关键代码：

- 第 `396` 行：`env_output = await self.recv_env_output(input_channel)`
- 第 `397` 行：`actions, result = self.predict(env_output["obs"])`

先看 `recv_env_output(...)`：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py:489`

关键代码：

- 第 `503` 行：`src_ranks_and_sizes = self.src_ranks[mode]`
- 第 `505-517` 行：从多个 env rank 拉数据
- 第 `518` 行：`return self._merge_obs_batches(obs_batches)`

再看合并：

- 第 `551-587` 行：`_merge_obs_batches(...)`

这里的逻辑是：

- 如果一个 rollout worker 对应多个 env worker
- 那就把多个 env shard 在 batch 维拼起来

输入输出：

- 输入：多个 env 发来的 `{"obs": ..., "final_obs": ...}`
- 输出：一个合并后的：
  - `{"obs": merged_obs, "final_obs": merged_final_obs}`

所以到第 `397` 行时，rollout 手里已经是一整个推理 batch 了。

### 5. `RolloutWorker.predict()` 真正调 `pi0.5`

看：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py:248`

关键代码：

- 第 `297-300` 行：
  `actions, result = self.hf_model.predict_action_batch(env_obs=env_obs, **kwargs)`

这里的 `self.hf_model`，对你当前这条主线，就是 `OpenPIActionModel`。

所以真正的 VLA 推理入口是：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py:518`

函数：

- `predict_action_batch(...)`

### 6. `OpenPIActionModel` 先把 env obs 变成模型输入

先看：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py:476`

关键代码：

- 第 `478-480` 行：
  - `"observation/image" = env_obs["main_images"]`
  - `"prompt" = env_obs["task_descriptions"]`
- 第 `489` 行：
  - `"observation/state" = env_obs["states"]`

这一步就是最重要的一层映射：

- env 世界里的键：
  - `main_images`
  - `states`
  - `task_descriptions`
- 被改成 openpi 世界里的键：
  - `observation/image`
  - `observation/state`
  - `prompt`

#### 6.1 再做 dataset/policy transform

看 `predict_action_batch(...)`：

- 第 `525` 行：`to_process_obs = self.obs_processor(env_obs)`
- 第 `526-528` 行：`processed_obs = self.input_transform(to_process_obs, transpose=False)`

这里的 `input_transform`，对 ManiSkill 来说会走：

- `src/rlinf/models/embodiment/openpi/policies/maniskill_policy.py:41`

`ManiSkillInputs.__call__(...)` 会把输入整理成模型真正吃的格式：

- `state`
- `image`
- `image_mask`
- `prompt`

关键代码：

- 第 `67` 行：取 `observation/image`
- 第 `70-84` 行：组织 `state / image / image_mask`
- 第 `94-95` 行：把 `prompt` 带进去

所以这一步后的数据已经不是 env 风格，而是 openpi 模型风格。

### 7. `pi0.5` 采样动作，并把训练时需要的上下文一起保存

继续看：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py:518`

关键代码：

- 第 `532` 行：`observation = _model.Observation.from_dict(processed_obs)`
- 第 `566-568` 行：`outputs = self.sample_actions(...)`
- 第 `569-571` 行：`actions = self.output_transform(... )["actions"]`

再看 `sample_actions(...)`：

- `src/rlinf/models/embodiment/openpi/openpi_action_model.py:613`

关键输出字段：

- 第 `741-747` 行：
  - `actions`
  - `chains`
  - `prev_logprobs`
  - `prev_values`
  - `denoise_inds`

这里的含义：

- `actions`
  模型原始采样出来的动作序列
- `prev_logprobs`
  rollout 时当前 policy 对这些动作的旧 logprob
- `prev_values`
  rollout 时当前 value 估计
- `chains`
  diffusion 采样链
- `denoise_inds`
  这次 PPO/logprob 训练对应的 denoise step 标记

#### 7.1 为什么还要构建 `forward_inputs`

回到 `predict_action_batch(...)`：

- 第 `576-588` 行：构建 `forward_inputs`
- 第 `599-603` 行：把观测拷贝也放进 `forward_inputs`

`forward_inputs` 里最关键的字段有：

- `chains`
- `denoise_inds`
- `tokenized_prompt`
- `tokenized_prompt_mask`
- `action`
- `model_action`
- 观测副本

这一步非常关键，因为 rollout 不只是要给 env 一个 action，还要给 actor 留下“训练时重新前向计算”所需的上下文。

也就是说：

- `actions` 给环境执行
- `forward_inputs` 给后面 actor 训练时复算 logprob/value

最终 `predict_action_batch(...)` 的输出是：

- 返回值 1：`actions`
- 返回值 2：`result`

其中 `result` 里至少有：

- `prev_logprobs`
- `prev_values`
- `forward_inputs`

### 8. `RolloutWorker` 把模型输出打包成 `RolloutResult`

回到：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py:392`

关键代码：

- 第 `407-425` 行：构造 `RolloutResult(...)`
- 第 `426` 行：`self.send_rollout_result(...)`

`RolloutResult` 的定义在：

- `src/rlinf/data/embodied_io_struct.py:261`

字段主要有：

- `actions`
- `prev_logprobs`
- `prev_values`
- `bootstrap_values`
- `save_flags`
- `forward_inputs`
- `versions`

这里每个字段的职责是：

- `actions`
  发回 env，用来真正执行
- `prev_logprobs`
  PPO 的旧策略 logprob
- `prev_values`
  PPO/GAE 的旧 value
- `bootstrap_values`
  给自动 reset/截断回报补 bootstrap
- `forward_inputs`
  后面 actor 重新前向用
- `versions`
  这批 rollout 对应的 policy 版本号

### 9. `RolloutWorker` 把 `RolloutResult` 切分后发回 `EnvWorker`

看：

- `src/rlinf/workers/rollout/hf/huggingface_worker.py:662`

关键代码：

- 第 `669-671` 行：按 env 目标 rank 切 `RolloutResult`
- 第 `675-680` 行：`output_channel.put(...)`

这里和前面的 env -> rollout 正好反过来：

- 前面是 env 按 rollout rank 切观测
- 这里是 rollout 按 env rank 切动作和 rollout 结果

所以 `RolloutResult` 不是一个全局大包直接丢回去，而是按 env worker 需要的 batch 大小分片回传。

### 10. `EnvWorker` 收到 `RolloutResult`，先记账，再真正 step 环境

回到：

- `src/rlinf/workers/env/env_worker.py:966`

关键代码：

- 第 `966-968` 行：`rollout_result = self.recv_rollout_results(...)`
- 第 `969-971` 行：`rewards = self.compute_bootstrap_rewards(...)`
- 第 `972-990` 行：构造 `ChunkStepResult(...)`
- 第 `991` 行：`append_step_result(...)`

先看 `recv_rollout_results(...)`：

- `src/rlinf/workers/env/env_worker.py:611`

关键代码：

- 第 `635-648` 行：从多个 rollout rank 收 shard
- 第 `650` 行：`RolloutResult.merge_rollout_results(...)`

这和 rollout 收 obs 时一样，也是：

- 先分片传
- 到目标 env rank 再拼回来

#### 10.1 `ChunkStepResult` 在这里起什么作用

定义在：

- `src/rlinf/data/embodied_io_struct.py:332`

字段主要有：

- `actions`
- `prev_logprobs`
- `prev_values`
- `dones`
- `truncations`
- `terminations`
- `rewards`
- `forward_inputs`
- `versions`

这一步本质上是在做：

**把“rollout 给的策略信息”和“env 当前这一步的环境反馈”拼成一个训练样本块。**

然后：

- 第 `991` 行：`self.rollout_results[stage_id].append_step_result(chunk_step_result)`

看这个容器：

- `src/rlinf/data/embodied_io_struct.py:501`

`EmbodiedRolloutResult.append_step_result(...)` 会把：

- `actions`
- `rewards`
- `dones`
- `prev_logprobs`
- `prev_values`
- `versions`
- `forward_inputs`

逐步累积起来，等 rollout 结束后再转成 trajectory。

### 11. 记完这一步以后，`EnvWorker` 才真正执行环境 step

看：

- `src/rlinf/workers/env/env_worker.py:997`

关键代码：

- 第 `997-999` 行：
  `env_output, env_info = self.env_interact_step(rollout_result.actions, stage_id)`

看 `env_interact_step(...)`：

- `src/rlinf/workers/env/env_worker.py:380`

关键代码：

- 第 `386-394` 行：`prepare_actions(...)`
- 第 `397-399` 行：`self.env_list[stage_id].chunk_step(chunk_actions)`
- 第 `440-449` 行：把 step 结果重新打包成 `EnvOutput`

这里的数据流非常重要：

#### 11.1 输入是什么

- `rollout_result.actions`

这就是刚才 policy 给出来的动作 chunk。

#### 11.2 中间发生了什么

- `prepare_actions(...)`
  把模型输出改造成环境真正要的动作格式
- `chunk_step(chunk_actions)`
  让 ManiSkill 连续执行一个 action chunk

#### 11.3 输出是什么

新的 `EnvOutput`，里面有：

- `obs`
  执行完 chunk 后的最新观测
- `rewards`
  这个 chunk 内各子步的 reward
- `dones`
- `terminations`
- `truncations`
- `final_obs`

所以这里形成了真正的环境交互闭环：

`obs -> policy -> actions -> env.step -> next_obs/reward/done`

### 12. 新的 `EnvOutput` 再被送回 rollout，下一轮继续

回到 `_run_interact_once(...)`：

- 第 `1000` 行：`env_batch = env_output.to_dict()`
- 第 `1001-1007` 行：再次 `send_env_batch(...)`

也就是说：

1. rollout 给出动作
2. env 真正执行
3. env 得到新的 obs
4. 新 obs 再发回 rollout

于是形成循环。

### 13. rollout 过程中如果要存 transition，也是在这里补

看：

- `src/rlinf/workers/env/env_worker.py:1008`

关键代码：

- 第 `1009-1013` 行：确定 `next_obs`
- 第 `1014-1016` 行：`append_transitions(curr_obs, next_obs)`

对应容器方法在：

- `src/rlinf/data/embodied_io_struct.py:616`

它会把：

- 当前观测 `curr_obs`
- 下一个观测 `next_obs`

也存起来，给以后需要 transition 级训练或分析的分支使用。

### 14. 一整个 rollout 结束后，env 才把攒好的结果发给 actor

看：

- `src/rlinf/workers/env/env_worker.py:1060`

关键代码：

- 第 `1061-1064` 行：`send_rollout_trajectories(...)`

发送函数在：

- `src/rlinf/workers/env/env_worker.py:901`

关键代码：

- 第 `904-906` 行：`rollout_result.to_splited_trajectories(...)`
- 第 `907-908` 行：`channel.put(trajectory, async_op=True)`

再看 `EmbodiedRolloutResult.to_splited_trajectories(...)`：

- `src/rlinf/data/embodied_io_struct.py:682`

它会先把累计好的 step 结果转成 `Trajectory`，再沿 batch 维切给 actor rank。

### 15. 这一块结束时，数据流已经完成了什么

到这里为止，已经完成的是：

1. env reset 得到第一帧 obs
2. obs 通过 channel 发给 rollout
3. rollout 调 `pi0.5/openpi` 做推理
4. rollout 产出：
   - `actions`
   - `prev_logprobs`
   - `prev_values`
   - `forward_inputs`
5. rollout 把这些封装成 `RolloutResult` 发回 env
6. env 用 `actions` 真正 step 环境
7. env 得到新的 `obs/reward/done`
8. env 把这一步包装成 `ChunkStepResult`
9. 多步累计成 `EmbodiedRolloutResult`
10. rollout 结束后转成 `Trajectory` 发给 actor

所以这一块的心智模型可以压成一句话：

`EnvWorker` 负责拿环境数据，`RolloutWorker` 负责把它变成策略输出，`EnvWorker` 再把策略输出变回真实环境反馈，并把训练需要的上下文一起攒起来。

### 16. 这一块最关键的两个中间对象

如果你现在只想记住两个核心对象，记这两个就够了：

- `EnvOutput`
  表示“环境这一侧刚产生了什么”
- `RolloutResult`
  表示“策略这一侧刚产生了什么”

然后 `ChunkStepResult` 是把这两边在一个 chunk step 上拼起来的训练中间态。


## 第三块：`Trajectory` 到底怎么变成训练更新

这一块讲的是：

`EnvWorker 把 trajectory 发给 ActorWorker -> ActorWorker 整理 batch -> 算 advantage/return -> 用 forward_inputs 重新前向 -> 算 PPO loss -> 反向传播和 optimizer step`

这一块才是“真正训练”的部分。

### 1. `ActorWorker` 先收 trajectory

看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py:1110`

函数：

- `recv_rollout_trajectories(...)`

关键代码：

- 第 `1119-1121` 行：先算这次应该收多少个 trajectory split
- 第 `1123-1126` 行：循环从 channel 里收 `Trajectory`
- 第 `1128` 行：`self.rollout_batch = convert_trajectories_to_batch(recv_list)`
- 第 `1130` 行：`self.rollout_batch = self._process_received_rollout_batch(...)`

这里的数据流：

- 输入：多个 `Trajectory`
- 输出：一个 `rollout_batch`

也就是说，actor 不直接拿单个 trajectory 训练，而是先把多个 trajectory 合成一个标准 batch。

### 2. `Trajectory` 先被拼成统一 batch

看：

- `src/rlinf/data/embodied_io_struct.py:733`

函数：

- `convert_trajectories_to_batch(...)`

它做的事很直接：

- 把多个 `Trajectory` 沿 batch 维拼起来
- 输出一个 dict
- 每个字段的形状统一成 `[T, B, ...]`

这里最重要的字段会进入 `rollout_batch`：

- `actions`
- `rewards`
- `dones`
- `prev_logprobs`
- `prev_values`
- `versions`
- `forward_inputs`
- `curr_obs`
- `next_obs`

所以这一步之后，actor 拿到的是一个“时间维 + batch 维”都整好的训练批。

### 3. actor 会先把 batch 改成 advantage 计算更方便的形状

看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py:1132`

函数：

- `_process_received_rollout_batch(...)`

关键代码：

- 第 `1136-1137` 行注释：
  - 原始 shape：`[rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]`
  - 目标 shape：`[n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]`
- 第 `1139-1140` 行：`process_nested_dict_for_adv(...)`

这一步本质上是在重排维度，让后面做 GAE 更顺手。

#### 3.1 还可能生成 `loss_mask`

看：

- 第 `1142-1156` 行

如果：

- 不是 auto reset
- 也不是 ignore terminations

那它会根据 `dones` 算：

- `loss_mask`
- `loss_mask_sum`

这个 mask 后面会控制：

- 哪些时间步有效
- PPO loss 和 value loss 在哪里计算

### 4. 接着开始算 advantage 和 return

看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py:1209`

函数：

- `compute_advantages_and_returns(...)`

关键代码：

- 第 `1213-1225` 行：组装参数
- 第 `1227` 行：`advantages_and_returns = calculate_adv_and_returns(**kwargs)`
- 第 `1229-1233` 行：把结果写回 `self.rollout_batch`

这里传进去的核心输入有：

- `rewards`
- `dones`
- `prev_values`
- `gamma`
- `gae_lambda`
- `loss_mask`

也就是说，advantage 不是凭空来的，而是由 rollout 时存下来的：

- reward
- done
- old value

一起算出来的。

#### 4.1 真正分发到哪种 advantage 算法

看：

- `src/rlinf/algorithms/registry.py:95`

关键代码：

- 第 `101` 行：`adv_type = kwargs["adv_type"]`
- 第 `102` 行：`fn = get_adv_and_returns(adv_type)`
- 第 `105-112` 行：如果是 `embodied`，先做 embodied 的预处理，再调对应算法

你当前这条主线里，默认是：

- `adv_type = "gae"`

所以会走：

- `src/rlinf/algorithms/advantages.py:24`

#### 4.2 GAE 具体怎么算

看：

- `src/rlinf/algorithms/advantages.py:24`

关键代码：

- 第 `56` 行：`T = rewards.shape[0]`
- 第 `66` 行：倒序遍历时间步
- 第 `70-74` 行：算 TD 误差 `delta`
- 第 `76` 行：递推 `gae`
- 第 `77` 行：算 `returns`
- 第 `79` 行：算 `advantages`
- 第 `81-84` 行：可选归一化

输入输出：

- 输入：
  - `rewards`
  - `values`
  - `dones`
  - `gamma`
  - `gae_lambda`
- 输出：
  - `advantages`
  - `returns`

所以这一步结束后，`self.rollout_batch` 里至少会新增：

- `advantages`
- `returns`

### 5. 然后才真正进入 `run_training()`

看：

- `src/rlinf/workers/actor/fsdp_actor_worker.py:1326`

函数：

- `run_training(...)`

这一段就是 PPO 更新主循环。

### 6. 先把 rollout batch 打乱并切成训练 batch

关键代码：

- 第 `1336-1342` 行：算 `rollout_size`，并生成 `shuffle_id`
- 第 `1344-1347` 行：`process_nested_dict_for_train(...)`

这里的作用是：

- 把 rollout 数据打乱
- 变成适合训练的平铺形式

#### 6.1 算 gradient accumulation

- 第 `1349-1358` 行

这里会检查：

- `global_batch_size % (micro_batch_size * world_size) == 0`

然后算：

- `self.gradient_accumulation`

也就是每次 optimizer step 之前，要累计多少个 micro batch。

#### 6.2 再切成 global batch 和 micro batch

看：

- 第 `1363-1374` 行：把整个 rollout batch 切成多个 `train_global_batch`
- 第 `1377-1390` 行：再把每个 `train_global_batch` 切成多个 `train_micro_batch`

所以训练时的层次是：

`rollout_batch -> train_global_batch -> train_micro_batch`

### 7. 每个 micro batch 里，最关键的是重新前向

看：

- 第 `1402-1409` 行：从 batch 里取：
  - `advantages`
  - `prev_logprobs`
  - `returns`
  - `prev_values`
  - `loss_mask`
  - `forward_inputs`

注意这里最关键的是：

- 训练时并不是直接用 rollout 里的 `logprobs`
- 而是重新把 `forward_inputs` 喂给当前 actor 模型，再算一遍新的 `logprobs / values`

真正的前向在：

- 第 `1431-1438` 行：

```python
output_dict = self.model(
    forward_inputs=forward_inputs,
    compute_logprobs=True,
    compute_entropy=...,
    compute_values=compute_values,
    use_cache=False,
)
```

这里的输出至少包括：

- `output_dict["logprobs"]`
- `output_dict["values"]`
- `output_dict["entropy"]`

所以这一步的本质是：

- rollout 时保存的是旧策略信息
- training 时 actor 用当前参数重新算新策略信息

然后 PPO 才能比较“新旧策略差了多少”。

### 8. 为什么 `forward_inputs` 这么重要

因为 rollout 时环境已经往前走了，训练时不能再重新跑一遍环境。

所以 rollout 必须把“重新前向所需的最小上下文”存下来。

前面第二块里 `OpenPIActionModel.predict_action_batch(...)` 已经把这些塞进了 `forward_inputs`：

- `chains`
- `denoise_inds`
- `tokenized_prompt`
- `tokenized_prompt_mask`
- `action`
- `model_action`
- 观测副本

所以 actor 训练时不用再访问 env，也不用重新推理一次 rollout，只需要拿这份 `forward_inputs` 就能复算 policy/value。

### 9. 然后开始算 PPO loss

看：

- 第 `1446-1467` 行：组装 `policy_loss(**kwargs)` 的参数
- 第 `1468` 行：`loss, metrics_data = policy_loss(**kwargs)`

传进去的核心内容是：

- `logprobs`
  当前 actor 刚重新算出来的
- `values`
  当前 actor 刚重新算出来的
- `old_logprobs`
  rollout 时的旧 logprob
- `advantages`
  刚算出来的 GAE advantage
- `returns`
  刚算出来的目标 return
- `prev_values`
  rollout 时的旧 value

也就是说，PPO loss 的关键对比关系是：

- 新 logprob vs 旧 logprob
- 新 value vs return / 旧 value

#### 9.1 `policy_loss` 会先按 `loss_type` 分发

看：

- `src/rlinf/algorithms/registry.py:80`

关键代码：

- 第 `81` 行：`loss_type = kwargs["loss_type"]`
- 第 `82` 行：`loss_fn = get_policy_loss(loss_type)`
- 第 `88` 行：`loss, metrics_data = loss_fn(**kwargs)`

你当前 smoke config 里，sync PPO 走的是：

- `loss_type: actor_critic`

所以会走：

- `src/rlinf/algorithms/losses.py:403`

### 10. `actor_critic` loss 是怎么拼的

看：

- `src/rlinf/algorithms/losses.py:403`

关键代码：

- 第 `424` 行：`actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)`
- 第 `425` 行：`critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)`
- 第 `427` 行：`loss = actor_loss + critic_loss`

所以 sync PPO 当前主线的总损失就是：

- `总 loss = actor loss + critic loss`

### 11. actor loss 具体怎么算

看：

- `src/rlinf/algorithms/losses.py:167`

关键代码：

- 第 `241` 行：`log_ratio = logprobs - old_logprobs`
- 第 `246` 行：`ratio = exp(log_ratio)`
- 第 `249` 行：`clipped_ratio = clamp(ratio, 1-low, 1+high)`
- 第 `250-251` 行：
  - `policy_loss1 = -advantages * ratio`
  - `policy_loss2 = -advantages * clipped_ratio`
- 第 `255` 行：`policy_loss = max(policy_loss1, policy_loss2)`
- 第 `267-269` 行：对 mask 后的 loss 做聚合

这就是标准 PPO 的 clipping 逻辑：

- 如果新策略偏得太远，就用裁剪后的 ratio
- 防止更新过猛

### 12. critic loss 具体怎么算

看：

- `src/rlinf/algorithms/losses.py:312`

关键代码：

- 第 `347-349` 行：先构造 `value_pred_clipped`
- 第 `351-356` 行：分别算原始和 clipped 的 value loss
- 第 `357-358` 行：取两者较大值，再做 mask 聚合

也就是说，critic 这边也用了 PPO 风格的 value clipping。

### 13. 如果开了 entropy bonus，还会额外减一个 entropy 项

回到：

- `src/rlinf/workers/actor/fsdp_actor_worker.py:1470`

关键代码：

- 第 `1473-1475` 行：如果 `entropy_bonus > 0`
- 第 `1477` 行：取 `output_dict["entropy"]`
- 第 `1484` 行：`entropy_loss = masked_mean(...)`
- 第 `1485` 行：`loss -= entropy_bonus * entropy_loss`

也就是说最终总损失还可能是：

- `actor_loss + critic_loss - entropy_bonus * entropy`

### 14. 最后才反向传播和优化器更新

回到 `run_training(...)`：

- 第 `1491` 行：`loss /= self.gradient_accumulation`
- 第 `1493` 行：`self.grad_scaler.scale(loss).backward()`

这一步只是对当前 micro batch 做反向传播，还没 step optimizer。

真正的参数更新在：

- 第 `1500` 行：`grad_norm, lr_list = self.optimizer_step()`

也就是说顺序是：

1. 每个 micro batch forward
2. 算 loss
3. backward
4. 累积够了以后 optimizer step

### 15. 训练完这一轮，actor 返回 metrics，下一轮再同步权重给 rollout

最后几行：

- 第 `1509` 行：`self.lr_scheduler.step()`
- 第 `1512-1515` 行：聚合 metrics
- 第 `1517` 行：`return mean_metric_dict`

然后回到 runner：

- `src/rlinf/runners/embodied_runner.py:277`

下一轮开始前，如果到了同步间隔：

- 第 `278-279` 行：`self.update_rollout_weights()`

所以完整闭环是：

1. rollout 用旧权重和 env 交互
2. actor 用 trajectory 做 PPO 更新
3. actor 参数更新完成
4. runner 把新权重同步给 rollout
5. rollout 再用新权重采样下一轮数据

#### 15.1 为什么是“同步给 rollout”，不是“同步给 env”

这里最容易混的是：

- `actor`
- `rollout`
- `VLA/pi0.5`

它们不是并列关系。

更准确地说：

- `actor worker` 里有一份 **训练版** `pi0.5/openpi`
- `rollout worker` 里也有一份 **推理版** `pi0.5/openpi`
- `env worker` 不持有 policy 权重，它只负责执行动作、返回观测和奖励

所以：

- 真正修改 VLA 权重的是 `actor`
- 真正拿 VLA 去和环境交互、产出 action 的是 `rollout`

也就是说，`rollout` 不是“另一个和 VLA 平级的模块”，而是：

- 一个 worker/进程角色
- 里面装着一份当前 policy 副本

因此第 `4` 步说的“把新权重同步给 rollout”，本质上就是：

- 把 `actor` 里训练好的最新 `pi0.5` 权重
- 拷贝到 `rollout worker` 里那份 `pi0.5` 上

这样下一轮采样时，rollout 才会用最新策略和环境交互。

如果不同步，会变成：

- actor 已经训练到新权重
- rollout 还在拿旧权重采样

那下一批数据就不再是“当前策略”对应的 on-policy 数据了。

### 16. 这一块结束时，你应该怎么理解“训练”

这里的训练不是：

- “环境里边走一步边更新一步”

而是：

1. 先由 env + rollout 生成一整批 trajectory
2. actor 收到 trajectory 后统一整理
3. 算 advantage / return
4. 用 rollout 保存的 `forward_inputs` 重新前向
5. 用 PPO loss 更新参数
6. 再把新权重发回 rollout

所以这条 sync PPO 主线更像：

`collect rollout -> build batch -> compute adv -> train actor -> sync weights -> next rollout`

### 17. 这一块最关键的三个对象

如果你现在只想记住最关键的训练对象，就记这三个：

- `Trajectory`
  rollout 结束后发给 actor 的整段训练样本
- `rollout_batch`
  actor 端整理好的 `[T, B, ...]` 训练 batch
- `forward_inputs`
  actor 重新前向、复算 logprob/value 的核心上下文


