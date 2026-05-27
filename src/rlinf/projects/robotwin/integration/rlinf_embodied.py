from __future__ import annotations

import json

from omegaconf import DictConfig, OmegaConf

from rlinf.config import validate_cfg as validate_rlinf_cfg
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _print_resolved_cfg(cfg: DictConfig) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))


def _maybe_create_reward_group(cfg: DictConfig, cluster, component_placement):
    if not cfg.get("reward", {}).get("use_reward_model", False):
        return None
    if cfg.get("reward", {}).get("standalone_realworld", False):
        return None

    reward_placement = component_placement.get_strategy("reward")
    return EmbodiedRewardWorker.create_group(cfg).launch(
        cluster,
        name=cfg.reward.group_name,
        placement_strategy=reward_placement,
    )


def _create_sync_actor_group(cfg: DictConfig, cluster, component_placement):
    actor_placement = component_placement.get_strategy("actor")
    loss_type = cfg.algorithm.loss_type

    if loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif loss_type == "actor_critic":
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor
    else:
        raise ValueError(
            f"Unsupported sync embodied loss_type {loss_type!r} in the local vendor. "
            "Supported values are ['actor_critic', 'embodied_sac']."
        )

    return actor_worker_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )


def _create_async_actor_group_and_runner_cls(
    cfg: DictConfig, cluster, component_placement
):
    actor_placement = component_placement.get_strategy("actor")
    loss_type = cfg.algorithm.loss_type

    if loss_type == "embodied_sac":
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
            AsyncEmbodiedSACFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedSACFSDPPolicy
    elif loss_type == "decoupled_actor_critic":
        from rlinf.workers.actor.async_ppo_fsdp_worker import (
            AsyncPPOEmbodiedFSDPActor,
        )

        runner_cls = AsyncPPOEmbodiedRunner
        actor_worker_cls = AsyncPPOEmbodiedFSDPActor
    else:
        raise ValueError(
            f"Unsupported async embodied loss_type {loss_type!r} in the local vendor. "
            "Supported values are ['decoupled_actor_critic', 'embodied_sac']."
        )

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )
    return actor_group, runner_cls


def run_sync_embodied_training(cfg: DictConfig) -> None:
    #
    cfg = validate_rlinf_cfg(cfg)
    _print_resolved_cfg(cfg)

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_group = _create_sync_actor_group(cfg, cluster, component_placement)

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement,
    )

    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )

    reward_group = _maybe_create_reward_group(cfg, cluster, component_placement)

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )
    runner.init_workers()
    runner.run()


def run_async_embodied_training(cfg: DictConfig) -> None:
    cfg = validate_rlinf_cfg(cfg)
    _print_resolved_cfg(cfg)

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_group, runner_cls = _create_async_actor_group_and_runner_cls(
        cfg, cluster, component_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement,
    )

    env_placement = component_placement.get_strategy("env")
    env_group = AsyncEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )

    reward_group = _maybe_create_reward_group(cfg, cluster, component_placement)

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )
    runner.init_workers()
    runner.run()
