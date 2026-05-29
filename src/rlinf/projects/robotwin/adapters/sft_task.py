from __future__ import annotations

import json
from typing import Any

from rlinf.projects.robotwin.training.sft.args import Args


def _impl():
    from rlinf.projects.robotwin.adapters import _sft_impl

    return _sft_impl


class RobotwinSFTTaskSpec:
    """RoboTwin/pi0.5 hooks for the generic SFT training loop."""

    name = "train_robotwin_vla"

    def init_runtime(self, args: Args) -> None:
        del args
        from rlinf.projects.robotwin.runtime.env import ensure_torch_transformers_runtime

        ensure_torch_transformers_runtime()

    def build_dataloader(self, args: Args, dist_ctx: Any) -> tuple[Any, Any]:
        return _impl()._build_loader(args, dist_ctx)

    def build_model(self, args: Args, device: Any, dist_ctx: Any) -> Any:
        return _impl()._build_model(args, device, dist_ctx)

    def on_after_dataloader_built(
        self,
        args: Args,
        train_config: Any,
        data_loader: Any,
        dist_ctx: Any,
    ) -> dict[str, Any]:
        impl = _impl()
        rollout_probe_samples = impl._build_rollout_probe_samples(
            args,
            action_horizon=train_config.model.action_horizon,
        )
        if dist_ctx.is_main and rollout_probe_samples:
            probe_preview = [
                {
                    "repo_id": sample.repo_id,
                    "dataset_item": sample.dataset_item,
                    "prompt": sample.prompt,
                }
                for sample in rollout_probe_samples[
                    : min(3, len(rollout_probe_samples))
                ]
            ]
            print(
                "[train_robotwin_vla] "
                f"rollout_probe_num_samples={len(rollout_probe_samples)} "
                f"rollout_probe_every={args.action_probe_every} "
                f"rollout_probe_preview={json.dumps(probe_preview, ensure_ascii=False)}"
            )
        return {"rollout_probe_samples": rollout_probe_samples}

    def on_after_model_built(
        self,
        args: Args,
        model: Any,
        dist_ctx: Any,
    ) -> None:
        if not dist_ctx.is_main:
            return
        param_counts = _impl()._count_parameters(model)
        print(
            "[train_robotwin_vla] "
            f"is_lora={args.is_lora} "
            f"lora_rank={args.lora_rank} "
            f"lora_path={args.lora_path} "
            f"total_params={param_counts['total']} "
            f"trainable_params={param_counts['trainable']} "
            f"lora_params={param_counts['lora']} "
            f"trainable_lora_params={param_counts['trainable_lora']} "
            f"trainable_ratio={param_counts['trainable_ratio']:.6f}",
            flush=True,
        )

    def build_optimizer(self, args: Args, model: Any) -> Any:
        import torch

        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            eps=1.0e-8,
        )

    def move_batch_to_device(self, args: Args, batch: Any, device: Any) -> Any:
        del args
        import torch

        observation, actions = batch
        observation = _impl()._move_observation_to_device(observation, device)
        actions = actions.to(torch.float32).to(device, non_blocking=True)
        return observation, actions

    def forward_loss(
        self,
        args: Args,
        model: Any,
        batch: Any,
        device: Any,
    ) -> Any:
        del args
        impl = _impl()
        from rlinf.training.sft.losses import loss_from_output

        observation, actions = batch
        losses = model(
            data={"observation": observation, "actions": actions},
            forward_type=impl.ForwardType.SFT,
        )
        return loss_from_output(losses, device)

    def on_step_end(
        self,
        args: Args,
        step: int,
        model: Any,
        data_loader: Any,
        device: Any,
        dist_ctx: Any,
        wandb_logger: Any,
        callback_state: Any,
    ) -> dict[str, float]:
        del data_loader, wandb_logger
        rollout_probe_samples = callback_state.get("rollout_probe_samples", [])
        should_probe_actions = (
            dist_ctx.is_main
            and bool(rollout_probe_samples)
            and args.action_probe_every > 0
            and (
                step == 1
                or step % args.action_probe_every == 0
                or step == args.train_steps
            )
        )
        if not should_probe_actions:
            return {}

        impl = _impl()
        impl._log_stage(
            f"step {step}: rollout probe start",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=False,
        )
        rollout_probe_metrics = impl._compute_rollout_probe_metrics(
            model,
            rollout_probe_samples,
            device,
        )
        impl._log_stage(
            f"step {step}: rollout probe done mae="
            f"{rollout_probe_metrics.get('probe/raw_train_rollout_action_mae', float('nan')):.6f}",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=False,
        )
        return rollout_probe_metrics

    def save_checkpoint(
        self,
        args: Args,
        output_dir: Any,
        model: Any,
        optimizer: Any,
        step: int,
        data_loader: Any,
        dist_ctx: Any,
    ) -> None:
        _impl()._save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            step=step,
            args=args,
            data_config=data_loader.data_config(),
            dist_ctx=dist_ctx,
        )

    def should_eval(self, args: Args, step: int) -> bool:
        return args.eval_every > 0 and (
            step % args.eval_every == 0 or step == args.train_steps
        )

    def export_eval_model_dir(
        self,
        args: Args,
        output_dir: Any,
        model: Any,
        step: int,
        data_loader: Any,
        dist_ctx: Any,
    ) -> Any:
        return _impl()._export_eval_model_dir(
            eval_dir=output_dir / "_eval_runtime" / "latest",
            model=model,
            step=step,
            args=args,
            data_config=data_loader.data_config(),
            dist_ctx=dist_ctx,
        )

    def run_eval(
        self,
        args: Args,
        step: int,
        model_dir: Any,
        output_dir: Any,
        dist_ctx: Any,
        wandb_logger: Any,
    ) -> Any:
        return _impl()._run_periodic_eval(
            args=args,
            step=step,
            model_dir=model_dir,
            output_root=output_dir,
            dist_ctx=dist_ctx,
            wandb_logger=wandb_logger,
        )


__all__ = ["RobotwinSFTTaskSpec"]
