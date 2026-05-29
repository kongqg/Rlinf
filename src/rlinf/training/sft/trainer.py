from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from rlinf.training.common.device import device_for_training
from rlinf.training.common.distributed import (
    destroy_distributed,
    init_distributed,
    reduce_mean,
)
from rlinf.training.common.logging import WandbLogger, log_stage
from rlinf.training.common.lr_scheduler import cosine_learning_rate_for_update
from rlinf.training.common.seed import seed_everything
from rlinf.training.sft.interfaces import ArgsT, SFTTaskSpec


def _init_wandb_logger(
    args: Any,
    dist_ctx: Any,
    output_dir: Path,
) -> WandbLogger | None:
    if (
        not getattr(args, "wandb_enabled", False)
        or getattr(args, "wandb_mode", "online") == "disabled"
        or not dist_ctx.is_main
    ):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in the current environment. "
            "Install it first or run with --wandb-enabled False."
        ) from exc

    settings = None
    if getattr(args, "wandb_proxy", None):
        settings = wandb.Settings(https_proxy=args.wandb_proxy)

    wandb_log_dir = (
        Path(args.wandb_log_dir).expanduser().resolve()
        if getattr(args, "wandb_log_dir", None) is not None
        else output_dir / "wandb"
    )
    wandb_log_dir.mkdir(parents=True, exist_ok=True)
    run_name = getattr(args, "wandb_run_name", None) or output_dir.name
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=asdict(args),
        settings=settings,
        dir=str(wandb_log_dir),
        group=getattr(args, "wandb_group", None),
        tags=list(getattr(args, "wandb_tags", ())),
        mode=args.wandb_mode,
        reinit=True,
    )
    return WandbLogger(run)


def run_sft_training(args: ArgsT, task: SFTTaskSpec[ArgsT]) -> None:
    """Run the generic supervised fine-tuning loop through project hooks."""

    task.init_runtime(args)

    import torch
    import torch.distributed as dist
    from rlinf.training.common.optim import set_optimizer_learning_rate

    prefix = task.name
    dist_ctx = init_distributed(args)
    log_stage(
        f"initialized distributed backend={dist_ctx.backend} world_size={dist_ctx.world_size} local_rank={dist_ctx.local_rank}",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
        prefix=prefix,
    )
    seed_everything(args.seed)
    device = device_for_training(args.device, dist_ctx)
    log_stage(
        f"using device={device}",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
        prefix=prefix,
    )

    log_stage(
        "building data loader",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
        prefix=prefix,
    )
    train_config, data_loader = task.build_dataloader(args, dist_ctx)
    log_stage(
        "data loader ready",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
        prefix=prefix,
    )
    callback_state = task.on_after_dataloader_built(
        args,
        train_config,
        data_loader,
        dist_ctx,
    )

    log_stage(
        "building model",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
        prefix=prefix,
    )
    model = task.build_model(args, device, dist_ctx)
    log_stage(
        "model ready",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
        prefix=prefix,
    )
    task.on_after_model_built(args, model, dist_ctx)

    optimizer = task.build_optimizer(args, model)
    optimizer_updates = 0
    set_optimizer_learning_rate(
        optimizer,
        cosine_learning_rate_for_update(args, update_step=1),
    )

    amp_enabled = device.type == "cuda"
    loader_iter = iter(data_loader)
    optimizer.zero_grad(set_to_none=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    wandb_logger = _init_wandb_logger(args, dist_ctx, output_dir)
    if args.eval_every > 0 and getattr(args, "assets_path", None) is None:
        raise ValueError("assets_path must be set when eval_every > 0.")

    failure: BaseException | None = None
    try:
        for step in range(1, args.train_steps + 1):
            log_stage(
                f"step {step}: fetching batch",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
                prefix=prefix,
            )
            batch = next(loader_iter)
            log_stage(
                f"step {step}: batch fetched",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
                prefix=prefix,
            )
            batch = task.move_batch_to_device(args, batch, device)

            log_stage(
                f"step {step}: forward start",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
                prefix=prefix,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                loss = task.forward_loss(args, model, batch, device)
            log_stage(
                f"step {step}: forward done loss={loss.detach().item():.6f}",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
                prefix=prefix,
            )

            (loss / args.grad_accum_steps).backward()
            log_stage(
                f"step {step}: backward done",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
                prefix=prefix,
            )

            should_step = (step % args.grad_accum_steps == 0) or (
                step == args.train_steps
            )
            if should_step:
                log_stage(
                    f"step {step}: optimizer step start",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                    prefix=prefix,
                )
                current_update = optimizer_updates + 1
                set_optimizer_learning_rate(
                    optimizer,
                    cosine_learning_rate_for_update(args, update_step=current_update),
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_grad_norm,
                )
                optimizer.step()
                optimizer_updates = current_update
                optimizer.zero_grad(set_to_none=True)
                log_stage(
                    f"step {step}: optimizer step done",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                    prefix=prefix,
                )
            else:
                grad_norm = None

            extra_metrics = task.on_step_end(
                args,
                step,
                model,
                data_loader,
                device,
                dist_ctx,
                wandb_logger,
                callback_state,
            )

            if step % args.log_every == 0 or step == 1:
                reduced_loss = reduce_mean(loss.detach(), dist_ctx)
                train_log_payload: dict[str, float] = {
                    "train/loss": float(reduced_loss),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                train_log_payload.update(extra_metrics)
                grad_text = "accumulating"
                if grad_norm is not None:
                    reduced_grad = reduce_mean(grad_norm.detach().float(), dist_ctx)
                    grad_text = f"{float(reduced_grad):.4f}"
                    train_log_payload["train/grad_norm"] = float(reduced_grad)
                probe_text = ""
                if "probe/raw_train_rollout_action_mae" in extra_metrics:
                    probe_text = (
                        " "
                        f"rollout_action_mae="
                        f"{extra_metrics['probe/raw_train_rollout_action_mae']:.6f}"
                    )
                if dist_ctx.is_main:
                    print(
                        f"[{prefix}] step={step}/{args.train_steps} "
                        f"loss={float(reduced_loss):.6f} grad_norm={grad_text}{probe_text}"
                    )
                    if wandb_logger is not None:
                        wandb_logger.log(train_log_payload, step=step)

            checkpoint_for_eval: Path | None = None
            if step % args.save_every == 0 or step == args.train_steps:
                log_stage(
                    f"step {step}: checkpoint save start",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                    prefix=prefix,
                )
                task.save_checkpoint(
                    args,
                    output_dir,
                    model,
                    optimizer,
                    step,
                    data_loader,
                    dist_ctx,
                )
                log_stage(
                    f"step {step}: checkpoint save done",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                    prefix=prefix,
                )
                if dist_ctx.is_main:
                    print(f"[{prefix}] saved checkpoint to {output_dir}")
                checkpoint_for_eval = output_dir

            if task.should_eval(args, step):
                if checkpoint_for_eval is None:
                    checkpoint_for_eval = task.export_eval_model_dir(
                        args,
                        output_dir,
                        model,
                        step,
                        data_loader,
                        dist_ctx,
                    )
                task.run_eval(
                    args,
                    step,
                    checkpoint_for_eval or (output_dir / "_eval_runtime" / "latest"),
                    output_dir,
                    dist_ctx,
                    wandb_logger,
                )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
        log_stage(
            "training shutdown",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
            prefix=prefix,
        )
        if dist_ctx.is_distributed and dist.is_initialized():
            destroy_distributed(dist_ctx, success=failure is None)
