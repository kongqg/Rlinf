from __future__ import annotations

import math


def cosine_learning_rate_for_update(args, update_step: int) -> float:
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}.")
    if args.min_learning_rate < 0:
        raise ValueError(
            f"min_learning_rate must be non-negative, got {args.min_learning_rate}."
        )
    if args.min_learning_rate > args.learning_rate:
        raise ValueError(
            "min_learning_rate must be <= learning_rate, "
            f"but got min_learning_rate={args.min_learning_rate} > learning_rate={args.learning_rate}."
        )
    if args.lr_scheduler == "none":
        return args.learning_rate
    if args.lr_scheduler != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    total_steps = max(int(args.lr_total_steps), 1)
    warmup_steps = max(min(int(args.lr_warmup_steps), total_steps), 0)
    step = min(max(int(update_step), 1), total_steps)
    if warmup_steps > 0 and step <= warmup_steps:
        return args.learning_rate * (step / warmup_steps)
    if total_steps == warmup_steps:
        return args.learning_rate
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * 2.0 * args.lr_num_cycles * progress))
    return args.min_learning_rate + (
        args.learning_rate - args.min_learning_rate
    ) * cosine_decay
