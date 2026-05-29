from __future__ import annotations

from typing import Any, Protocol, TypeVar


ArgsT = TypeVar("ArgsT")


class SFTTaskSpec(Protocol[ArgsT]):
    """Project adapter hooks for the generic SFT training loop."""

    name: str

    def init_runtime(self, args: ArgsT) -> None:
        """Prepare project-specific runtime state before heavy imports."""

    def build_dataloader(self, args: ArgsT, dist_ctx: Any) -> tuple[Any, Any]:
        """Build and return the project train config plus dataloader."""

    def build_model(self, args: ArgsT, device: Any, dist_ctx: Any) -> Any:
        """Build the trainable model."""

    def on_after_dataloader_built(
        self,
        args: ArgsT,
        train_config: Any,
        data_loader: Any,
        dist_ctx: Any,
    ) -> Any:
        """Create optional callback state after data construction."""

    def on_after_model_built(
        self,
        args: ArgsT,
        model: Any,
        dist_ctx: Any,
    ) -> None:
        """Run optional project-specific model logging."""

    def build_optimizer(self, args: ArgsT, model: Any) -> Any:
        """Build the optimizer for trainable model parameters."""

    def move_batch_to_device(self, args: ArgsT, batch: Any, device: Any) -> Any:
        """Move one dataloader batch to the training device."""

    def forward_loss(
        self,
        args: ArgsT,
        model: Any,
        batch: Any,
        device: Any,
    ) -> Any:
        """Compute the scalar loss for one moved batch."""

    def on_step_end(
        self,
        args: ArgsT,
        step: int,
        model: Any,
        data_loader: Any,
        device: Any,
        dist_ctx: Any,
        wandb_logger: Any,
        callback_state: Any,
    ) -> dict[str, float]:
        """Return optional metrics to merge into the training log payload."""

    def save_checkpoint(
        self,
        args: ArgsT,
        output_dir: Any,
        model: Any,
        optimizer: Any,
        step: int,
        data_loader: Any,
        dist_ctx: Any,
    ) -> None:
        """Save a checkpoint using the project checkpoint format."""

    def should_eval(self, args: ArgsT, step: int) -> bool:
        """Return whether eval should run at this training step."""

    def export_eval_model_dir(
        self,
        args: ArgsT,
        output_dir: Any,
        model: Any,
        step: int,
        data_loader: Any,
        dist_ctx: Any,
    ) -> Any:
        """Export a temporary eval model directory when no checkpoint was saved."""

    def run_eval(
        self,
        args: ArgsT,
        step: int,
        model_dir: Any,
        output_dir: Any,
        dist_ctx: Any,
        wandb_logger: Any,
    ) -> Any:
        """Run project-specific evaluation."""
