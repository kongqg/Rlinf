# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor-shape adapters shared by RL losses and advantage estimators.

Most algorithm implementations expect a compact mathematical layout such as
``[time, batch]`` or ``[batch, chunk]``.  Real embodied and reasoning batches use
slightly different conventions, so this module centralizes the reshape,
transpose, score aggregation, and logging post-processing steps.
"""

from typing import Optional

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber penalty used by value losses to be quadratic near zero and linear for outliers."""
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        # k1 is the signed log-ratio estimator; callers decide how to reduce it.
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # Clamp the exponent input so a very bad policy/reference mismatch does
        # not create inf values before the trainer can clip or log the update.
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    if kwargs["reward_type"] == "chunk_level":
        # Chunk-level rewards collapse the action dimension before temporal
        # advantage computation, because downstream estimators work on one scalar
        # reward per environment step.
        # rewards, dones, loss_mask, loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] -> [n_chunk_steps, bsz, 1]
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        # GAE needs V_t and V_{t+1}; keep the extra bootstrap value after
        # flattening the chunked temporal layout.
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    """Accumulate discounted-free episodic scores for group-relative methods."""
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        # Reset the running suffix sum whenever the following timestep starts a
        # new episode, then add the reward at the current timestep.
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    res = {}

    # Reverse preprocess_embodied_advantages_inputs: [T, B] -> [chunk, B, chunk_size].
    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    # NOTE: to align with embodied inputs, we transpose loss mask and rewards when needed.

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        # Sparse final rewards are expanded to token time so GAE can share the
        # same [seq_len, batch] interface as embodied environments.
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # only last token has reward
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "grpo_dynamic":
        # Dynamic GRPO keeps one reward per flattened turn/sequence, then the
        # estimator uses idx_to_traj to recover trajectory/question groupings.
        grouped_rewards = (
            rewards.reshape(-1, kwargs["num_sequence"]).transpose(0, 1).contiguous()
        )
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    elif kwargs["adv_type"] == "raw":
        kwargs.update({"rewards": rewards})

    else:
        assert False, f"Unsupported adv_type {kwargs['adv_type']}"

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # Create done flags (episode ends at the last token)
    dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool, device=rewards.device)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """
    Post-process results for Reasoning tasks; transpose tensors back.
    """

    # remember to call contiguous() to ensure correctness when being
    # transmitted through channels
    advantages = advantages.transpose(0, 1).contiguous()  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1).contiguous()  # [bsz, seq_len]

    return advantages, returns


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    versions: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """Reshape embodied actor-loss tensors to the configured log-prob granularity."""
    if reward_type == "chunk_level":
        # Chunk-level reward mode treats the whole action chunk as one training
        # example, so auxiliary tensors must be flattened to match logprob layout.
        advantages = advantages.flatten()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    proximal_logprobs = kwargs.get("proximal_logprobs", None)
    if logprob_type == "token_level":
        # Keep one log-prob per action dimension; PPO clipping is applied at the
        # finest granularity.
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # Sum across action dimensions so each chunk step has one joint-action log-prob.
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(
                bsz, -1, single_action_dim
            ).sum(dim=-1)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[..., 0]

    elif logprob_type == "chunk_level":
        # Sum all action dimensions and chunk steps so one trajectory chunk has
        # one scalar log-prob for PPO/GRPO ratio computation.
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(
                bsz, -1, single_action_dim
            ).sum(dim=[1, 2])
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[:, 0, 0]

    target_shape = logprobs.shape
    # Broadcast scalar/step-level annotations to the final loss tensor shape.
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)
    versions = expand_to_target_dim(versions, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "proximal_logprobs": proximal_logprobs,
            "versions": versions,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    """Detach tensor metrics so loggers receive plain Python numbers."""
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def expand_to_target_dim(tensor, target_shape):
    """Add trailing singleton dimensions until ``tensor`` can broadcast to target_shape."""
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array, loss_mask):
    """Normalize only valid entries while leaving invalid masked positions harmless."""
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
